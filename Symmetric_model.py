import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, MaxPooling1D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model
import pickle

# Function to load dataset from a given path
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df.rename(columns={'Algorithm': 'algorithm', 'Plaintext': 'plaintext', 'Ciphertext': 'ciphertext', 'Key': 'key'}, inplace=True)
    df.dropna(subset=['plaintext', 'ciphertext', 'key'], inplace=True)
    return df

# Preprocess data to fixed-length byte sequences
def preprocess_data(df, sequence_length=256):
    def to_byte_sequence(data, max_length=sequence_length):
        byte_array = np.array(list(data.encode()), dtype=np.uint8)
        padded_array = np.zeros(max_length, dtype=np.uint8)
        length = min(len(byte_array), max_length)
        padded_array[:length] = byte_array[:length]
        return padded_array

    # Apply the conversion to plaintext, ciphertext, and key columns
    plaintexts = np.array([to_byte_sequence(pt) for pt in df['plaintext']])
    ciphertexts = np.array([to_byte_sequence(ct) for ct in df['ciphertext']])
    keys = np.array([to_byte_sequence(k) for k in df['key']])

    # Stack to create a 3D input of shape (samples, sequence_length, channels)
    return np.stack([plaintexts, ciphertexts, keys], axis=-1)

# Load dataset and preprocess
file_path = r'D:/SIH/Symmetric model/symmetric_with_key.csv'  # Replace with your dataset path
df = load_dataset(file_path)
X = preprocess_data(df)
labels = df['algorithm']

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
y = to_categorical(y)  # One-hot encode for CNN

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN model
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    MaxPooling1D(pool_size=2),
    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')  # Output layer with softmax for multi-class classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=32, callbacks=[early_stopping])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the model and label encoder
model.save('cnn_model.h5')
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)
print("CNN model and label encoder saved.")
