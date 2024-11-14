import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# Function to load dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path, dtype={'algorithm': 'category', 'plaintext': 'string', 'ciphertext': 'string'})
    df.rename(columns={'Algorithm': 'algorithm', 'Plaintext': 'plaintext', 'Ciphertext': 'ciphertext'}, inplace=True)
    df.dropna(subset=['plaintext', 'ciphertext'], inplace=True)
    return df

# Feature Engineering: Convert plaintext and ciphertext to padded byte arrays
def convert_to_byte_array(text, max_len=256):
    byte_array = np.array([ord(c) for c in text[:max_len]], dtype=np.uint8)
    if len(byte_array) < max_len:
        # Padding with zeros if the length is less than max_len
        byte_array = np.pad(byte_array, (0, max_len - len(byte_array)), 'constant')
    return byte_array

def extract_features(df, max_len=256):
    plaintext_features = []
    ciphertext_features = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Features"):
        plaintext_features.append(convert_to_byte_array(row['plaintext'], max_len))
        ciphertext_features.append(convert_to_byte_array(row['ciphertext'], max_len))
    return np.array(plaintext_features), np.array(ciphertext_features)

# Load dataset and extract features
file_path = r"/content/symmetric_without_key_testing.csv"  # Change this path to your testing file
df = load_dataset(file_path)
plaintext_features, ciphertext_features = extract_features(df)

# Stack plaintext and ciphertext features as two channels for CNN input
X = np.stack((plaintext_features, ciphertext_features), axis=-1)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['algorithm'])
y = to_categorical(y)  # Convert labels to categorical format for Keras

# Split dataset (95% training, 5% testing) — using 100% for testing in this case
X_test = X  # Use the whole dataset as test data in this case
y_test = y

# Load the trained CNN model
cnn_model = load_model('cnn_model.h5')

# Evaluate the model on the test data
loss, accuracy = cnn_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# You can also make predictions using:
y_pred = cnn_model.predict(X_test)
print("Predictions:", y_pred[:10])  # Displaying the first 10 predictions

