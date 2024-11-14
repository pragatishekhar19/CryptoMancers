  import pandas as pd
  import numpy as np
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import LabelEncoder, StandardScaler
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
  from tensorflow.keras.utils import to_categorical
  from tqdm import tqdm
  from scipy.stats import entropy
  import pickle

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
  file_path = r"/symmetric_without_key.csv"
  df = load_dataset(file_path)
  plaintext_features, ciphertext_features = extract_features(df)

  # Stack plaintext and ciphertext features as two channels for CNN input
  X = np.stack((plaintext_features, ciphertext_features), axis=-1)

  # Encode labels
  label_encoder = LabelEncoder()
  y = label_encoder.fit_transform(df['algorithm'])
  y = to_categorical(y)  # Convert labels to categorical format for Keras

  # Split dataset (95% training, 5% testing)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

  # Define the CNN model
  def build_cnn_model(input_shape, num_classes):
      model = Sequential()
      model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
      model.add(MaxPooling1D(pool_size=2))
      model.add(Conv1D(128, kernel_size=3, activation='relu'))
      model.add(MaxPooling1D(pool_size=2))
      model.add(Conv1D(256, kernel_size=3, activation='relu'))
      model.add(MaxPooling1D(pool_size=2))
      model.add(Flatten())
      model.add(Dense(128, activation='relu'))
      model.add(Dropout(0.5))
      model.add(Dense(num_classes, activation='softmax'))
      model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
      return model

  # Build and train the model
  input_shape = (X_train.shape[1], X_train.shape[2])  # Shape of the input data for CNN
  num_classes = y_train.shape[1]  # Number of classes in the target variable

  cnn_model = build_cnn_model(input_shape, num_classes)
  cnn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

  # Evaluate the model
  loss, accuracy = cnn_model.evaluate(X_test, y_test)
  print(f"CNN Model Test Accuracy: {accuracy:.4f}")

  # Save the CNN model
  cnn_model.save('cnn_model.h5')
  print("CNN model saved to cnn_model.h5.")
