import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from scipy.stats import entropy
import pickle

# Function to load dataset from a given path
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df.rename(columns={'Algorithm': 'algorithm', 'Plaintext': 'plaintext', 'Ciphertext': 'ciphertext', 'Key': 'key'}, inplace=True)
    df.dropna(subset=['plaintext', 'ciphertext', 'key'], inplace=True)
    return df

# Feature Engineering
def extract_features(plaintext, ciphertext, key):
    def compute_features(data):
        byte_array = np.array(list(data.encode()), dtype=np.uint8)
        mean = np.mean(byte_array)
        variance = np.var(byte_array)
        entropy_value = entropy(np.bincount(byte_array, minlength=256) / len(byte_array))
        return [mean, variance, entropy_value]

    # Extract features for plaintext, ciphertext, and key
    plaintext_features = compute_features(plaintext)
    ciphertext_features = compute_features(ciphertext)
    key_features = compute_features(key)

    # Combine features into a single feature vector
    return plaintext_features + ciphertext_features + key_features

# Load training data
training_file_path = r'/content/symmetric_with_key1.csv'  # Replace with your training dataset path
df_train = load_dataset(training_file_path)

# Apply feature extraction
features_train = np.array([extract_features(row['plaintext'], row['ciphertext'], row['key']) for _, row in df_train.iterrows()])
labels_train = df_train['algorithm']

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(labels_train)

# Train Gradient Boosting model
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(features_train, y_train)
print("Model trained successfully.")

# Predict on the training set for evaluation
y_pred_train = gb_model.predict(features_train)

# Calculate overall accuracy
overall_accuracy = accuracy_score(y_train, y_pred_train)
print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")

# Calculate accuracy for each algorithm (class) and print detailed report
class_report = classification_report(y_train, y_pred_train, target_names=label_encoder.classes_)
print("\nClassification Report:\n", class_report)

# Save the model to a pickle file
model_filename = 'gb_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(gb_model, file)
