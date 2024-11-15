import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from scipy.stats import entropy, skew, kurtosis
import pickle

# Function to load dataset from a given path
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df.rename(columns={'Algorithm': 'algorithm', 'Plaintext': 'plaintext', 'Ciphertext': 'ciphertext', 'Key': 'key'}, inplace=True)
    df.dropna(subset=['plaintext', 'ciphertext', 'key'], inplace=True)
    return df

# Feature Engineering with additional statistical features
def extract_features(plaintext, ciphertext, key):
    def compute_features(data):
        byte_array = np.array(list(data.encode()), dtype=np.uint8)
        mean = np.mean(byte_array)
        variance = np.var(byte_array)
        entropy_value = entropy(np.bincount(byte_array, minlength=256) / len(byte_array))
        skewness = skew(byte_array)
        kurt = kurtosis(byte_array)
        return [mean, variance, entropy_value, skewness, kurt]

    # Extract features for plaintext, ciphertext, and key
    plaintext_features = compute_features(plaintext)
    ciphertext_features = compute_features(ciphertext)
    key_features = compute_features(key)

    # Combine features into a single feature vector
    return plaintext_features + ciphertext_features + key_features

# Load and preprocess data
training_file_path = r'/content/symmetric_with_key1.csv'  # Replace with your dataset path
df_train = load_dataset(training_file_path)

# Feature extraction and label encoding
features_train = np.array([extract_features(row['plaintext'], row['ciphertext'], row['key']) for _, row in df_train.iterrows()])
labels_train = df_train['algorithm']

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(labels_train)

# Split data into training and validation sets for better evaluation
X_train, X_val, y_train, y_val = train_test_split(features_train, y_train, test_size=0.2, random_state=42)

# Gradient Boosting with hyperparameter tuning
gb_model = GradientBoostingClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(gb_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Best model from GridSearch
best_model = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate on validation data
y_pred_val = best_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred_val)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Detailed classification report for each class
class_report = classification_report(y_val, y_pred_val, target_names=label_encoder.classes_)
print("\nClassification Report on Validation Data:\n", class_report)

# Retrain best model on full training data (if necessary)
best_model.fit(features_train, label_encoder.fit_transform(labels_train))  # Use original labels here
y_pred_train = best_model.predict(features_train)
overall_accuracy = accuracy_score(label_encoder.fit_transform(labels_train), y_pred_train) #Use original labels here
print(f"Overall Training Accuracy: {overall_accuracy * 100:.2f}%")

# Save the trained model
model_filename = 'optimized_gb_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(best_model, file)

print("Optimized model saved successfully!")
