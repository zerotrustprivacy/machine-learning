#  Python script for training an ML model using the processed data from Cloud Storage. This script runs as a training job 
import argparse
import pandas as pd
import os
import joblib # Or import pickle as joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Example model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input-data-path', type=str, required=True,
                    help='Cloud Storage or local path to the processed input data (CSV file).')
parser.add_argument('--model-output-dir', type=str, required=True,
                    help='Cloud Storage or local directory to save the trained model.')

args = parser.parse_args()

# --- 1. Load the processed data ---
print(f"Loading data from: {args.input_data_path}")
try:
    # pandas can read directly from gs:// paths if google-cloud-storage is installed
    processed_data_df = pd.read_csv(args.input_data_path)
    print("Data loaded successfully.")
    print(f"Data shape: {processed_data_df.shape}")
    print("Data columns:", processed_data_df.columns.tolist())
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- 2. Separate features (X) and target (y) ---
# IMPORTANT: Replace 'outcome' with the actual name of your target column
# IMPORTANT: List your actual feature columns or drop the target column
TARGET_COLUMN = 'outcome' # <--- REPLACE WITH YOUR TARGET COLUMN NAME

if TARGET_COLUMN not in processed_data_df.columns:
     print(f"Error: Target column '{TARGET_COLUMN}' not found in the data.")
     exit()

y = processed_data_df[TARGET_COLUMN]
X = processed_data_df.drop(columns=[TARGET_COLUMN])

# --- IMPORTANT: Apply any final preprocessing if needed ---
# If your Dataflow pipeline didn't do all preprocessing (e.g., one-hot encoding
# for the model), do it here. Ensure columns match what your chosen model expects.
# Example: X = pd.get_dummies(X, columns=['categorical_column'])


print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")


# --- 3. Split data into training and testing sets ---
# Using a fixed random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # stratify for classification

print(f"Train data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")


# --- 4. Choose and train an ML model ---
print("Training model...")
# You can replace RandomForestClassifier with another scikit-learn model
# or code for a TensorFlow/PyTorch model
model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)
print("Model training complete.")


# --- 5. Evaluate the model ---
print("Evaluating model...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred) # Adjust average if needed
recall = recall_score(y_test, y_pred)     # Adjust average if needed
f1 = f1_score(y_test, y_pred)           # Adjust average if needed

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("------------------------")


# --- 6. Save the trained model ---
# Vertex AI expects model artifacts to be saved to the GCS path provided
# in the model_output_dir argument.
model_filename = 'model.pkl' # Or 'model.joblib'
model_path = os.path.join(args.model_output_dir, model_filename)

print(f"Saving model to: {model_path}")

try:
    # Ensure the directory exists if saving locally during testing
    if not model_path.startswith('gs://'):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # joblib can save directly to gs:// paths
    joblib.dump(model, model_path)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")


print("\nTraining script finished.")
