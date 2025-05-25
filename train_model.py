import argparse
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Example model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- MLflow Import ---
import mlflow
import mlflow.sklearn # This helps MLflow log scikit-learn models automatically

# Define command-line arguments (keep these from your existing script)
parser = argparse.ArgumentParser()
parser.add_argument('--input-data-path', type=str, required=True,
                    help='Path to the processed input data (CSV file).')
parser.add_argument('--model-output-dir', type=str, default='./mlruns/model_output',
                    help='Local directory to save the trained model for MLflow tracking.') # Default for local
parser.add_argument('--n_estimators', type=int, default=100,
                    help='Number of trees in the random forest.') # Example hyperparameter
parser.add_argument('--random_state', type=int, default=42,
                    help='Random state for reproducibility.') # Example hyperparameter

args = parser.parse_args()

# --- MLflow Setup ---
# Set an experiment name to group related runs
mlflow.set_experiment("Patient_Outcome_Prediction_Experiment")

# Start an MLflow run. All subsequent MLflow calls will be associated with this run.
# 'with mlflow.start_run():' ensures the run is properly closed even if errors occur.
with mlflow.start_run():
    # --- Log Parameters ---
    mlflow.log_param("input_data_path", args.input_data_path)
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("random_state", args.random_state)
    mlflow.log_param("test_size", 0.2) # Log fixed split size

    # --- 1. Load the processed data ---
    print(f"Loading data from: {args.input_data_path}")
    try:
        processed_data_df = pd.read_csv(args.input_data_path)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    # --- 2. Separate features (X) and target (y) ---
    TARGET_COLUMN = 'outcome' # <--- REPLACE WITH YOUR ACTUAL TARGET COLUMN NAME

    if TARGET_COLUMN not in processed_data_df.columns:
         print(f"Error: Target column '{TARGET_COLUMN}' not found in the data.")
         exit()

    y = processed_data_df[TARGET_COLUMN]
    X = processed_data_df.drop(columns=[TARGET_COLUMN])

    # Ensure consistent columns for X based on your original training features
    # (Important if you apply one-hot encoding or other dynamic preprocessing)
    # If your preprocessing creates new columns, you'll need to save/load
    # the list of expected columns or a preprocessing pipeline.
    # For now, we assume X has the right columns after reading the processed CSV.

    # --- 3. Split data into training and testing sets ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.random_state, stratify=y)


    # --- 4. Choose and train an ML model ---
    print("Training model...")
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state)
    model.fit(X_train, y_train)
    print("Model training complete.")


    # --- 5. Evaluate the model ---
    print("Evaluating model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("------------------------")

    # --- Log Metrics ---
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # --- 6. Save the trained model with MLflow ---
    # MLflow automatically saves the model under the run's artifact URI
    # and tracks its dependencies.
    # 'sklearn_model' is a flavor for scikit-learn models.
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="patient_outcome_model", # Name of the folder where the model will be saved
        registered_model_name="PatientOutcomePredictor" # Name to register the model in MLflow Model Registry
    )
    print(f"Model saved to MLflow artifacts path.")
    print(f"Run ID: {mlflow.active_run().info.run_id}")

print("\nMLflow training script finished.")
