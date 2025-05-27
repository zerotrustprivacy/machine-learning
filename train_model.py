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

# Define command-line arguments
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

# MLflow run
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
        print(f"Data shape: {processed_data_df.shape}")
        print("Data columns (headers):", processed_data_df.columns.tolist())
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    # --- 2. Separate features (X) and target (y) ---
    TARGET_COLUMN = 'Test Results'

    if TARGET_COLUMN not in processed_data_df.columns:
        print(f"Error: Target column '{TARGET_COLUMN}' not found in the data. "
              f"Please check the spelling and capitalization.")
        exit()
    y = processed_data_df[TARGET_COLUMN]

    # --- Define Column Types and Actions ---
    # These lists are based on your CSV headers and typical data handling.

    # Columns to explicitly drop from features
    DROP_COLS = [
        'Name',
        'Doctor',
        'Date of Admission',
        'Discharge Date',
    ]

    # Columns that should be numerical features (will be converted to float)
    NUMERIC_FEAT_COLS = ['Age', 'Billing Amount', 'Room Number']

    # Columns that are categorical features (will be one-hot encoded)
    # Ensure these are not in DROP_COLS or NUMERIC_FEAT_COLS
    CATEGORICAL_FEAT_COLS = [
        'Gender',
        'Blood Type',
        'Medical Condition',
        'Hospital',
        'Insurance Provider',
        'Admission Type',
        'Medication'
    ]


    # --- Create X (Features DataFrame) ---
    # Start X by dropping the target column and any explicitly defined DROP_COLS
    initial_drop_cols = [TARGET_COLUMN] + DROP_COLS
    existing_initial_drop_cols = [col for col in initial_drop_cols if col in processed_data_df.columns]
    X = processed_data_df.drop(columns=existing_initial_drop_cols)


    # --- Convert Numeric Columns to Appropriate Types ---
    print("\n--- Converting Numeric Columns ---")
    for col in NUMERIC_FEAT_COLS:
        if col in X.columns:
            # Convert to numeric, coercing errors to NaN (Not a Number) if values can't be converted
            X[col] = pd.to_numeric(X[col], errors='coerce')
            # Fill any NaNs created by conversion or original missing values
            # Common strategies: mean, median, or 0. Using mean here.
            X[col] = X[col].fillna(X[col].mean())
            print(f"Converted '{col}' to numeric and filled NaNs.")
        else:
            print(f"Warning: Numeric feature '{col}' not found in features X. Skipping conversion.")


    # --- Handle Categorical Columns (One-Hot Encoding) ---
    # Identify remaining 'object' columns *after* dropping and numeric conversion.
    categorical_cols_to_encode = X.select_dtypes(include='object').columns.tolist()
    
    # Filter to only include those we specifically defined as categorical features
    categorical_cols_to_encode = [col for col in categorical_cols_to_encode if col in CATEGORICAL_FEAT_COLS]
    
    print(f"\n--- One-Hot Encoding Categorical Columns ---")
    print(f"Columns to be one-hot encoded: {categorical_cols_to_encode}")

    if categorical_cols_to_encode:
        # Apply one-hot encoding
        X = pd.get_dummies(X, columns=categorical_cols_to_encode, drop_first=True) # drop_first=True
        print("Categorical columns one-hot encoded.")
    else:
        print("No categorical (object) columns found in X to one-hot encode.")
    
    # --- Final Check for Non-Numeric Columns in X ---
    print("\n--- Final Check for Non-Numerical Columns in X ---")
    final_object_cols_in_X = X.select_dtypes(include='object').columns.tolist()
    if final_object_cols_in_X:
        print(f"ERROR: Found 'object' (string) columns remaining in X: {final_object_cols_in_X}. Model training will fail.")
        print("Please review your DROP_COLS, NUMERIC_FEAT_COLS, and CATEGORICAL_FEAT_COLS lists.")
        # Print sample values to help identify problem
        for col in final_object_cols_in_X:
            print(f"- Column '{col}': {X[col].unique()[:5]} (showing first 5 unique values)")
        exit() # Exit to prevent model.fit error
    else:
        print("All columns in X are now numerical.")

    print("\n--- Final X DataFrame Info before training ---")
    X.info()
    print("------------------------------------------\n")


    # --- 3. Split data into training and testing sets ---
    # stratify=y if your target column is suitable (more than 1 sample per class)
    # If you get the "least populated class" error again, remove stratify=y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.random_state, stratify=y)


    # --- 4. Choose and train an ML model ---
    print("Training model...")
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state)
    model.fit(X_train, y_train)
    print("Model training complete.")


    # --- 5. Evaluate the model ---
    print("Evaluating model...")
    y_pred = model.predict(X_test)

    # Convert y_test and y_pred to a common numeric type if target is categorical and not numeric
    # For example, if 'Test Results' contains 'Normal', 'Abnormal' strings,
    # you might need to encode y as well for metrics if they expect numeric input.
    # For example, if y is string categories, these metrics will work:
    # from sklearn.preprocessing import LabelEncoder
    # le = LabelEncoder()
    # y_test_encoded = le.fit_transform(y_test)
    # y_pred_encoded = le.transform(y_pred)
    # accuracy = accuracy_score(y_test_encoded, y_pred_encoded) etc.
    # Let's assume for now y is fine for these metrics, or they handle strings.

    accuracy = accuracy_score(y_test, y_pred)
    # Precision, Recall, F1-score require specifying 'average' for multi-class classification,
    # or you'll get a warning/error if your target has >2 unique values.
    # Since 'Test Results' could be multi-class ('Normal', 'Abnormal', etc.),
    # let's add average='weighted' or 'macro'
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)


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
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="patient_outcome_model", # Name of the folder where the model will be saved
        registered_model_name="PatientOutcomePredictor" # Name to register the model in MLflow Model Registry
    )
    print(f"Model saved to MLflow artifacts path.")
    print(f"Run ID: {mlflow.active_run().info.run_id}")

print("\nMLflow training script finished.")
