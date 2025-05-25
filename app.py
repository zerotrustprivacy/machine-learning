# app.py
# This Python script will load your trained model and expose a prediction endpoint using a lightweight web framework.
import os
import joblib # Or import pickle as joblib
from flask import Flask, request, jsonify
import pandas as pd
from google.cloud import storage # Required to download model from GCS

app = Flask(__name__)

# --- Configuration ---
# IMPORTANT: Replace with the exact GCS path to your trained model file
# Example: 'gs://your-model-bucket/models/model.pkl'
# This path should point to the output of your training job
GCS_MODEL_PATH = os.environ.get('GCS_MODEL_PATH', 'gs://your-model-bucket/models/model.pkl') # <--- UPDATE THIS!

MODEL = None # Global variable to hold the loaded model

def download_model_from_gcs(gcs_path):
    """Downloads the model from GCS to a local temporary file."""
    bucket_name = gcs_path.split('//')[1].split('/')[0]
    blob_path = '/'.join(gcs_path.split('//')[1].split('/')[1:])
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    local_file_path = f"/tmp/{blob_path.split('/')[-1]}" # Save to /tmp
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
    blob.download_to_filename(local_file_path)
    return local_file_path

@app.before_first_request
def load_model():
    """Load the model only once when the app starts."""
    global MODEL
    print(f"Loading model from GCS: {GCS_MODEL_PATH}")
    try:
        local_model_path = download_model_from_gcs(GCS_MODEL_PATH)
        MODEL = joblib.load(local_model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        # In a real app, you might want to exit or log more severely
        MODEL = None

@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    try:
        data = request.get_json(force=True)
        # IMPORTANT: Adapt this part based on your model's expected input
        # The input 'data' should match the features your model was trained on.
        # Example assumes input is a list of lists or a dictionary that can form a DataFrame
        
        # If your model expects a dictionary, you might convert directly:
        # prediction_input = pd.DataFrame([data]) 
        # If your model expects specific order of columns, you need to ensure it here:
        
        # Example for single prediction with specific columns:
        # Assume data is {'feature1': value1, 'feature2': value2, ...}
        # features = [data['feature1'], data['feature2'], ...] 
        # prediction_input = pd.DataFrame([features], columns=['feature1', 'feature2', ...])
        
        # For simplicity, let's assume the incoming JSON matches the column order
        # of the features used for training. You'll need to adapt this precisely.
        prediction_input = pd.DataFrame(data) 
        
        # Make prediction
        prediction = MODEL.predict(prediction_input)
        
        # For classification, you might want probabilities too
        # probabilities = MODEL.predict_proba(prediction_input)

        return jsonify({'prediction': prediction.tolist()}) # Convert numpy array to list for JSON
    except Exception as e:
        return jsonify({'error': str(e), 'message': 'Prediction failed.'}), 400

@app.route('/health', methods=['GET'])
def health_check():
    # Simple health check to ensure the service is running and model is loaded
    if MODEL is not None:
        return jsonify({'status': 'healthy', 'model_loaded': True}), 200
    else:
        return jsonify({'status': 'unhealthy', 'model_loaded': False}), 500

if __name__ == '__main__':
    # When running locally, set the GCS_MODEL_PATH environment variable
    # For testing, you might need to run `gcloud auth application-default login`
    # if you're not in a cloud environment that provides credentials automatically.
    # Also, ensure your local user has permissions to access the GCS bucket.
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
