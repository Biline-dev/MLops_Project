from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from backend.inference.predict_instance import inference  # Ensure correct import

# Ensure the project root is accessible
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Load the model once at startup
MODEL_PATH = "mlops_project/backend/models/brain_mri_classifier.keras"
MODEL = tf.keras.models.load_model(MODEL_PATH)

# Create FastAPI app
app = FastAPI()

# Ensure temp directory exists
TEMP_DIR = "mlops_project/tmp"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """API endpoint to predict the class of an uploaded brain MRI image."""
    try:
        # Save file to a temporary directory
        temp_file_path = os.path.join(TEMP_DIR, file.filename)

        with open(temp_file_path, "wb") as f:
            contents = await file.read()  # Read file once
            f.write(contents)

        # Convert to OpenCV format
        img_array = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image format."}

        # Call inference function from predict.py
        prediction = inference(temp_file_path, MODEL)

        return {"filename": file.filename, "prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {str(e)}")
    
    finally:
        # Clean up: Delete the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8003)
