from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from inference.predict_instance import explain_with_lime  # Ensure correct import
import subprocess
import time
# Ensure the project root is accessible
#project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.append(project_root)


# Create FastAPI app
app = FastAPI()

# Ensure temp directory exists
TEMP_DIR = "tmp"
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
        prediction, explanation_img = explain_with_lime(temp_file_path)
        

        return {"filename": file.filename, "prediction": prediction, "explanation_url": explanation_img}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {str(e)}")
    
    finally:
        # Clean up: Delete the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    # Run the FastAPI app
    #run_gradio()
    #time.sleep(10)  # Wait for Gradio to start
    uvicorn.run(app, host="0.0.0.0", port=8003)

    