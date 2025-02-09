import gradio as gr
import requests
import base64
import cv2
import numpy as np
import io

API_URL = "http://127.0.0.1:8003/predict/"  # FastAPI endpoint

def predict_image(image):
    """Sends the image to the FastAPI server and gets the prediction & explanation."""
    if image is None:
        return "No image uploaded", None, None

    # Convert image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    # Send image to FastAPI
    files = {"file": ("image.png", image_bytes, "image/png")}
    response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        response_data = response.json()
        prediction = response_data.get("prediction", "Unknown")

        # Get explanation image
        explanation_url = response_data.get("explanation_url", None)
        if explanation_url:
            explanation_bytes = base64.b64decode(explanation_url)
            np_arr = np.frombuffer(explanation_bytes, np.uint8)
            explanation_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            explanation_img = None
        
    else:
        prediction = f"Error: {response.text}"
        explanation_img = None

    return prediction,  explanation_img

# Gradio Interface
with gr.Blocks() as demo:
    # Header Section
    gr.Markdown("# 🧠 Brain MRI Classifier")
    gr.Markdown("""
        ### This tool helps in analyzing brain MRI images for potential abnormalities. 
        Upload a brain MRI image, and the model will predict the condition with an explanation of the result.
    """)
    
    with gr.Row():
        with gr.Column():
            # Image Input Section
            image_input = gr.Image(type="pil", label="Upload Brain MRI", elem_id="image-input")
            predict_button = gr.Button("Predict", elem_id="predict-button")
            gr.Markdown("### Step 1: Upload an MRI image of the brain.")

        with gr.Column():
            # Prediction and Explanation Section
        
            explanation_image = gr.Image(label="LIME Explanation", elem_id="explanation-image")
            output_text = gr.Label(label="Prediction", elem_id="prediction-output")
            gr.Markdown("### Step 2: Receive the prediction and explanation.")

    
    # Button click event
    predict_button.click(fn=predict_image, inputs=image_input, outputs=[output_text, explanation_image])

# Run Gradio
if __name__ == "__main__":
    demo.launch()
