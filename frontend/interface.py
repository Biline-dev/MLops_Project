import gradio as gr
import requests

API_URL = "http://localhost:8003/predict/"  # FastAPI endpoint

def predict_image(image):
    """Sends the image to the FastAPI server and gets the prediction."""
    if image is None:
        return "No image uploaded", None

    # Save the image temporarily
    image.save("temp_upload.png")

    # Send image to FastAPI
    with open("temp_upload.png", "rb") as img_file:
        files = {"file": img_file}
        response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        prediction = response.json().get("prediction", "Unknown")
    else:
        prediction = f"Error: {response.text}"

    return prediction

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Brain MRI Classifier")
    
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Brain MRI")
        output_text = gr.Label(label="Prediction")

    with gr.Row():
        
        predict_button = gr.Button("Predict")
        #output_image = gr.Image(label="Uploaded Image")

    predict_button.click(fn=predict_image, inputs=image_input, outputs=[output_text])

# Run Gradio
if __name__ == "__main__":
    demo.launch()