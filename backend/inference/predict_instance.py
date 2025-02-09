import os
import sys 
import io
import base64

import cv2
import numpy as np

import tensorflow as tf

from lime import lime_image
from skimage.color import label2rgb

from inference.mlflow_config import load_latest_model
from inference.google_drive_config import download_model_from_drive
#project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.append(project_root)


# Load the latest model
new_model = load_latest_model()
if new_model is None:
    output_file = 'models/my_model.keras'
    new_model = download_model_from_drive(output_file)

# Map class index to label
y_labels = {
    0: 'healthy_brain',
    1: 'tumor_brain',
    2: 'alzheimer_brain'
}



# LIME explanation function with prediction
def explain_with_lime(image_path, model=new_model, class_labels=y_labels, top_labels=5, num_samples=10
                      ):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")

    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 512))  # Resize to model input shape
    img = img.astype('float32') / 255.0  # Normalize
    img_expanded = np.expand_dims(img, axis=0)  # Add batch dimension

    # Get prediction
    y_pred = model.predict(img_expanded)
    predicted_class_idx = np.argmax(y_pred, axis=-1)[0]
    predicted_class = class_labels[predicted_class_idx]

    # LIME Explainer
    explainer = lime_image.LimeImageExplainer()

    # Define a function for LIME to use model predictions
    def model_predict(images):
        images = np.array(images).astype("float32") / 255.0  # Ensure normalization
        predictions = model.predict(images)
        return predictions

    explanation = explainer.explain_instance(
        img, 
        model_predict, 
        top_labels=top_labels, 
        hide_color=0, 
        num_samples=num_samples
    )

    # Generate explanation image
    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=10,
        hide_rest=False
    )
    overlay = label2rgb(mask, image=img, bg_label=0, alpha=0.5)
    _, buffer = cv2.imencode('.png', (overlay * 255).astype(np.uint8))
    explanation_base64 = base64.b64encode(buffer).decode("utf-8")

    return predicted_class, explanation_base64
