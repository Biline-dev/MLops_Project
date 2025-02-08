import os
import cv2
import numpy as np
import tensorflow as tf



#new_model = tf.keras.models.load_model('mlops_project/backend/models/brain_mri_classifier.keras')

def inference(img_path, model, img_height=512, img_width=512):
    
    if not os.path.exists(img_path):
        
        raise FileNotFoundError(f"Image file '{img_path}' not found.")
    
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_width, img_height))  # Resize to the target shape
    
    # Normalize pixel values to [0, 1]
    img = img.astype('float32') / 255.0  
    
    # Expand dimensions to match model input shape
    img = np.expand_dims(img, axis=0)  
    
    # Make prediction
    y_pred = model.predict(img)
    y_pred_class = tf.argmax(y_pred, axis=1).numpy()[0]  # Convert tensor to numpy and get class index
    
    # Map class index to label
    y_labels = {
        0: 'healthy_brain',
        1: 'tumor_brain',
        2: 'alzheimer_brain'
    }

    return y_labels[y_pred_class]  # Return the correct label
    
# Example usage:
#predicted_label = inference("mlops_project/database/healthy_brain/healthy_brain_1.jpg", new_model)
#print(f"Predicted Label: {predicted_label}")

