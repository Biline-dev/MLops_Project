import os
import numpy as np
import random
import cv2

def load_and_preprocess_images(directory, label, num_images=None, img_height=128, img_width=128):
    """
    Loads images from the directory, resizes them, and assigns the corresponding label.

    Args:
    - directory (str): Path to the image directory.
    - label (int): The label to assign to the images in this directory.
    - num_images (int, optional): The number of images to load. Loads all images if not provided.
    - img_height (int): Height of the resized images.
    - img_width (int): Width of the resized images.

    Returns:
    - images (list): List of processed image arrays.
    - labels (list): List of labels corresponding to the images.
    - filenames (list): List of filenames of the images.
    """
    images = []
    labels = []
    filenames = []

    # List all files in the directory
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('jpg', 'png', 'jpeg'))]

    # Limit the number of images if `num_images` is provided
    if num_images is not None:
        image_files = random.sample(image_files, min(len(image_files), num_images))  # Randomly sample the required number of images

    for filename in image_files:
        img_path = os.path.join(directory, filename)

        # Load and resize the image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_width, img_height))  # Resize to the target shape
        images.append(img)
        labels.append(label)
        filenames.append(filename)

    return images, labels, filenames


# Function to load and preprocess images from local directories
def load_local_images_and_preprocess(final_dataset_dir, num_images_per_class, IMG_HEIGHT, IMG_WIDTH):
    """
    Load and preprocess images from local directories, and return the combined data.
    
    Parameters:
        - final_dataset_dir: The base directory where the dataset is stored.
        - num_images_per_class: The number of images to load per class.
        - IMG_HEIGHT, IMG_WIDTH: The target dimensions for resizing images.
        
    Returns:
        - X_train, X_test, y_train, y_test: Preprocessed and split training and test data.
    """
    
    # Define directories for each class
    healthy_brain_final_dir = os.path.abspath(os.path.join(final_dataset_dir, "healthy_brain"))
    tumor_brain_final_dir = os.path.abspath(os.path.join(final_dataset_dir, "tumor_brain"))
    alzheimer_brain_final_dir = os.path.abspath(os.path.join(final_dataset_dir, "alzheimer_brain"))

    # Load and preprocess images for each class
    healthy_brain_images, healthy_brain_labels, healthy_brain_filenames = load_and_preprocess_images(
        healthy_brain_final_dir, label=0, num_images=num_images_per_class, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
    
    tumor_brain_images, tumor_brain_labels, tumor_brain_filenames = load_and_preprocess_images(
        tumor_brain_final_dir, label=1, num_images=num_images_per_class, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
    
    alzheimer_brain_images, alzheimer_brain_labels, alzheimer_brain_filenames = load_and_preprocess_images(
        alzheimer_brain_final_dir, label=2, num_images=num_images_per_class, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)

    # Combine the data from all categories
    X = np.array(healthy_brain_images + tumor_brain_images + alzheimer_brain_images)
    y = np.array(healthy_brain_labels + tumor_brain_labels + alzheimer_brain_labels)
  
    return X, y

