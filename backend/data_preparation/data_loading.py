import os
import sys
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
