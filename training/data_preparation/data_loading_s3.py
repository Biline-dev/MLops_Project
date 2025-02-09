
from io import BytesIO
from PIL import Image
import numpy as np

import boto3


# AWS S3 interaction function to fetch the image directly
def load_image_from_s3(bucket_name, file_key, s3_client):
    try:
        # Fetch the file from S3 as a byte stream
        obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        img_data = obj['Body'].read()
        
        # Open the image directly from the byte stream
        img = Image.open(BytesIO(img_data))
        return img
    except Exception as e:
        print(f"Error loading image '{file_key}' from S3: {e}")
        return None

# Load and preprocess images function (modified to load directly from S3)
def load_and_preprocess_images_from_s3(bucket_name, file_keys, label, num_images, img_height, img_width, s3_client):
    images = []
    labels = []
    filenames = []

    for key in file_keys:
        if key.lower().endswith('.jpg'):  # Make sure it's a JPG file
            img = load_image_from_s3(bucket_name, key, s3_client)
            if img:
                img = img.resize((img_width, img_height))  # Resize image
                images.append(np.array(img))
                labels.append(label)
                filenames.append(key)
                if len(images) >= num_images:
                    break  # Limit to the desired number of images
    return images, labels, filenames


# Function to load and preprocess images either from S3 or local storage
def load_images_from_source(bucket_name, healthy_brain_file_keys, tumor_brain_file_keys, alzheimer_brain_file_keys,
                            num_images_per_class, IMG_HEIGHT, IMG_WIDTH, use_s3=True):
    """
    Load and preprocess images either from S3 or local storage based on the 'use_s3' flag.
    
    Parameters:
        - bucket_name: The S3 bucket name.
        - healthy_brain_file_keys, tumor_brain_file_keys, alzheimer_brain_file_keys: List of file keys for each class.
        - num_images_per_class: Number of images to load per class.
        - IMG_HEIGHT, IMG_WIDTH: Image dimensions to resize.
        - use_s3: Flag to indicate whether to load images from S3 or local storage.
        
    Returns:
        - X_train, X_test, y_train, y_test: Preprocessed and split training and test data.
    """
    
    # AWS credentials and S3 client setup (if use_s3 is True)
    s3_client = None
    if use_s3:
        aws_access_key_id = "aws_access_key_id"
        aws_secret_access_key = "aws_secret_access_key"
        region_name = "region"

        s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id,
                                 aws_secret_access_key=aws_secret_access_key, region_name=region_name)
    
    # Load and preprocess images (either from S3 or local)
    healthy_brain_images, healthy_brain_labels, healthy_brain_filenames = load_and_preprocess_images_from_s3(
        bucket_name, healthy_brain_file_keys, label=0, num_images=num_images_per_class,
        img_height=IMG_HEIGHT, img_width=IMG_WIDTH, s3_client=s3_client, use_s3=use_s3)

    tumor_brain_images, tumor_brain_labels, tumor_brain_filenames = load_and_preprocess_images_from_s3(
        bucket_name, tumor_brain_file_keys, label=1, num_images=num_images_per_class,
        img_height=IMG_HEIGHT, img_width=IMG_WIDTH, s3_client=s3_client, use_s3=use_s3)

    alzheimer_brain_images, alzheimer_brain_labels, alzheimer_brain_filenames = load_and_preprocess_images_from_s3(
        bucket_name, alzheimer_brain_file_keys, label=2, num_images=num_images_per_class,
        img_height=IMG_HEIGHT, img_width=IMG_WIDTH, s3_client=s3_client, use_s3=use_s3)

    # Combine the data from all categories
    X = np.array(healthy_brain_images + tumor_brain_images + alzheimer_brain_images)
    y = np.array(healthy_brain_labels + tumor_brain_labels + alzheimer_brain_labels)

    
    return X, y
