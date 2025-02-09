import gdown

import os
import tensorflow as tf

def download_model_from_drive(output_filename):
    """
    Downloads a model file from Google Drive based on the provided URL or file ID.

    Parameters:
    - file_url_or_id (str): The Google Drive file ID or shared folder URL.
    - output_filename (str): The local path where the model file will be saved.
    """

    # Remplacez ceci par l'ID du fichier de votre modèle
    file_id = '1bL2B5CoC9MKri8EikuXPGWwlvtUOtD0T'


    # Générer l'URL de téléchargement direct
    url = f'https://drive.google.com/uc?id={file_id}'

    # si le output_filename existe on load le model
    if os.path.exists(output_filename):
        new_model = tf.keras.models.load_model(output_filename)
        print(f"Model loaded from {output_filename}.")
        return new_model
    else:
        print(f"Downloading model from {url}...")
        try:
            # Download the file using gdown
            gdown.download(url, output_filename, quiet=False)
            new_model = tf.keras.models.load_model(output_filename)
            print(f"File downloaded successfully and saved as {output_filename}.")
            return new_model
        except Exception as e:
            print(f"Error downloading file: {e}")

    return None
