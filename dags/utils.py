from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import os
from dotenv import load_dotenv
import logging

load_dotenv()

# Accéder aux variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')


import os
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

def download_files_from_s3(bucket_name: str, s3_prefix: str, local_dir: str, aws_conn_id: str = 'aws_s3'):
    """
    Télécharge un dossier entier depuis un bucket S3 et l'enregistre dans un répertoire local.

    Parameters:
        bucket_name (str): Le nom du bucket S3.
        s3_prefix (str): Le préfixe (dossier) dans S3 que l'on veut télécharger.
        local_dir (str): Le répertoire local où les fichiers doivent être enregistrés.
        aws_conn_id (str): L'identifiant de la connexion AWS dans Airflow.
    """
    hook = S3Hook(aws_conn_id=aws_conn_id)
    
    # Lister tous les objets avec le préfixe donné
    files = hook.list_keys(bucket_name=bucket_name, prefix=s3_prefix)
    
    if not files:
        print(f"Aucun fichier trouvé dans le dossier {s3_prefix} du bucket {bucket_name}")
        return
    
    # Créer le répertoire local s'il n'existe pas
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    for file_key in files:
        # Déterminer le chemin local du fichier
        local_file_path = os.path.join(local_dir, os.path.relpath(file_key, s3_prefix))
        
        # Créer les sous-dossiers nécessaires
        local_file_dir = os.path.dirname(local_file_path)
        if not os.path.exists(local_file_dir):
            os.makedirs(local_file_dir)
        
        # Télécharger le fichier
        hook.get_key(file_key, bucket_name).download_file(local_file_path)
        print(f"Fichier {file_key} téléchargé depuis le bucket {bucket_name} vers {local_file_path}")





