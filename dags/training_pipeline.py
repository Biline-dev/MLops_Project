import pandas as pd
from datetime import datetime, timedelta
from airflow.decorators import task_group, task, dag
from airflow.utils.dates import days_ago
import logging
import numpy as np
import mlflow
import mlflow.keras
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from utils import download_files_from_s3
from backend.data_preparation.data_loading import load_and_preprocess_images
from backend.data_preparation.data_processing import preprocess_and_split_data
from backend.train.model_registry import *

# Paramètres par défaut pour le DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=200),
}

@dag(
    dag_id='training_MRI_images_classification',
    default_args=default_args,
    description='A data pipeline for MRI image classification',
    schedule_interval='0 8 * * 1',
    start_date=days_ago(1),
    catchup=False,
    dagrun_timeout=timedelta(hours=2)
)
def MRI_images_classification_dag():
    # Importer les datasets
    @task_group(group_id='import_data')
    def import_data():
        
        @task(task_id='download_tumor_brain')
        def download_tumor_brain():
            brain_tumor_path = "/usr/local/airflow/data/tumor_brain"
            download_files_from_s3(
                bucket_name='mlopspipe',
                s3_key='tumor_brain',
                local_path=brain_tumor_path,
                aws_conn_id='aws_s3'
            )
            logging.info(f"Downloaded brain tumor data to {brain_tumor_path}")
            return brain_tumor_path

        @task(task_id='download_healthy_brain')
        def download_healthy_brain():
            brain_healthy_path = "/usr/local/airflow/data/healthy_brain"
            download_files_from_s3(
                bucket_name='mlopspipe',
                s3_key='healthy_brain',
                local_path=brain_healthy_path,
                aws_conn_id='aws_s3'
            )
            logging.info(f"Downloaded brain healthy data to {brain_healthy_path}")
            return brain_healthy_path
        
        @task(task_id='download_alzheimer_brain')
        def download_alzheimer_brain():
            brain_alzheimer_path = "/usr/local/airflow/data/alzheimer_brain"
            download_files_from_s3(
                bucket_name='mlopspipe',
                s3_key='alzheimer_brain',
                local_path=brain_alzheimer_path,
                aws_conn_id='aws_s3'
            )
            logging.info(f"Downloaded brain alzheimer data to {brain_alzheimer_path}")
            return brain_alzheimer_path
    
        tumor_brain_data = download_tumor_brain()
        healthy_brain_data = download_healthy_brain()
        alzheimer_brain_data = download_alzheimer_brain()
        
        return tumor_brain_data, healthy_brain_data, alzheimer_brain_data
    
    @task_group(group_id='preprocess_images')
    def preprocess_images(tumor_brain_data, healthy_brain_data, alzheimer_brain_data, num_images_per_class=800, IMG_HEIGHT=512, IMG_WIDTH=512):
        @task(task_id='preprocess_images_tumor_brain')
        def preprocess_images_tumor_brain():
            return load_and_preprocess_images(tumor_brain_data, 0, num_images=num_images_per_class, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)

        @task(task_id='preprocess_images_healthy_brain')
        def preprocess_images_healthy_brain():
            return load_and_preprocess_images(healthy_brain_data, 0, num_images=num_images_per_class, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)

        @task(task_id='preprocess_images_alzheimer_brain')
        def preprocess_images_alzheimer_brain():
            return load_and_preprocess_images(alzheimer_brain_data, 0, num_images=num_images_per_class, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)

        return preprocess_images_tumor_brain(), preprocess_images_healthy_brain(), preprocess_images_alzheimer_brain()

    @task(task_id='merge_data')
    def merge_data(tumor_brain, healthy_brain, alzheimer_brain):
        X = np.array(tumor_brain[0] + healthy_brain[0] + alzheimer_brain[0])
        y = np.array(tumor_brain[1] + healthy_brain[1] + alzheimer_brain[1])
        merged_data_path = '/usr/local/airflow/data/merged_data.npy'
        np.save(merged_data_path, {"X": X, "y": y})
        return X, y
    
    @task(task_id='split_data')
    def split_data(X, y):
        X_train, X_test, y_train, y_test = preprocess_and_split_data(X, y)
        process_data_path = '/usr/local/airflow/data/split_data.npy'
        np.save(process_data_path, {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test})
        return X_train, X_test, y_train, y_test
        
    imported_data = import_data()
    preprocess_data = preprocess_images(imported_data[0], imported_data[1], imported_data[2])
    merged_data_result = merge_data(preprocess_data[0], preprocess_data[1], preprocess_data[2])
    split_data_result = split_data(merged_data_result[0], merged_data_result[1])

    @task(task_id='register_mlflow_model')
    def register_mlflow_model(top_n: int = 5):
        client = MlflowClient()
        
        experiment = client.get_experiment_by_name("HPO_EXPERIMENT_NAME")
        runs = client.search_runs(
            experiment_ids=experiment.experiment_id,
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=top_n,
            order_by=["metrics.test_accuracy ASC"]
        )

        for run in runs:
            train_and_track_best_model(
                split_data_result[0], split_data_result[1], 
                split_data_result[2], split_data_result[3], 
                params=run.data.params
            )
        
        experiment = client.get_experiment_by_name("EXPERIMENT_NAME")
        best_run = client.search_runs(
            experiment.experiment_id,
            order_by=["metrics.test_accuracy ASC"]
        )[0]

        logging.info(f"Best Model: {best_run.info.run_id}")
        
        mlflow.register_model(
            f"runs:/{best_run.info.run_id}/model",
            "EXPERIMENT_NAME_best_model"
        )
    
    register_mlflow_model()  

dag = MRI_images_classification_dag()
