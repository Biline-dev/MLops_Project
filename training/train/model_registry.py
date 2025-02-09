import os
import sys 
import ast

import click
import mlflow
import mlflow.keras
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

import numpy as np

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping

# Ensure the project root is accessible
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from mlops_project.backend.data_preparation.data_loading_locally import load_local_images_and_preprocess
from mlops_project.backend.data_preparation.data_loading_s3 import load_images_from_source
from data_preparation.data_processing import preprocess_and_split_data



HPO_EXPERIMENT_NAME = "CNN_Model_Experiment"
EXPERIMENT_NAME = "CNN_Best_Model_Experiment"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)



IMG_HEIGHT = 512
IMG_WIDTH = 512
NUM_IMAGES_PER_CLASSES = 800

# choose between aws s3 or local storage
use_s3 = False
if use_s3:
    # Function to generate file keys for each class
    def generate_s3_file_keys(class_name, num_images):
        return [f'{class_name}/{class_name}_{i}.jpg' for i in range(1, num_images + 1)]

    # Generate file keys for each class
    healthy_brain_file_keys = generate_s3_file_keys('healthy_brain', NUM_IMAGES_PER_CLASSES)
    tumor_brain_file_keys = generate_s3_file_keys('tumor_brain', NUM_IMAGES_PER_CLASSES)
    alzheimer_brain_file_keys = generate_s3_file_keys('alzheimer_brain', NUM_IMAGES_PER_CLASSES)
    bucket_name = "mlopspipe"
    healthy_brain_file_keys = ["healthy_brain_file_key1", "healthy_brain_file_key2", "healthy_brain_file_key3"]
    tumor_brain_file_keys = ["tumor_brain_file_key1", "tumor_brain_file_key2", "tumor_brain_file_key3"]
    alzheimer_brain_file_keys = ["alzheimer_brain_file_key1", "alzheimer_brain_file_key2", "alzheimer_brain_file_key3"]
    X, y = load_images_from_source(bucket_name, healthy_brain_file_keys, tumor_brain_file_keys, alzheimer_brain_file_keys,
                                   num_images_per_class=NUM_IMAGES_PER_CLASSES, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, use_s3=use_s3)
    # Preprocess and Split the Data
    X_train, X_test, y_train, y_test = preprocess_and_split_data(X, y)
else :
    final_dataset_dir = "mlops_project/database"
    X, y = load_local_images_and_preprocess(final_dataset_dir, num_images_per_class=NUM_IMAGES_PER_CLASSES, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH)
    # Preprocess and Split the Data
    X_train, X_test, y_train, y_test = preprocess_and_split_data(X, y)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)



# Function to build the model with customizable layers and nodes
def build_model(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, num_classes=3, 
                conv_layers=[32, 64, 128], dense_nodes=128, dropout_rate=0.5, learning_rate=0.001):
    model = models.Sequential()
    
    # Add convolutional layers
    for filters in conv_layers:
        model.add(layers.Conv2D(filters, (3, 3), activation='relu', input_shape=(img_height, img_width, 3) if len(model.layers) == 0 else None))
        model.add(layers.MaxPooling2D(2, 2))

    # Flatten the output
    model.add(layers.Flatten())
    
    # Add dense layers
    model.add(layers.Dense(dense_nodes, activation='relu'))
    model.add(layers.Dropout(dropout_rate))  # Regularization
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Function to train the model and track the best model based on accuracy
def train_and_track_best_model(X_train, y_train, X_test, y_test, params):
    
    with mlflow.start_run():

        mlflow.log_params(params)
        
        conv_layers = ast.literal_eval(params['conv_layers'])

        model = build_model(int(params["img_height"]), int(params["img_width"]), int(params["num_classes"]),
                            conv_layers, int(params["dense_nodes"]), float(params["dropout_rate"]), float(params["learning_rate"]))
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        history = model.fit(X_train, y_train, batch_size=int(params["batch_size"]), epochs=int(params["epochs"]),
                            validation_split=0.2, callbacks=[early_stopping])

        num_epochs_completed = len(history.history['loss']) 

        # Log all training metrics
        for epoch in range(num_epochs_completed):
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)

        # Move model evaluation OUTSIDE the loop
        test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=int(params["batch_size"]))
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)

        # Save best model only if test accuracy improves
        model_path = "models/brain_mri_classifier.keras"
        model.save(model_path)  
        mlflow.keras.log_model(model, artifact_path="best_model")

        # Log final metrics
        mlflow.log_metrics({
            "final_train_loss": history.history['loss'][-1],
            "final_train_accuracy": history.history['accuracy'][-1],
            "final_val_loss": history.history['val_loss'][-1],
            "final_val_accuracy": history.history['val_accuracy'][-1]
        })

    return model  # Return the trained model
@click.command()
@click.option(
    "--top_n",
    default=2,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(top_n: int):

    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.test_accuracy DESC"]
    )
    for run in runs:
        train_and_track_best_model(X_train, y_train, X_test, y_test, params=run.data.params)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(experiment.experiment_id, order_by=["metrics.test_accuracy DESC"])[0]
    print(f"Best Model: {best_run.info.run_id}")
    # Register the best model
    mlflow.register_model(
        f"runs:/{best_run.info.run_id}/model",
        f"{EXPERIMENT_NAME}_best_model"
    )


if __name__ == '__main__':
    run_register_model()
