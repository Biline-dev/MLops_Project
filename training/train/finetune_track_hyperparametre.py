import mlflow
import mlflow.keras
import numpy as np
import random
import os 
import sys

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping


from train.data_preparation.data_loading_locally import load_local_images_and_preprocess
from train.data_preparation.data_loading_s3 import load_images_from_source
from train.data_preparation.data_processing import preprocess_and_split_data

# Ensure the project root is accessible
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)



HPO_EXPERIMENT_NAME = "CNN_Model_Experiment"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(HPO_EXPERIMENT_NAME)



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
def build_model(img_height=512, img_width=512, num_classes=3, 
                conv_layers=[32, 64, 128], dense_nodes=128, dropout_rate=0.5, learning_rate=0.001):
    model = models.Sequential()
    
    for filters in conv_layers:
        model.add(layers.Conv2D(filters, (3, 3), activation='relu', input_shape=(img_height, img_width, 3) if len(model.layers) == 0 else None))
        model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Flatten())
    model.add(layers.Dense(dense_nodes, activation='relu'))
    model.add(layers.Dropout(dropout_rate))  
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Function to train and track the best model based on accuracy
def train_and_track_best_model(X_train, y_train, X_test, y_test, img_height=512, img_width=512, batch_size=8, epochs=50, patience=5, 
                conv_layers=[32, 64, 128], dense_nodes=128, dropout_rate=0.5, 
                learning_rate=0.001, num_classes=3):
    
    with mlflow.start_run():
        mlflow.log_params({
            "img_height": img_height, "img_width": img_width, "batch_size": batch_size,
            "epochs": epochs, "conv_layers": conv_layers, "dense_nodes": dense_nodes,
            "dropout_rate": dropout_rate, "learning_rate": learning_rate, "num_classes": num_classes,
            "patience": patience
        })

        model = build_model(img_height, img_width, num_classes, conv_layers, dense_nodes, dropout_rate, learning_rate)
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, 
                            validation_split=0.2, callbacks=[early_stopping])

        num_epochs_completed = len(history.history['loss'])

        for epoch in range(num_epochs_completed):
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)

        test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)

        model_path = f"models/brain_mri_classifier_run_{random.randint(1, 1000)}.keras"
        model.save(model_path)  
        mlflow.keras.log_model(model, artifact_path="best_model")

        mlflow.log_metrics({
            "final_train_loss": history.history['loss'][-1],
            "final_train_accuracy": history.history['accuracy'][-1],
            "final_val_loss": history.history['val_loss'][-1],
            "final_val_accuracy": history.history['val_accuracy'][-1]
        })

    return model


# --- MULTIPLE EXECUTIONS (8 runs) ---
if __name__ == "__main__":

    for i in range(8):
        print(f"Execution {i+1}/8")

        # Generate random hyperparameters
        learning_rate = round(10**random.uniform(-4, -2), 6)  # Random between 0.0001 and 0.01
        dropout_rate = round(random.uniform(0.3, 0.6), 2)     # Random between 0.3 and 0.6
        patience = random.randint(3, 8)                       # Random patience between 3 and 8 epochs

        best_model = train_and_track_best_model(
            X_train, y_train, X_test, y_test,
            img_height=512, img_width=512, batch_size=8, epochs=20, patience=patience,
            conv_layers=[32, 64, 128], dense_nodes=128, dropout_rate=dropout_rate,
            learning_rate=learning_rate, num_classes=3
        )

        print(f"Run {i+1}/8 finished!\n")
