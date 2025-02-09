import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "CNN_Best_Model_Experiment"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

def load_latest_model():
    """
    Loads the latest trained model from MLflow for the specified experiment.

    Returns:
        The loaded ML model if found, otherwise None.
    """
    client = MlflowClient()

    # Get latest experiment ID
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        print(f"Experiment '{EXPERIMENT_NAME}' not found.")
        return None
    
    latest_experiment_id = experiment.experiment_id

    # Get the latest run from the latest experiment
    runs = client.search_runs(experiment_ids=[latest_experiment_id], order_by=["start_time DESC"], max_results=1)

    if runs:
        last_run_id = runs[0].info.run_id
        model_uri = f"runs:/{last_run_id}/best_model"  # Adjust model path as necessary
        print(f"Loading model from experiment ID {latest_experiment_id}, run {last_run_id}")

        try:
            model = mlflow.keras.load_model(model_uri)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    else:
        print("No runs found in the latest experiment.")
        return None
