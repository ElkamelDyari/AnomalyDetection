"""
MLflow tracking for Drain log parsing model in the AnomalyDetection project
"""
import os
import json
import mlflow
from mlflow_utils import init_tracking, log_model_params_from_json, log_artifact_if_exists, log_directory_artifacts, log_numpy_array_summary

def track_drain_existing_model(base_dir="data/Drain"):
    """
    Track an existing Drain log parsing model that has already been trained
    Logs model parameters, artifacts, and results
    """
    init_tracking()
    
    # Define common paths
    model_path = os.path.join(base_dir, "drain_model.pkl")  # Assuming pickle format for Drain
    config_path = os.path.join(base_dir, "drain_config.json")
    results_path = os.path.join(base_dir, "drain_results.json")
    logs_dir = os.path.join(base_dir, "parsed_logs")
    
    # Verify files exist
    if not os.path.exists(model_path) and not os.path.exists(results_path):
        print(f"Drain model or results not found in {base_dir}")
        return False
        
    print(f"Tracking existing Drain log parsing model")
    
    with mlflow.start_run(run_name="Drain_Log_Parsing"):
        # Log model configuration parameters
        if os.path.exists(config_path):
            log_model_params_from_json(config_path)
        else:
            # Default Drain parameters if config doesn't exist
            mlflow.log_param("model", "Drain")
            mlflow.log_param("sim_th", 0.4)  # Common default
            mlflow.log_param("depth", 4)     # Common default
        
        # Log the model file if it exists
        log_artifact_if_exists(model_path)
        
        # Log results file if it exists
        log_artifact_if_exists(results_path)
        
        # Log results metrics if available
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    results = json.load(f)
                
                # Log metrics from results
                if isinstance(results, dict):
                    for key, value in results.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(key, value)
            except Exception as e:
                print(f"Error logging Drain results: {e}")
        
        # Log parsed logs if they exist
        if os.path.exists(logs_dir):
            log_count = log_directory_artifacts(logs_dir)
            mlflow.log_metric("log_template_count", log_count)
        
        print("Drain model tracking completed")
        return True

def track_drain_workflow(base_dir="data/Drain", log_input_path="data/Drain/input_logs"):
    """
    Track the full Drain log parsing workflow with MLflow
    This would be integrated into the log parsing workflow
    """
    init_tracking()
    
    # Define paths
    config_path = os.path.join(base_dir, "drain_config.json")
    model_path = os.path.join(base_dir, "drain_model.pkl")
    results_path = os.path.join(base_dir, "drain_results.json")
    logs_dir = os.path.join(base_dir, "parsed_logs")
    
    # Ensure directories exist
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Track model configuration and training
    with mlflow.start_run(run_name="Drain_Parsing"):
        # Log parameters
        mlflow.log_param("log_input_path", log_input_path)
        if os.path.exists(config_path):
            log_model_params_from_json(config_path)
        else:
            # Default parameters
            mlflow.log_param("model", "Drain")
            mlflow.log_param("sim_th", 0.4)
            mlflow.log_param("depth", 4)
        
        # The actual Drain parsing would happen here:
        # drain_parser = parse_logs_with_drain(
        #     log_input_path=log_input_path,
        #     output_model_path=model_path,
        #     output_results_path=results_path,
        #     output_logs_dir=logs_dir
        # )
        
        # Log model and results
        log_artifact_if_exists(model_path)
        log_artifact_if_exists(results_path)
        
        # Log parsed logs
        log_directory_artifacts(logs_dir)
    
    print("Drain workflow tracking completed")
    return True

if __name__ == "__main__":
    print("Tracking Drain log parsing model with MLflow")
    track_drain_existing_model()
    # Uncomment to simulate full workflow tracking:
    # track_drain_workflow()
    print("Drain tracking completed")
