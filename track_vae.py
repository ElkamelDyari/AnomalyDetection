"""
MLflow tracking for VAE model in the AnomalyDetection project
"""
import os
import json
import mlflow
from mlflow_utils import init_tracking, log_model_params_from_json, log_artifact_if_exists, log_directory_artifacts, log_numpy_array_summary

def track_vae_existing_model(base_dir="data/VAE", original_data_path="data/SimCLR/SimCLR_data.npy"):
    """
    Track an existing VAE model that has already been trained
    Logs model parameters, artifacts, and visualizations
    """
    init_tracking()
    
    # Define common paths
    json_path = os.path.join(base_dir, "best_vae_architecture.json")
    model_path = os.path.join(base_dir, "final_vae_pcdarts.pth")
    transformed_path = os.path.join(base_dir, "vae_transformed.npy")
    plots_dir = os.path.join(base_dir, "plots")
    
    # Verify model file exists
    if not os.path.exists(model_path):
        print(f"VAE model file not found at {model_path}")
        return False
        
    print(f"Tracking existing VAE model at {model_path}")
    
    with mlflow.start_run(run_name="VAE_Model"):
        # Log model architecture parameters
        if os.path.exists(json_path):
            log_model_params_from_json(json_path)
        
        # Log additional parameters
        mlflow.log_param("original_data_path", original_data_path)
        mlflow.log_param("model_type", "VAE_PCDARTS")
        
        # Log the model file
        log_artifact_if_exists(model_path)
        
        # Log transformed data summary
        if os.path.exists(transformed_path):
            log_numpy_array_summary(transformed_path, "vae_latent")
            log_artifact_if_exists(transformed_path)
        
        # Log visualizations if they exist
        if os.path.exists(plots_dir):
            vis_count = log_directory_artifacts(plots_dir)
            mlflow.log_metric("visualization_count", vis_count)
        
        print("VAE model tracking completed")
        return True

def track_vae_workflow(base_dir="data/VAE", original_data_path="data/SimCLR/SimCLR_data.npy"):
    """
    Track the full VAE workflow with MLflow by modifying the existing workflow
    This would be integrated into the VAE.py file
    """
    init_tracking()
    
    # Define paths
    json_path = os.path.join(base_dir, "best_vae_architecture.json")
    model_path = os.path.join(base_dir, "final_vae_pcdarts.pth")
    transformed_path = os.path.join(base_dir, "vae_transformed.npy")
    plots_dir = os.path.join(base_dir, "plots")
    
    # Ensure output directories exist
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Track architecture search
    with mlflow.start_run(run_name="VAE_Architecture_Search"):
        mlflow.log_param("data_path", original_data_path)
        mlflow.log_param("n_trials", 2)
        
        # The actual search function would be called here in the real workflow:
        # find_best_vae_architecture(
        #     data_path=original_data_path,
        #     json_out=json_path,
        #     n_trials=2
        # )
        
        # After search is done, log the best parameters
        log_model_params_from_json(json_path)
    
    # Track model training
    with mlflow.start_run(run_name="VAE_Training"):
        mlflow.log_param("data_path", original_data_path)
        mlflow.log_param("model_type", "VAE_PCDARTS")
        
        # The actual training function would be called here:
        # train_vae_model(
        #     json_path=json_path,
        #     data_path=original_data_path,
        #     model_out=model_path,
        #     transformed_out=transformed_path
        # )
        
        # Log the trained model
        log_artifact_if_exists(model_path)
        
        # Log the transformed data
        if os.path.exists(transformed_path):
            log_numpy_array_summary(transformed_path, "vae_latent")
            log_artifact_if_exists(transformed_path)
    
    # Track evaluation and visualization
    with mlflow.start_run(run_name="VAE_Evaluation"):
        mlflow.log_param("original_data", original_data_path)
        mlflow.log_param("transformed_data", transformed_path)
        mlflow.log_param("model_path", model_path)
        
        # The actual evaluation function would be called here:
        # evaluate_and_visualize_vae(
        #     json_path=json_path,
        #     original_data_path=original_data_path,
        #     transformed_path=transformed_path,
        #     model_path=model_path,
        #     output_dir=plots_dir
        # )
        
        # Log the visualizations
        log_directory_artifacts(plots_dir)
    
    print("VAE workflow tracking completed")
    return True

if __name__ == "__main__":
    print("Tracking VAE model with MLflow")
    track_vae_existing_model()
    # Uncomment to simulate full workflow tracking:
    # track_vae_workflow()
    print("VAE tracking completed")
