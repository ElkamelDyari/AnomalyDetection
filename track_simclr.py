"""
MLflow tracking for SimCLR model in the AnomalyDetection project
"""
import os
import json
import mlflow
from mlflow_utils import init_tracking, log_model_params_from_json, log_artifact_if_exists, log_directory_artifacts, log_numpy_array_summary

def track_simclr_existing_model(base_dir="data/SimCLR"):
    """
    Track an existing SimCLR model that has already been trained
    Logs model parameters, artifacts, and visualizations
    """
    init_tracking()
    
    # Define common paths
    model_path = os.path.join(base_dir, "simclr_head.pth")
    json_path = os.path.join(base_dir, "best_arch.json")
    data_path = os.path.join(base_dir, "SimCLR_data.npy")
    original_data_path = os.path.join(base_dir, "test_cls_embeddings_with_label.npy")
    vis_dir = os.path.join(base_dir, "visualizations")
    
    # Verify files exist
    if not os.path.exists(model_path):
        print(f"SimCLR model file not found at {model_path}")
        return False
        
    print(f"Tracking existing SimCLR model at {model_path}")
    
    with mlflow.start_run(run_name="SimCLR_Model"):
        # Log model architecture parameters
        if os.path.exists(json_path):
            log_model_params_from_json(json_path)
        
        # Log the model file
        log_artifact_if_exists(model_path)
        
        # Log transformed data summary
        if os.path.exists(data_path):
            log_numpy_array_summary(data_path, "simclr_data")
            log_artifact_if_exists(data_path)
        
        # Log visualizations if they exist
        if os.path.exists(vis_dir):
            vis_count = log_directory_artifacts(vis_dir)
            mlflow.log_metric("visualization_count", vis_count)
        
        print("SimCLR model tracking completed")
        return True

def track_simclr_workflow(base_dir="data/SimCLR"):
    """
    Track the full SimCLR workflow with MLflow by modifying the existing workflow
    This would be integrated into the SimCLR.py file
    """
    init_tracking()
    
    # Define paths
    train_npy_path = os.path.join(base_dir, "train_cls_embeddings_with_label.npy")
    test_npy_path = os.path.join(base_dir, "test_cls_embeddings_with_label.npy")
    json_path = os.path.join(base_dir, "best_arch.json")
    model_path = os.path.join(base_dir, "simclr_head.pth")
    data_path = os.path.join(base_dir, "SimCLR_data.npy")
    vis_dir = os.path.join(base_dir, "visualizations")
    
    # Track architecture search
    with mlflow.start_run(run_name="SimCLR_Architecture_Search"):
        mlflow.log_param("data_path", train_npy_path)
        mlflow.log_param("n_trials", 5)
        
        # The actual search function would be called here in the real workflow:
        # best = search_best_architecture(
        #     train_npy_path=train_npy_path,
        #     output_json_path=json_path,
        #     n_trials=5
        # )
        
        # After search is done, log the best parameters
        log_model_params_from_json(json_path)
    
    # Track model training
    with mlflow.start_run(run_name="SimCLR_Training"):
        mlflow.log_param("epochs", 5)
        mlflow.log_param("data_path", train_npy_path)
        
        # The actual training function would be called here:
        # model = train_final_model(
        #     train_npy_path=train_npy_path,
        #     arch_json_path=json_path,
        #     output_model_path=model_path,
        #     epochs=5
        # )
        
        # Log the trained model
        log_artifact_if_exists(model_path)
    
    # Track data transformation
    with mlflow.start_run(run_name="SimCLR_Data_Transformation"):
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("input_data", test_npy_path)
        
        # The actual transform function would be called here:
        # sim_data = transform_data(
        #     model_path=model_path,
        #     input_npy_path=test_npy_path,
        #     output_npy_path=data_path,
        #     arch_json_path=json_path
        # )
        
        # Log the transformed data
        log_numpy_array_summary(data_path, "simclr_data")
        log_artifact_if_exists(data_path)
    
    # Track evaluation and visualization
    with mlflow.start_run(run_name="SimCLR_Evaluation"):
        mlflow.log_param("original_data", test_npy_path)
        mlflow.log_param("transformed_data", data_path)
        
        # The actual evaluation function would be called here:
        # evaluate_and_plot(
        #     original_npy=test_npy_path,
        #     simclr_npy=data_path,
        #     output_fig=os.path.join(vis_dir, "comparison.png")
        # )
        
        # Log the visualizations
        log_directory_artifacts(vis_dir)
    
    print("SimCLR workflow tracking completed")
    return True

if __name__ == "__main__":
    print("Tracking SimCLR model with MLflow")
    track_simclr_existing_model()
    # Uncomment to simulate full workflow tracking:
    # track_simclr_workflow()
    print("SimCLR tracking completed")
