"""
MLflow tracking for the AnomalyDetection project
Logs models, parameters, and metrics to DAGsHub
"""
import os
import json
import numpy as np
import torch
import mlflow
import dagshub
from pathlib import Path

# Import model utilities
from models.VAE_utils import (
    VAE_PCDARTS,
    train_vae,
    find_best_vae_architecture,
    train_vae_model,
    evaluate_and_visualize_vae,
    load_and_transform_with_vae
)
from models.SimCLR_utils import (
    search_best_architecture,
    train_final_model,
    transform_data,
    evaluate_and_plot
)

# Initialize DAGsHub and MLflow
def init_tracking():
    """Initialize DAGsHub and MLflow tracking"""
    dagshub.init(repo_owner='ElkamelDyari', repo_name='AnomalyDetection', mlflow=True)
    print("MLflow tracking URI:", mlflow.get_tracking_uri())
    return mlflow

# SimCLR model tracking
def track_simclr_workflow(base_dir="data/SimCLR"):
    """Track the SimCLR workflow with MLflow"""
    init_tracking()
    
    # Define paths
    train_npy_path = os.path.join(base_dir, "train_cls_embeddings_with_label.npy")
    test_npy_path = os.path.join(base_dir, "test_cls_embeddings_with_label.npy")
    arch_json_path = os.path.join(base_dir, "best_arch.json")
    model_path = os.path.join(base_dir, "simclr_head.pth")
    simclr_data_path = os.path.join(base_dir, "SimCLR_data.npy")
    vis_path = os.path.join(base_dir, "visualizations", "comparison.png")
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(vis_path), exist_ok=True)

    # 1. Search for the best architecture
    with mlflow.start_run(run_name="SimCLR_Architecture_Search"):
        mlflow.log_param("dataset", train_npy_path)
        mlflow.log_param("n_trials", 5)
        
        best = search_best_architecture(
            train_npy_path=train_npy_path,
            output_json_path=arch_json_path,
            n_trials=5
        )
        
        # Log best hyperparameters
        for key, value in best.items():
            mlflow.log_param(f"best_{key}", value)
        
        # Log JSON as artifact
        mlflow.log_artifact(arch_json_path)

    # 2. Train final model
    with mlflow.start_run(run_name="SimCLR_Training"):
        # Load best architecture
        with open(arch_json_path, 'r') as f:
            arch_params = json.load(f)
        
        # Log parameters from best architecture
        for key, value in arch_params.items():
            mlflow.log_param(key, value)
        
        # Log training params
        mlflow.log_param("epochs", 5)
        mlflow.log_param("dataset", train_npy_path)
        
        # Train model and track metrics
        model = train_final_model(
            train_npy_path=train_npy_path,
            arch_json_path=arch_json_path,
            output_model_path=model_path,
            epochs=5,
            mlflow_tracking=True  # This would need to be implemented in the function
        )
        
        # Log model as artifact
        mlflow.log_artifact(model_path)

    # 3. Transform data
    with mlflow.start_run(run_name="SimCLR_Data_Transformation"):
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("input_data", test_npy_path)
        
        sim_data = transform_data(
            model_path=model_path,
            input_npy_path=test_npy_path,
            output_npy_path=simclr_data_path,
            arch_json_path=arch_json_path
        )
        
        # Log number of samples transformed
        mlflow.log_metric("num_samples", len(sim_data))
        
        # Log transformed data shape
        mlflow.log_param("transformed_data_shape", str(sim_data.shape))

    # 4. Evaluate and plot
    with mlflow.start_run(run_name="SimCLR_Evaluation"):
        mlflow.log_param("original_data", test_npy_path)
        mlflow.log_param("transformed_data", simclr_data_path)
        
        metrics = evaluate_and_plot(
            original_npy=test_npy_path,
            simclr_npy=simclr_data_path,
            output_fig=vis_path
        )
        
        # Log evaluation metrics (assuming the function is modified to return metrics)
        if metrics:
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
        
        # Log visualization
        if os.path.exists(vis_path):
            mlflow.log_artifact(vis_path)

# VAE model tracking
def track_vae_workflow(original_data_path="data/SimCLR/SimCLR_data.npy"):
    """Track the VAE workflow with MLflow"""
    init_tracking()
    
    # Define paths
    base_dir = "data/VAE"
    best_json_path = os.path.join(base_dir, "best_vae_architecture.json")
    vae_model_path = os.path.join(base_dir, "final_vae_pcdarts.pth")
    vae_transformed = os.path.join(base_dir, "vae_transformed.npy")
    output_dir = os.path.join(base_dir, "plots")
    
    # Ensure directories exist
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Find best VAE architecture
    with mlflow.start_run(run_name="VAE_Architecture_Search"):
        mlflow.log_param("data_path", original_data_path)
        mlflow.log_param("n_trials", 2)
        
        best_params = find_best_vae_architecture(
            data_path=original_data_path,
            json_out=best_json_path,
            n_trials=2
        )
        
        # Log best hyperparameters
        for key, value in best_params.items():
            mlflow.log_param(f"best_{key}", value)
        
        # Log JSON as artifact
        mlflow.log_artifact(best_json_path)

    # 2. Train final VAE model
    with mlflow.start_run(run_name="VAE_Training"):
        # Load best architecture
        with open(best_json_path, 'r') as f:
            vae_params = json.load(f)
        
        # Log parameters from best architecture
        for key, value in vae_params.items():
            mlflow.log_param(key, value)
        
        # Log additional parameters
        mlflow.log_param("dataset", original_data_path)
        mlflow.log_param("epochs", 5)  # assuming default epoch count
        
        # Train model with MLflow tracking (assuming function is modified to return metrics)
        train_metrics = {"epochs": 5}  # placeholder for actual metrics
        
        # The original function doesn't return metrics, so we need a wrapper or modification
        train_vae_model(
            json_path=best_json_path,
            data_path=original_data_path,
            model_out=vae_model_path,
            transformed_out=vae_transformed
        )
        
        # Log model as artifact
        mlflow.log_artifact(vae_model_path)
        
        # Log transformed data info
        if os.path.exists(vae_transformed):
            transformed_data = np.load(vae_transformed)
            mlflow.log_metric("num_samples", transformed_data.shape[0])
            mlflow.log_param("latent_dim", vae_params.get("latent_dim", "unknown"))

    # 3. Evaluate and visualize VAE
    with mlflow.start_run(run_name="VAE_Evaluation"):
        mlflow.log_param("original_data", original_data_path)
        mlflow.log_param("transformed_data", vae_transformed)
        mlflow.log_param("model_path", vae_model_path)
        
        # Evaluate and create visualizations
        evaluate_and_visualize_vae(
            json_path=best_json_path,
            original_data_path=original_data_path,
            transformed_path=vae_transformed,
            model_path=vae_model_path,
            output_dir=output_dir
        )
        
        # Log visualizations directory as artifacts
        for vis_file in os.listdir(output_dir):
            vis_path = os.path.join(output_dir, vis_file)
            if os.path.isfile(vis_path):
                mlflow.log_artifact(vis_path)

# Track other models if present (Drain, etc.)
def track_drain_model():
    """Track the Drain log parsing model with MLflow"""
    init_tracking()
    
    # Implement Drain model tracking here if needed
    with mlflow.start_run(run_name="Drain_Log_Parsing"):
        # Example tracking - actual implementation would depend on how Drain is used
        mlflow.log_param("model", "Drain")
        mlflow.log_param("description", "Log parsing for anomaly detection")


if __name__ == "__main__":
    # Use this script to track all your models with MLflow
    print("Starting MLflow tracking for AnomalyDetection models...")
    
    # Choose which workflows to track
    track_simclr = True
    track_vae = True
    track_drain = False
    
    if track_simclr:
        print("Tracking SimCLR workflow...")
        track_simclr_workflow()
    
    if track_vae:
        print("Tracking VAE workflow...")
        track_vae_workflow()
    
    if track_drain:
        print("Tracking Drain model...")
        track_drain_model()
    
    print("MLflow tracking complete.")
