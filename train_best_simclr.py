"""
Train SimCLR model with best PCDarts architecture from MLflow

This script retrieves the best PCDarts architecture for SimCLR from MLflow,
trains a model with those parameters, and logs the trained model back to MLflow.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import mlflow
import dagshub
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import SimCLR components
from models.SimCLR_utils import (
    ProjectionHeadPCDARTS,
    nt_xent_loss,
    load_embeddings_with_labels,
    train_final_model
)

# Import MLflow utilities
from mlflow_utils import init_tracking

def get_best_simclr_params(experiment_name="SimCLR_PCDarts_Search"):
    """
    Retrieve the best SimCLR architecture parameters from MLflow
    
    Args:
        experiment_name: Name of the MLflow experiment where the search was logged
        
    Returns:
        Dictionary with best parameters
    """
    # Initialize MLflow
    init_tracking()
    
    # Find the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment '{experiment_name}' not found.")
        # Return some default parameters
        return {
            "num_cells": 3,
            "num_mixed_ops": 2,
            "hidden_dim": 165,
            "proj_dim": 99,
            "lr": 0.0024,
            "temperature": 0.39,
            "noise_std": 0.17,
            "batch_size": 67
        }
    
    # Get the run with "Best_Trial_Summary" in the name
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string="tags.mlflow.runName LIKE '%Best_Trial_Summary%'")
    
    if len(runs) == 0:
        print(f"No 'Best_Trial_Summary' run found in experiment '{experiment_name}'.")
        # Try to find any run with parameters
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string="params.num_cells != ''")
        if len(runs) == 0:
            # Return some default parameters
            return {
                "num_cells": 3,
                "num_mixed_ops": 2,
                "hidden_dim": 165,
                "proj_dim": 99,
                "lr": 0.0024,
                "temperature": 0.39,
                "noise_std": 0.17,
                "batch_size": 67
            }
    
    # Get the parameters from the best run
    best_run = runs.iloc[0]
    print(f"Found best SimCLR parameters from run: {best_run.run_id}")
    
    # Extract parameters
    params = {
        "num_cells": int(best_run["params.num_cells"]),
        "num_mixed_ops": int(best_run["params.num_mixed_ops"]),
        "hidden_dim": int(best_run["params.hidden_dim"]),
        "proj_dim": int(best_run["params.proj_dim"]),
        "lr": float(best_run["params.lr"]),
        "temperature": float(best_run["params.temperature"]),
        "noise_std": float(best_run["params.noise_std"]),
        "batch_size": int(best_run["params.batch_size"])
    }
    
    return params

def train_and_log_simclr(data_path, output_dir="simclr_models", experiment_name="SimCLR_Training", epochs=20):
    """
    Train SimCLR model with best architecture and log to MLflow
    
    Args:
        data_path: Path to input data (numpy array with features and labels)
        output_dir: Directory to save model artifacts
        experiment_name: Name for the new MLflow experiment
        epochs: Number of epochs to train
        
    Returns:
        Path to saved model file
    """
    # Get best parameters
    params = get_best_simclr_params()
    print(f"Training with parameters: {params}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize MLflow
    mlflow = init_tracking()
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    
    # Save parameters to JSON
    arch_json_path = os.path.join(output_dir, "best_arch.json")
    with open(arch_json_path, "w") as f:
        json.dump(params, f, indent=2)
    
    # Prepare output model path
    output_model_path = os.path.join(output_dir, "simclr_head.pth")
    
    # Start MLflow run
    with mlflow.start_run(run_name="SimCLR_Best_PCDarts"):
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("input_data", data_path)
        
        # Train model
        print(f"Training SimCLR model for {epochs} epochs...")
        model_info = train_final_model(
            train_npy_path=data_path,
            arch_json_path=arch_json_path,
            output_model_path=output_model_path,
            epochs=epochs
        )
        
        # Log metrics
        if model_info and hasattr(model_info, 'loss_history'):
            for i, loss in enumerate(model_info.loss_history):
                mlflow.log_metric("loss", loss, step=i)
        
        # Log model file
        mlflow.log_artifact(output_model_path)
        mlflow.log_artifact(arch_json_path)
        
        # Create and log loss curve
        if model_info and hasattr(model_info, 'loss_history'):
            fig = plt.figure(figsize=(10, 6))
            plt.plot(model_info.loss_history)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('SimCLR Training Loss')
            plt.grid(True)
            loss_curve_path = os.path.join(output_dir, "loss_curve.png")
            plt.savefig(loss_curve_path)
            mlflow.log_artifact(loss_curve_path)
            plt.close()
        
        print(f"Model saved to {output_model_path}")
        print(f"Architecture saved to {arch_json_path}")
        
    return output_model_path

def main():
    """Main function to run the SimCLR training"""
    parser = argparse.ArgumentParser(description="Train SimCLR with best PCDarts architecture from MLflow")
    parser.add_argument('--data-path', type=str, default="data/extracted_features/train_cls_embeddings_with_label.npy",
                      help='Path to input data (numpy array)')
    parser.add_argument('--output-dir', type=str, default='simclr_models',
                      help='Directory to save model artifacts')
    parser.add_argument('--experiment-name', type=str, default='SimCLR_Training',
                      help='Name for the MLflow experiment')
    parser.add_argument('--epochs', type=int, default=20,
                      help='Number of epochs to train')
    args = parser.parse_args()
    
    train_and_log_simclr(
        data_path=args.data_path,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        epochs=args.epochs
    )

if __name__ == "__main__":
    main()
