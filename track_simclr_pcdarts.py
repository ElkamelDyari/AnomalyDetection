"""
SimCLR PCDarts Architecture Search with MLflow Tracking

This script performs architecture search for SimCLR projection head using PCDarts and Optuna,
tracking all trials, parameters, and results in MLflow.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import mlflow
import dagshub
import optuna
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import SimCLR components
from models.SimCLR_utils import (
    ProjectionHeadPCDARTS,
    nt_xent_loss,
    load_embeddings_with_labels,
    search_best_architecture
)

# Import MLflow utilities
from mlflow_utils import init_tracking

def add_simclr_search_space_args(parser):
    """Add arguments for SimCLR architecture search"""
    # Data and output paths
    parser.add_argument('--data-path', type=str, required=True,
                      help='Path to input data (numpy array)')
    parser.add_argument('--output-dir', type=str, default='simclr_search_output',
                      help='Directory to save search results')
    
    # Experiment configuration
    parser.add_argument('--experiment-name', type=str, default='SimCLR_PCDarts_Search',
                      help='Name of MLflow experiment')
    parser.add_argument('--n-trials', type=int, default=20,
                      help='Number of Optuna trials')
    parser.add_argument('--epochs-per-trial', type=int, default=5,
                      help='Number of epochs per trial')
    
    # Search space parameters
    parser.add_argument('--num-cells-min', type=int, default=2,
                      help='Min number of cells')
    parser.add_argument('--num-cells-max', type=int, default=4,
                      help='Max number of cells')
    parser.add_argument('--num-mixed-ops-min', type=int, default=1,
                      help='Min number of mixed ops')
    parser.add_argument('--num-mixed-ops-max', type=int, default=4,
                      help='Max number of mixed ops')
    parser.add_argument('--hidden-dim-min', type=int, default=128,
                      help='Min hidden dimension')
    parser.add_argument('--hidden-dim-max', type=int, default=512,
                      help='Max hidden dimension')
    parser.add_argument('--proj-dim-min', type=int, default=32,
                      help='Min projection dimension')
    parser.add_argument('--proj-dim-max', type=int, default=128,
                      help='Max projection dimension')
    parser.add_argument('--lr-min', type=float, default=1e-4,
                      help='Min learning rate')
    parser.add_argument('--lr-max', type=float, default=1e-2,
                      help='Max learning rate')
    parser.add_argument('--temperature-min', type=float, default=0.1,
                      help='Min temperature for contrastive loss')
    parser.add_argument('--temperature-max', type=float, default=1.0,
                      help='Max temperature for contrastive loss')
    parser.add_argument('--noise-std-min', type=float, default=0.05,
                      help='Min noise standard deviation')
    parser.add_argument('--noise-std-max', type=float, default=0.2,
                      help='Max noise standard deviation')
    parser.add_argument('--batch-size-min', type=int, default=32,
                      help='Min batch size')
    parser.add_argument('--batch-size-max', type=int, default=128,
                      help='Max batch size')
    
    return parser

def find_best_simclr_architecture_with_mlflow(data_path, output_dir, experiment_name, 
                                             n_trials=20, epochs_per_trial=5, **search_space):
    """
    Run SimCLR architecture search with Optuna and log results to MLflow
    
    Args:
        data_path: Path to input data (numpy array with features and labels)
        output_dir: Directory to save search results
        experiment_name: Name of MLflow experiment
        n_trials: Number of Optuna trials
        epochs_per_trial: Number of epochs per trial
        search_space: Dictionary with search space parameters
    
    Returns:
        Best trial parameters
    """
    # Initialize MLflow
    init_tracking()
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    feats, labels = load_embeddings_with_labels(data_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define Optuna objective function
    def objective(trial):
        # Start MLflow run for this trial
        with mlflow.start_run(run_name=f"Trial_{trial.number}"):
            # Log trial number
            mlflow.log_param("trial_number", trial.number)
            
            # Sample hyperparameters from search space
            num_cells = trial.suggest_int("num_cells", 
                                         search_space.get('num_cells_min', 2), 
                                         search_space.get('num_cells_max', 4))
            num_mixed_ops = trial.suggest_int("num_mixed_ops", 
                                             search_space.get('num_mixed_ops_min', 1), 
                                             search_space.get('num_mixed_ops_max', 4))
            hidden_dim = trial.suggest_int("hidden_dim", 
                                          search_space.get('hidden_dim_min', 128), 
                                          search_space.get('hidden_dim_max', 512))
            proj_dim = trial.suggest_int("proj_dim", 
                                        search_space.get('proj_dim_min', 32), 
                                        search_space.get('proj_dim_max', 128))
            lr = trial.suggest_float("lr", 
                                    search_space.get('lr_min', 1e-4), 
                                    search_space.get('lr_max', 1e-2), 
                                    log=True)
            temp = trial.suggest_float("temperature", 
                                      search_space.get('temperature_min', 0.1), 
                                      search_space.get('temperature_max', 1.0))
            noise_std = trial.suggest_float("noise_std", 
                                          search_space.get('noise_std_min', 0.05), 
                                          search_space.get('noise_std_max', 0.2))
            batch_size = trial.suggest_int("batch_size", 
                                         search_space.get('batch_size_min', 32), 
                                         search_space.get('batch_size_max', 128))
            
            # Log hyperparameters to MLflow
            mlflow.log_params({
                "num_cells": num_cells,
                "num_mixed_ops": num_mixed_ops,
                "hidden_dim": hidden_dim,
                "proj_dim": proj_dim,
                "lr": lr,
                "temperature": temp,
                "noise_std": noise_std,
                "batch_size": batch_size,
                "epochs": epochs_per_trial,
                "input_dim": feats.shape[1],
            })
            
            # Log dataset information
            mlflow.log_param("dataset_shape", str(feats.shape))
            mlflow.log_param("num_samples", feats.shape[0])
            
            # Train and evaluate model for multiple epochs
            model = ProjectionHeadPCDARTS(feats.shape[1], hidden_dim, proj_dim,
                                       num_cells, num_mixed_ops).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # Train for multiple epochs and record loss
            loss_history = []
            
            for epoch in range(epochs_per_trial):
                epoch_losses = []
                
                # Process data in batches
                for _ in range(10):  # Process 10 batches per epoch for evaluation
                    idx = np.random.choice(len(feats), batch_size, replace=False)
                    x = torch.tensor(feats[idx], dtype=torch.float32).to(device)
                    
                    # Forward pass with noise for contrastive learning
                    optimizer.zero_grad()
                    z1 = model(x + torch.randn_like(x)*noise_std)
                    z2 = model(x + torch.randn_like(x)*noise_std)
                    
                    # Calculate contrastive loss
                    loss = nt_xent_loss(z1, z2, temp)
                    epoch_losses.append(loss.item())
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                
                # Calculate average loss for this epoch
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                loss_history.append(avg_loss)
                
                # Log metrics for this epoch
                mlflow.log_metric("contrastive_loss", avg_loss, step=epoch)
            
            # Log final metrics
            final_loss = loss_history[-1]
            mlflow.log_metric("final_loss", final_loss)
            
            # Log loss curve
            fig, ax = plt.figure(figsize=(10, 6)), plt.subplot(111)
            ax.plot(loss_history)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Contrastive Loss')
            ax.set_title('Training Loss')
            ax.grid(True)
            
            # Save and log figure
            loss_curve_path = os.path.join(output_dir, f"trial_{trial.number}_loss_curve.png")
            plt.savefig(loss_curve_path)
            mlflow.log_artifact(loss_curve_path)
            plt.close()
            
            # Log model architecture summary (as text)
            arch_summary = f"""
            SimCLR ProjectionHeadPCDARTS Architecture:
            - Input dimension: {feats.shape[1]}
            - Hidden dimension: {hidden_dim}
            - Projection dimension: {proj_dim}
            - Number of cells: {num_cells}
            - Number of mixed operations per cell: {num_mixed_ops}
            - Total parameters: {sum(p.numel() for p in model.parameters())}
            """
            arch_summary_path = os.path.join(output_dir, f"trial_{trial.number}_architecture.txt")
            with open(arch_summary_path, "w") as f:
                f.write(arch_summary)
            mlflow.log_artifact(arch_summary_path)
            
            # Log model (PyTorch)
            model_path = os.path.join(output_dir, f"trial_{trial.number}_model.pth")
            torch.save(model.state_dict(), model_path)
            mlflow.log_artifact(model_path)
            
            return final_loss
    
    # Create Optuna study and run optimization
    study = optuna.create_study(direction="minimize", 
                              study_name=f"SimCLR_PCDarts_{Path(data_path).stem}")
    study.optimize(objective, n_trials=n_trials)
    
    # Log best trial information
    with mlflow.start_run(run_name="Best_Trial_Summary"):
        best_trial = study.best_trial
        mlflow.log_params(best_trial.params)
        mlflow.log_metric("best_loss", best_trial.value)
        
        # Save best parameters to JSON
        best_params_path = os.path.join(output_dir, "best_simclr_architecture.json")
        with open(best_params_path, 'w') as f:
            json.dump(best_trial.params, f, indent=2)
        mlflow.log_artifact(best_params_path)
        
        # Visualization of parameter importance
        try:
            param_importances = optuna.importance.get_param_importances(study)
            fig = plt.figure(figsize=(10, 6))
            ax = plt.gca()
            importance_items = list(param_importances.items())
            names = [item[0] for item in importance_items]
            values = [item[1] for item in importance_items]
            
            ax.barh(names, values)
            ax.set_xlabel('Importance')
            ax.set_title('Hyperparameter Importance')
            
            importance_path = os.path.join(output_dir, "parameter_importance.png")
            plt.tight_layout()
            plt.savefig(importance_path)
            mlflow.log_artifact(importance_path)
            plt.close()
        except Exception as e:
            print(f"Could not generate parameter importance plot: {e}")
        
        # Log optimization history
        try:
            fig = plt.figure(figsize=(10, 6))
            # Handle different Optuna API versions
            try:
                optuna.visualization.matplotlib.plot_optimization_history(study, ax=plt.gca())
            except TypeError:
                # Older Optuna versions don't support ax parameter
                optuna.visualization.matplotlib.plot_optimization_history(study)
            
            history_path = os.path.join(output_dir, "optimization_history.png")
            plt.tight_layout()
            plt.savefig(history_path)
            mlflow.log_artifact(history_path)
            plt.close()
        except Exception as e:
            print(f"Could not generate optimization history plot: {e}")
        
        # Final training with best parameters
        print(f"Training final model with best parameters: {best_trial.params}")
        # This would be done in a separate script for actual deployment
        
    print(f"Best SimCLR architecture search results saved to {output_dir}")
    print(f"Best parameters: {study.best_trial.params}")
    print(f"Best loss value: {study.best_trial.value}")
    
    return study.best_trial.params

def main():
    """Main function to run the SimCLR architecture search"""
    parser = argparse.ArgumentParser(description="SimCLR PCDarts Architecture Search with MLflow Tracking")
    parser = add_simclr_search_space_args(parser)
    args = parser.parse_args()
    
    # Extract search space from arguments
    search_space = {
        'num_cells_min': args.num_cells_min,
        'num_cells_max': args.num_cells_max,
        'num_mixed_ops_min': args.num_mixed_ops_min,
        'num_mixed_ops_max': args.num_mixed_ops_max,
        'hidden_dim_min': args.hidden_dim_min,
        'hidden_dim_max': args.hidden_dim_max,
        'proj_dim_min': args.proj_dim_min,
        'proj_dim_max': args.proj_dim_max,
        'lr_min': args.lr_min,
        'lr_max': args.lr_max,
        'temperature_min': args.temperature_min,
        'temperature_max': args.temperature_max,
        'noise_std_min': args.noise_std_min,
        'noise_std_max': args.noise_std_max,
        'batch_size_min': args.batch_size_min,
        'batch_size_max': args.batch_size_max,
    }
    
    # Run architecture search
    find_best_simclr_architecture_with_mlflow(
        data_path=args.data_path,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        n_trials=args.n_trials,
        epochs_per_trial=args.epochs_per_trial,
        **search_space
    )

if __name__ == "__main__":
    main()
