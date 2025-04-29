"""
VAE PCDarts Architecture Search with MLflow Tracking

This script performs architecture search for VAE using PCDarts and Optuna,
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

# Import VAE components
from models.VAE_utils import (
    VAE_PCDARTS,
    vae_loss,
    train_vae,
    find_best_vae_architecture,
)

# Import MLflow utilities
from mlflow_utils import init_tracking

def add_vae_search_space_args(parser):
    """Add arguments for VAE architecture search"""
    # Data and output paths
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to input data (numpy array)')
    parser.add_argument('--output-dir', type=str, default='vae_search_output',
                        help='Directory to save search results')
    
    # Experiment configuration
    parser.add_argument('--experiment-name', type=str, default='VAE_PCDarts_Search',
                        help='Name of MLflow experiment')
    parser.add_argument('--n-trials', type=int, default=20,
                        help='Number of Optuna trials')
    parser.add_argument('--epochs-per-trial', type=int, default=5,
                        help='Number of epochs per trial')
    
    # Search space parameters
    parser.add_argument('--cells-enc-min', type=int, default=1,
                        help='Min number of encoder cells')
    parser.add_argument('--cells-enc-max', type=int, default=3,
                        help='Max number of encoder cells')
    parser.add_argument('--cells-dec-min', type=int, default=1,
                        help='Min number of decoder cells')
    parser.add_argument('--cells-dec-max', type=int, default=3,
                        help='Max number of decoder cells')
    parser.add_argument('--mixed-ops-enc-min', type=int, default=1,
                        help='Min number of mixed ops in encoder')
    parser.add_argument('--mixed-ops-enc-max', type=int, default=3,
                        help='Max number of mixed ops in encoder')
    parser.add_argument('--mixed-ops-dec-min', type=int, default=1,
                        help='Min number of mixed ops in decoder')
    parser.add_argument('--mixed-ops-dec-max', type=int, default=3,
                        help='Max number of mixed ops in decoder')
    parser.add_argument('--hidden-dim-enc-min', type=int, default=64,
                        help='Min hidden dimension for encoder')
    parser.add_argument('--hidden-dim-enc-max', type=int, default=256,
                        help='Max hidden dimension for encoder')
    parser.add_argument('--hidden-dim-dec-min', type=int, default=64,
                        help='Min hidden dimension for decoder')
    parser.add_argument('--hidden-dim-dec-max', type=int, default=256,
                        help='Max hidden dimension for decoder')
    parser.add_argument('--latent-dim-min', type=int, default=8,
                        help='Min latent dimension')
    parser.add_argument('--latent-dim-max', type=int, default=64,
                        help='Max latent dimension')
    parser.add_argument('--lr-min', type=float, default=1e-4,
                        help='Min learning rate')
    parser.add_argument('--lr-max', type=float, default=1e-2,
                        help='Max learning rate')
    parser.add_argument('--beta-min', type=float, default=0.5,
                        help='Min beta value for KL weight')
    parser.add_argument('--beta-max', type=float, default=2.0,
                        help='Max beta value for KL weight')
    parser.add_argument('--batch-size-min', type=int, default=32,
                        help='Min batch size')
    parser.add_argument('--batch-size-max', type=int, default=128,
                        help='Max batch size')
    
    return parser

def find_best_vae_architecture_with_mlflow(data_path, output_dir, experiment_name, 
                                          n_trials=20, epochs_per_trial=5, **search_space):
    """
    Run VAE architecture search with Optuna and log results to MLflow
    
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
    arr = np.load(data_path)
    X = arr[:, :-1]  # Features
    labels = arr[:, -1]  # Labels
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define Optuna objective function
    def objective(trial):
        # Start MLflow run for this trial
        with mlflow.start_run(run_name=f"Trial_{trial.number}"):
            # Log trial number
            mlflow.log_param("trial_number", trial.number)
            
            # Sample hyperparameters from search space
            num_cells_enc = trial.suggest_int("num_cells_enc", 
                                            search_space.get('cells_enc_min', 1), 
                                            search_space.get('cells_enc_max', 3))
            num_mixed_enc = trial.suggest_int("num_mixed_ops_enc", 
                                            search_space.get('mixed_ops_enc_min', 1), 
                                            search_space.get('mixed_ops_enc_max', 3))
            hid_enc = trial.suggest_int("hidden_dim_enc", 
                                        search_space.get('hidden_dim_enc_min', 64), 
                                        search_space.get('hidden_dim_enc_max', 256))
            num_cells_dec = trial.suggest_int("num_cells_dec", 
                                            search_space.get('cells_dec_min', 1), 
                                            search_space.get('cells_dec_max', 3))
            num_mixed_dec = trial.suggest_int("num_mixed_ops_dec", 
                                            search_space.get('mixed_ops_dec_min', 1), 
                                            search_space.get('mixed_ops_dec_max', 3))
            hid_dec = trial.suggest_int("hidden_dim_dec", 
                                        search_space.get('hidden_dim_dec_min', 64), 
                                        search_space.get('hidden_dim_dec_max', 256))
            latent_dim = trial.suggest_int("latent_dim", 
                                        search_space.get('latent_dim_min', 8), 
                                        search_space.get('latent_dim_max', 64))
            lr = trial.suggest_float("lr", 
                                    search_space.get('lr_min', 1e-4), 
                                    search_space.get('lr_max', 1e-2), 
                                    log=True)
            beta = trial.suggest_float("beta", 
                                    search_space.get('beta_min', 0.5), 
                                    search_space.get('beta_max', 2.0))
            batch_size = trial.suggest_int("batch_size", 
                                        search_space.get('batch_size_min', 32), 
                                        search_space.get('batch_size_max', 128))
            
            # Log hyperparameters to MLflow
            mlflow.log_params({
                "num_cells_enc": num_cells_enc,
                "num_mixed_ops_enc": num_mixed_enc,
                "hidden_dim_enc": hid_enc,
                "num_cells_dec": num_cells_dec,
                "num_mixed_ops_dec": num_mixed_dec,
                "hidden_dim_dec": hid_dec,
                "latent_dim": latent_dim,
                "lr": lr,
                "beta": beta,
                "batch_size": batch_size,
                "epochs": epochs_per_trial,
                "input_dim": X.shape[1],
            })
            
            # Log dataset information
            mlflow.log_param("dataset_shape", str(X.shape))
            mlflow.log_param("num_samples", X.shape[0])
            
            # Initialize model
            model = VAE_PCDARTS(X.shape[1], hid_enc, hid_dec, latent_dim,
                               num_cells_enc, num_mixed_enc, num_cells_dec, num_mixed_dec)
            model = model.to(device)
            
            # Set up optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # Define our own custom train function to avoid the pin_memory issue
            model.to(device).train()
            loss_hist, recon_hist, kl_hist = [], [], []
            
            # Create dataset from NumPy array (keep on CPU)
            X_tensor = torch.tensor(X, dtype=torch.float32)
            ds = torch.utils.data.TensorDataset(X_tensor)
            loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=False)
            
            for epoch in range(epochs_per_trial):
                total_loss = total_recon = total_kl = 0.0
                for (batch,) in loader:
                    # Transfer batch to device inside loop
                    batch = batch.to(device)
                    x_recon, mu, logvar = model(batch)
                    loss, recon_l, kl_l = vae_loss(batch, x_recon, mu, logvar, beta)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * batch.size(0)
                    total_recon += recon_l.item() * batch.size(0)
                    total_kl += kl_l.item() * batch.size(0)
                
                n = len(ds)
                loss_hist.append(total_loss/n)
                recon_hist.append(total_recon/n)
                kl_hist.append(total_kl/n)
            
            # Log metrics for each epoch
            for epoch in range(epochs_per_trial):
                mlflow.log_metrics({
                    "total_loss": loss_hist[epoch],
                    "reconstruction_loss": recon_hist[epoch],
                    "kl_loss": kl_hist[epoch]
                }, step=epoch)
            
            # Log final metrics
            final_metrics = {
                "final_loss": loss_hist[-1],
                "final_recon_loss": recon_hist[-1],
                "final_kl_loss": kl_hist[-1],
            }
            mlflow.log_metrics(final_metrics)
            
            # Log loss curves
            fig, ax = plt.figure(figsize=(10, 6)), plt.subplot(111)
            ax.plot(loss_hist, label='Total Loss')
            ax.plot(recon_hist, label='Reconstruction Loss')
            ax.plot(kl_hist, label='KL Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)
            
            # Save and log figure
            loss_curve_path = os.path.join(output_dir, f"trial_{trial.number}_loss_curve.png")
            plt.savefig(loss_curve_path)
            mlflow.log_artifact(loss_curve_path)
            plt.close()
            
            # Log model architecture summary (as text)
            arch_summary = f"""
            VAE_PCDARTS Architecture:
            - Input dimension: {X.shape[1]}
            - Encoder: {num_cells_enc} cells with {num_mixed_enc} mixed ops each, hidden dim {hid_enc}
            - Latent dimension: {latent_dim}
            - Decoder: {num_cells_dec} cells with {num_mixed_dec} mixed ops each, hidden dim {hid_dec}
            - Total parameters: {sum(p.numel() for p in model.parameters())}
            """
            arch_summary_path = os.path.join(output_dir, f"trial_{trial.number}_architecture.txt")
            with open(arch_summary_path, "w") as f:
                f.write(arch_summary)
            mlflow.log_artifact(arch_summary_path)
            
            return loss_hist[-1]  # Return final loss for Optuna to minimize
    
    # Create Optuna study and run optimization
    study = optuna.create_study(direction="minimize", 
                              study_name=f"VAE_PCDarts_{Path(data_path).stem}")
    study.optimize(objective, n_trials=n_trials)
    
    # Log best trial information
    with mlflow.start_run(run_name="Best_Trial_Summary"):
        best_trial = study.best_trial
        mlflow.log_params(best_trial.params)
        mlflow.log_metric("best_loss", best_trial.value)
        
        # Save best parameters to JSON
        best_params_path = os.path.join(output_dir, "best_vae_architecture.json")
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
    
    print(f"Best VAE architecture search results saved to {output_dir}")
    print(f"Best parameters: {study.best_trial.params}")
    print(f"Best loss value: {study.best_trial.value}")
    
    return study.best_trial.params

def main():
    """Main function to run the VAE architecture search"""
    parser = argparse.ArgumentParser(description="VAE PCDarts Architecture Search with MLflow Tracking")
    parser = add_vae_search_space_args(parser)
    args = parser.parse_args()
    
    # Extract search space from arguments
    search_space = {
        'cells_enc_min': args.cells_enc_min,
        'cells_enc_max': args.cells_enc_max,
        'cells_dec_min': args.cells_dec_min,
        'cells_dec_max': args.cells_dec_max,
        'mixed_ops_enc_min': args.mixed_ops_enc_min,
        'mixed_ops_enc_max': args.mixed_ops_enc_max,
        'mixed_ops_dec_min': args.mixed_ops_dec_min,
        'mixed_ops_dec_max': args.mixed_ops_dec_max,
        'hidden_dim_enc_min': args.hidden_dim_enc_min,
        'hidden_dim_enc_max': args.hidden_dim_enc_max,
        'hidden_dim_dec_min': args.hidden_dim_dec_min,
        'hidden_dim_dec_max': args.hidden_dim_dec_max,
        'latent_dim_min': args.latent_dim_min,
        'latent_dim_max': args.latent_dim_max,
        'lr_min': args.lr_min,
        'lr_max': args.lr_max,
        'beta_min': args.beta_min,
        'beta_max': args.beta_max,
        'batch_size_min': args.batch_size_min,
        'batch_size_max': args.batch_size_max,
    }
    
    # Run architecture search
    find_best_vae_architecture_with_mlflow(
        data_path=args.data_path,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        n_trials=args.n_trials,
        epochs_per_trial=args.epochs_per_trial,
        **search_space
    )

if __name__ == "__main__":
    main()
