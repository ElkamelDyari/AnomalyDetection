"""
Train VAE model with best PCDarts architecture from MLflow

This script retrieves the best PCDarts architecture for VAE from MLflow,
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

# Import VAE components
from models.VAE_utils import (
    VAE_PCDARTS,
    vae_loss,
    train_vae,
    train_vae_model
)

# Import MLflow utilities
from mlflow_utils import init_tracking
def get_best_vae_params(experiment_name="VAE_PCDarts_Search"):
    """
    Retrieve the best VAE architecture parameters from MLflow
    
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
        # Return some default parameters based on our search
        return {
            "num_cells_enc": 3,
            "num_mixed_ops_enc": 3,
            "hidden_dim_enc": 242,
            "num_cells_dec": 3,
            "num_mixed_ops_dec": 2,
            "hidden_dim_dec": 93,
            "latent_dim": 27,
            "lr": 0.004103738940783192,
            "beta": 1.3664758364970262,
            "batch_size": 62
        }
    
    # Get the run with "Best_Trial_Summary" in the name
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string="tags.mlflow.runName LIKE '%Best_Trial_Summary%'")
    
    if len(runs) == 0:
        print(f"No 'Best_Trial_Summary' run found in experiment '{experiment_name}'.")
        # Try to find any run with parameters
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string="params.latent_dim != ''")
        if len(runs) == 0:
            # Return some default parameters
            return {
                "num_cells_enc": 3,
                "num_mixed_ops_enc": 3,
                "hidden_dim_enc": 242,
                "num_cells_dec": 3,
                "num_mixed_ops_dec": 2,
                "hidden_dim_dec": 93,
                "latent_dim": 27,
                "lr": 0.004103738940783192,
                "beta": 1.3664758364970262,
                "batch_size": 62
            }
    
    # Get the parameters from the best run
    best_run = runs.iloc[0]
    print(f"Found best VAE parameters from run: {best_run.run_id}")
    
    # Extract parameters
    params = {
        "num_cells_enc": int(best_run["params.num_cells_enc"]),
        "num_mixed_ops_enc": int(best_run["params.num_mixed_ops_enc"]),
        "hidden_dim_enc": int(best_run["params.hidden_dim_enc"]),
        "num_cells_dec": int(best_run["params.num_cells_dec"]),
        "num_mixed_ops_dec": int(best_run["params.num_mixed_ops_dec"]),
        "hidden_dim_dec": int(best_run["params.hidden_dim_dec"]),
        "latent_dim": int(best_run["params.latent_dim"]),
        "lr": float(best_run["params.lr"]),
        "beta": float(best_run["params.beta"]),
        "batch_size": int(best_run["params.batch_size"])
    }
    
    return params

def train_and_log_vae(data_path, output_dir="vae_models", experiment_name="VAE_Training", epochs=20):
    """
    Train VAE model with best architecture and log to MLflow
    
    Args:
        data_path: Path to input data (numpy array with features and labels)
        output_dir: Directory to save model artifacts
        experiment_name: Name for the new MLflow experiment
        epochs: Number of epochs to train
        
    Returns:
        Path to saved model file
    """
    # Get best parameters
    params = get_best_vae_params()
    print(f"Training with parameters: {params}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize MLflow
    mlflow = init_tracking()
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    
    # Save parameters to JSON
    arch_json_path = os.path.join(output_dir, "best_vae_architecture.json")
    with open(arch_json_path, "w") as f:
        json.dump(params, f, indent=2)
    
    # Prepare output model path
    output_model_path = os.path.join(output_dir, "final_vae_pcdarts.pth")
    
    # Start MLflow run
    with mlflow.start_run(run_name="VAE_Best_PCDarts"):
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("input_data", data_path)
        
        # Load data
        data = np.load(data_path)
        X = data[:, :-1]  # Remove label column
        input_dim = X.shape[1]
        mlflow.log_param("input_dim", input_dim)
        
        # Create and train VAE model directly
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Create model
        model = VAE_PCDARTS(
            input_dim=input_dim,
            hidden_dim_enc=params["hidden_dim_enc"],
            hidden_dim_dec=params["hidden_dim_dec"],
            latent_dim=params["latent_dim"],
            num_cells_enc=params["num_cells_enc"],
            num_mixed_ops_enc=params["num_mixed_ops_enc"],
            num_cells_dec=params["num_cells_dec"],
            num_mixed_ops_dec=params["num_mixed_ops_dec"]
        )
        
        # Move to device
        model = model.to(device)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        
        # Define custom training loop
        print(f"Training VAE for {epochs} epochs...")
        X_tensor = torch.tensor(X, dtype=torch.float32)
        ds = torch.utils.data.TensorDataset(X_tensor)
        loader = torch.utils.data.DataLoader(ds, batch_size=params["batch_size"], shuffle=True, pin_memory=False)
        
        model.train()
        loss_hist, recon_hist, kl_hist = [], [], []
        
        for epoch in range(epochs):
            total_loss = total_recon = total_kl = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                x_recon, mu, logvar = model(batch)
                loss, recon_l, kl_l = vae_loss(batch, x_recon, mu, logvar, params["beta"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.size(0)
                total_recon += recon_l.item() * batch.size(0)
                total_kl += kl_l.item() * batch.size(0)
            
            # Calculate average loss
            n = len(ds)
            loss_hist.append(total_loss/n)
            recon_hist.append(total_recon/n)
            kl_hist.append(total_kl/n)
            
            # Log metrics
            mlflow.log_metrics({
                "loss": loss_hist[-1],
                "recon_loss": recon_hist[-1],
                "kl_loss": kl_hist[-1]
            }, step=epoch)
            
            print(f"Epoch {epoch+1}/{epochs} Loss={loss_hist[-1]:.4f} Recon={recon_hist[-1]:.4f} KL={kl_hist[-1]:.4f}")
        
        # Save model
        torch.save(model.state_dict(), output_model_path)
        
        # Log model and architecture
        mlflow.log_artifact(output_model_path)
        mlflow.log_artifact(arch_json_path)
        
        # Create and log loss curve
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(loss_hist, label='Total Loss')
        ax.plot(recon_hist, label='Reconstruction Loss')
        ax.plot(kl_hist, label='KL Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        loss_curve_path = os.path.join(output_dir, "loss_curves.png")
        plt.savefig(loss_curve_path)
        mlflow.log_artifact(loss_curve_path)
        plt.close()
        
        # Generate latent space visualization
        if data.shape[1] > params["latent_dim"]:
            try:
                # Generate latent representations
                model.eval()
                with torch.no_grad():
                    batches = []
                    for (batch,) in loader:
                        batch = batch.to(device)
                        mu, _ = model.encode(batch)
                        batches.append(mu.cpu().numpy())
                    
                    latent_points = np.vstack(batches)
                    
                # If we have labels, use them for coloring
                if data.shape[1] > X.shape[1]:
                    labels = data[:, -1]
                    
                    # Create 2D visualization with t-SNE if available
                    from sklearn.manifold import TSNE
                    tsne = TSNE(n_components=2, random_state=42)
                    latent_2d = tsne.fit_transform(latent_points)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    scatter = ax.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
                    ax.set_title('t-SNE of VAE Latent Space')
                    plt.colorbar(scatter, ax=ax, label='Class')
                    
                    latent_viz_path = os.path.join(output_dir, "latent_space.png")
                    plt.savefig(latent_viz_path)
                    mlflow.log_artifact(latent_viz_path)
                    plt.close()
            except Exception as e:
                print(f"Could not generate latent space visualization: {e}")
        
        print(f"Model trained and saved to {output_model_path}")
        print(f"Architecture saved to {arch_json_path}")
        
    return output_model_path

def main():
    """Main function to run the VAE training"""
    parser = argparse.ArgumentParser(description="Train VAE with best PCDarts architecture from MLflow")
    parser.add_argument('--data-path', type=str, default="data/SimCLR/SimCLR_data.npy",
                      help='Path to input data (numpy array)')
    parser.add_argument('--output-dir', type=str, default='vae_models',
                      help='Directory to save model artifacts')
    parser.add_argument('--experiment-name', type=str, default='VAE_Training',
                      help='Name for the MLflow experiment')
    parser.add_argument('--epochs', type=int, default=20,
                      help='Number of epochs to train')
    args = parser.parse_args()
    
    train_and_log_vae(
        data_path=args.data_path,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        epochs=args.epochs
    )

if __name__ == "__main__":
    main()
