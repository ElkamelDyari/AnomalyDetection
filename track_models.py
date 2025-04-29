"""
Main MLflow tracking script for the AnomalyDetection project
Loads and runs the tracking for different models
"""
import os
import argparse
from mlflow_utils import init_tracking
from track_simclr import track_simclr_existing_model, track_simclr_workflow
from track_vae import track_vae_existing_model, track_vae_workflow
from track_drain import track_drain_existing_model, track_drain_workflow

def main():
    """Main function to track all or selected models"""
    parser = argparse.ArgumentParser(description='Track AnomalyDetection models with MLflow')
    parser.add_argument('--all', action='store_true', help='Track all models')
    parser.add_argument('--simclr', action='store_true', help='Track SimCLR model')
    parser.add_argument('--vae', action='store_true', help='Track VAE model')
    parser.add_argument('--drain', action='store_true', help='Track Drain model')
    parser.add_argument('--workflow', action='store_true', 
                        help='Track entire workflows (warning: this may re-run models)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Base directory containing model data')
    
    args = parser.parse_args()
    
    # If no specific model is selected, default to tracking all models
    if not (args.simclr or args.vae or args.drain) and not args.all:
        args.all = True
    
    # Initialize tracking
    init_tracking()
    
    # Create a tracking summary
    tracked_models = []
    
    # Track SimCLR if selected or if tracking all
    if args.simclr or args.all:
        simclr_dir = os.path.join(args.data_dir, 'SimCLR')
        print(f"\n{'='*50}\nTracking SimCLR Model\n{'='*50}")
        if args.workflow:
            success = track_simclr_workflow(simclr_dir)
        else:
            success = track_simclr_existing_model(simclr_dir)
        if success:
            tracked_models.append("SimCLR")
    
    # Track VAE if selected or if tracking all
    if args.vae or args.all:
        vae_dir = os.path.join(args.data_dir, 'VAE')
        original_data_path = os.path.join(args.data_dir, 'SimCLR', 'SimCLR_data.npy')
        print(f"\n{'='*50}\nTracking VAE Model\n{'='*50}")
        if args.workflow:
            success = track_vae_workflow(vae_dir, original_data_path)
        else:
            success = track_vae_existing_model(vae_dir, original_data_path)
        if success:
            tracked_models.append("VAE")
    
    # Track Drain if selected or if tracking all
    if args.drain or args.all:
        drain_dir = os.path.join(args.data_dir, 'Drain')
        print(f"\n{'='*50}\nTracking Drain Model\n{'='*50}")
        if args.workflow:
            success = track_drain_workflow(drain_dir)
        else:
            success = track_drain_existing_model(drain_dir)
        if success:
            tracked_models.append("Drain")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tracking Summary")
    print(f"{'='*50}")
    if tracked_models:
        print(f"Successfully tracked {len(tracked_models)} models:")
        for model in tracked_models:
            print(f"  - {model}")
    else:
        print("No models were successfully tracked.")
    
    print(f"\nMLflow tracking is complete!")
    print(f"You can view your tracked models on DAGsHub at:")
    print(f"https://dagshub.com/ElkamelDyari/AnomalyDetection.mlflow")

if __name__ == "__main__":
    main()
