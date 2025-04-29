"""
Prediction Pipeline for Anomaly Detection
Loads SimCLR and VAE models from MLflow and runs inference on test data
"""
import os
import json
import numpy as np
import torch
import mlflow
import dagshub
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt

# Import model utilities
from models.SimCLR_utils import transform_data as simclr_transform
from models.VAE_utils import VAE_PCDARTS, load_and_transform_with_vae

class AnomalyDetectionPipeline:
    """
    End-to-end anomaly detection pipeline that loads models from MLflow
    and processes data through SimCLR and VAE stages
    """
    def __init__(self, 
                 simclr_model_path=None, 
                 simclr_config_path=None,
                 vae_model_path=None, 
                 vae_config_path=None):
        """
        Initialize the anomaly detection pipeline with model paths or load from MLflow
        
        Args:
            simclr_model_path: Path to SimCLR model or None to load from MLflow
            simclr_config_path: Path to SimCLR config or None to load from MLflow
            vae_model_path: Path to VAE model or None to load from MLflow
            vae_config_path: Path to VAE config or None to load from MLflow
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize paths
        self.simclr_model_path = simclr_model_path
        self.simclr_config_path = simclr_config_path
        self.vae_model_path = vae_model_path
        self.vae_config_path = vae_config_path
        
        # Will be loaded later
        self.simclr_model = None
        self.simclr_config = None
        self.vae_model = None
        self.vae_config = None
        
    def init_mlflow(self):
        """Initialize MLflow client with DAGsHub"""
        try:
            dagshub.init(repo_owner='ElkamelDyari', repo_name='AnomalyDetection', mlflow=True)
            print("MLflow tracking URI:", mlflow.get_tracking_uri())
            return True
        except Exception as e:
            print(f"Error initializing MLflow: {e}")
            return False
    
    def load_models_from_mlflow(self, simclr_run_id=None, vae_run_id=None):
        """
        Load models from MLflow using run IDs
        If run IDs are not provided, will automatically find and use the most recent runs
        """
        self.init_mlflow()
        client = mlflow.tracking.MlflowClient()
        
        # Function to find and download artifacts from a run
        def find_and_download_model(run_id, model_type, search_filter=None):
            artifact_dir = None
            model_path = None
            config_path = None
            
            # Create temp directory for artifacts
            temp_dir = os.path.join(os.getcwd(), "temp_artifacts", model_type.lower())
            os.makedirs(temp_dir, exist_ok=True)
            
            # If run_id is provided, use it directly
            if run_id:
                try:
                    print(f"Loading {model_type} artifacts from specified run {run_id}")
                    artifact_dir = client.download_artifacts(run_id, "", temp_dir)
                except Exception as e:
                    print(f"Error loading {model_type} artifacts from run {run_id}: {e}")
                    return None, None
            # Otherwise find the latest run automatically
            elif search_filter:
                try:
                    print(f"Searching for the latest {model_type} run...")
                    runs = client.search_runs(
                        experiment_ids=["0"],  # Default experiment
                        filter_string=search_filter,
                        order_by=["attribute.start_time DESC"],
                        max_results=1
                    )
                    if runs:
                        latest_run = runs[0]
                        print(f"Found latest {model_type} run: {latest_run.info.run_id}")
                        artifact_dir = client.download_artifacts(
                            latest_run.info.run_id, "", temp_dir
                        )
                    else:
                        print(f"No {model_type} runs found matching the filter criteria")
                        return None, None
                except Exception as e:
                    print(f"Error finding latest {model_type} run: {e}")
                    return None, None
            
            # Find model and config files in the artifact directory
            if artifact_dir:
                try:
                    for root, _, files in os.walk(artifact_dir):
                        for file in files:
                            if file.endswith('.pth'):
                                model_path = os.path.join(root, file)
                            elif file.endswith('.json'):
                                config_path = os.path.join(root, file)
                    
                    if model_path:
                        print(f"Found {model_type} model: {model_path}")
                    if config_path:
                        print(f"Found {model_type} config: {config_path}")
                        
                    return model_path, config_path
                except Exception as e:
                    print(f"Error processing {model_type} artifacts: {e}")
                    return None, None
        
        # Get SimCLR model (either from specified run_id or latest run)
        simclr_model, simclr_config = find_and_download_model(
            simclr_run_id, 
            "SimCLR", 
            "tags.mlflow.runName LIKE '%SimCLR%'"
        )
        
        if simclr_model:
            self.simclr_model_path = simclr_model
        if simclr_config:
            self.simclr_config_path = simclr_config
        
        # Get VAE model (either from specified run_id or latest run)
        vae_model, vae_config = find_and_download_model(
            vae_run_id, 
            "VAE", 
            "tags.mlflow.runName LIKE '%VAE%'"
        )
        
        if vae_model:
            self.vae_model_path = vae_model
        if vae_config:
            self.vae_config_path = vae_config
        
        # Verify we have all necessary files
        if not self.simclr_model_path or not self.vae_model_path:
            print("Could not find all necessary model files")
            return False
        
        return True
    
    def load_local_models(self):
        """
        Load models from local paths if MLflow loading fails
        """
        # Default paths if not provided
        if not self.simclr_model_path:
            self.simclr_model_path = "data/SimCLR/simclr_head.pth"
        if not self.simclr_config_path:
            self.simclr_config_path = "data/SimCLR/best_arch.json"
        if not self.vae_model_path:
            self.vae_model_path = "data/VAE/final_vae_pcdarts.pth"
        if not self.vae_config_path:
            self.vae_config_path = "data/VAE/best_vae_architecture.json"
        
        # Check if files exist
        if not os.path.exists(self.simclr_model_path):
            print(f"SimCLR model not found at {self.simclr_model_path}")
            return False
        if not os.path.exists(self.vae_model_path):
            print(f"VAE model not found at {self.vae_model_path}")
            return False
        
        return True
    
    def load_configs(self):
        """Load model configurations from JSON files"""
        try:
            if os.path.exists(self.simclr_config_path):
                with open(self.simclr_config_path, 'r') as f:
                    self.simclr_config = json.load(f)
                print("Loaded SimCLR config")
            
            if os.path.exists(self.vae_config_path):
                with open(self.vae_config_path, 'r') as f:
                    self.vae_config = json.load(f)
                print("Loaded VAE config")
            
            return True
        except Exception as e:
            print(f"Error loading model configs: {e}")
            return False
    
    def process_simclr(self, input_data):
        """
        Process data through SimCLR model
        
        Args:
            input_data: Numpy array with extracted features
            
        Returns:
            Transformed data after SimCLR processing
        """
        # Check if we need to transform data
        if self.simclr_model_path is None:
            print("SimCLR model not loaded, skipping transformation")
            return input_data
        
        try:
            print(f"Processing data through SimCLR: {input_data.shape}")
            # Save input data to temporary file for SimCLR processing
            temp_input_path = os.path.join(os.getcwd(), "temp_data", "simclr_input.npy")
            temp_output_path = os.path.join(os.getcwd(), "temp_data", "simclr_output.npy")
            os.makedirs(os.path.dirname(temp_input_path), exist_ok=True)
            np.save(temp_input_path, input_data)
            
            # SimCLR transformation using the utility function with file paths
            simclr_transform(
                model_path=self.simclr_model_path,
                input_npy_path=temp_input_path,  # Use file path as expected by the function
                output_npy_path=temp_output_path,
                arch_json_path=self.simclr_config_path
            )
            
            # Load the transformed data
            if os.path.exists(temp_output_path):
                transformed_data = np.load(temp_output_path)
                print(f"SimCLR transformation complete: {transformed_data.shape}")
                return transformed_data
            else:
                print("SimCLR transformation failed to produce output file")
                return input_data
        except Exception as e:
            print(f"Error in SimCLR processing: {e}")
            print("Returning input data without transformation")
            return input_data
    
    def process_vae(self, input_data):
        """
        Process data through VAE model
        
        Args:
            input_data: Numpy array with SimCLR-transformed features
            
        Returns:
            Transformed data after VAE processing
        """
        if self.vae_model_path is None:
            print("VAE model not loaded, skipping transformation")
            return input_data
            
        try:
            print(f"Processing data through VAE: {input_data.shape}")
            
            # Save input data to temporary file for VAE processing
            temp_input_path = os.path.join(os.getcwd(), "temp_data", "vae_input.npy")
            os.makedirs(os.path.dirname(temp_input_path), exist_ok=True)
            
            # If the input has labels in the last column, separate them
            if input_data.shape[1] > 1:
                # Check if last column might be labels (containing only integers)
                last_col = input_data[:, -1]
                if np.all(np.equal(np.mod(last_col, 1), 0)):
                    # Looks like labels in the last column
                    features = input_data[:, :-1]
                    labels = last_col
                    np.save(temp_input_path, features)
                else:
                    # No obvious labels, save the whole array
                    np.save(temp_input_path, input_data)
                    labels = None
            else:
                np.save(temp_input_path, input_data)
                labels = None
                
            # Use the VAE utility function to transform the data with file path
            latent_space, transformed_labels = load_and_transform_with_vae(
                json_path=self.vae_config_path,
                model_path=self.vae_model_path,
                data_path=temp_input_path  # Use file path as expected by the function
            )
            
            # Use original labels if they exist, otherwise use transformed labels
            final_labels = labels if labels is not None else transformed_labels
            
            # Combine latent space with labels if available
            if final_labels is not None:
                output_data = np.column_stack((latent_space, final_labels))
            else:
                output_data = latent_space
                
            print(f"VAE transformation complete: {output_data.shape}")
            return output_data
        except Exception as e:
            print(f"Error in VAE processing: {e}")
            print("Returning input data without transformation")
            return input_data
    
    def detect_anomalies(self, transformed_data, threshold=None):
        """
        Detect anomalies in the transformed data
        
        Args:
            transformed_data: Data after SimCLR+VAE transformation
            threshold: Anomaly threshold (optional)
            
        Returns:
            Anomaly scores and binary predictions
        """
        try:
            # Simple anomaly detection based on reconstruction error
            # In a real scenario, you might want to compute this more carefully
            if transformed_data.shape[1] > 1:
                # If we have labels in the last column
                features = transformed_data[:, :-1]
                labels = transformed_data[:, -1]
            else:
                features = transformed_data
                labels = None
            
            # Compute anomaly score (e.g., distance from origin in latent space)
            anomaly_scores = np.linalg.norm(features, axis=1)
            
            # Determine threshold if not provided
            if threshold is None:
                # Use a percentile as threshold (e.g., top 5% are anomalies)
                threshold = np.percentile(anomaly_scores, 95)
            
            # Get binary predictions
            predictions = (anomaly_scores > threshold).astype(int)
            
            return anomaly_scores, predictions, labels, threshold
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            return None, None, None, None
    
    def evaluate(self, predictions, true_labels):
        """
        Evaluate anomaly detection performance
        
        Args:
            predictions: Binary anomaly predictions
            true_labels: Ground truth labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if true_labels is None or predictions is None:
            return None
            
        try:
            # Calculate evaluation metrics
            accuracy = accuracy_score(true_labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, predictions, average='binary')
            
            # ROC AUC if classes are binary
            unique_labels = np.unique(true_labels)
            if len(unique_labels) == 2:
                auc = roc_auc_score(true_labels, predictions)
            else:
                auc = None
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc
            }
            
            return metrics
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return None
    
    def visualize_results(self, data, anomaly_scores, predictions, labels=None, output_dir="results"):
        """
        Visualize anomaly detection results
        
        Args:
            data: Original or transformed data
            anomaly_scores: Computed anomaly scores
            predictions: Binary anomaly predictions
            labels: Ground truth labels (optional)
            output_dir: Directory to save visualizations
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Plot anomaly score distribution
            plt.figure(figsize=(10, 6))
            plt.hist(anomaly_scores, bins=50, alpha=0.7)
            plt.title('Distribution of Anomaly Scores')
            plt.xlabel('Anomaly Score')
            plt.ylabel('Frequency')
            plt.axvline(x=np.percentile(anomaly_scores, 95), color='r', 
                      linestyle='--', label='95th Percentile')
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'anomaly_score_distribution.png'))
            plt.close()
            
            # Scatter plot of first two dimensions with anomaly highlighting
            if data.shape[1] >= 2:
                plt.figure(figsize=(10, 8))
                
                # Choose dimensions to plot
                dim1 = 0
                dim2 = 1
                
                # Handle labels in data
                feature_data = data
                if labels is not None and data.shape[1] > 2:
                    feature_data = data[:, :-1]
                
                # Scatter plot colored by predictions
                scatter = plt.scatter(feature_data[:, dim1], feature_data[:, dim2], 
                          c=predictions, cmap='coolwarm', alpha=0.7)
                plt.colorbar(scatter, label='Anomaly (1) / Normal (0)')
                plt.title('Anomaly Detection Results')
                plt.xlabel(f'Dimension {dim1}')
                plt.ylabel(f'Dimension {dim2}')
                plt.savefig(os.path.join(output_dir, 'anomaly_detection_plot.png'))
                plt.close()
                
                # If ground truth is available, plot confusion matrix
                if labels is not None:
                    try:
                        from sklearn.metrics import confusion_matrix
                        # Try to import seaborn, but make it optional
                        try:
                            import seaborn as sns
                            use_seaborn = True
                        except ImportError:
                            use_seaborn = False
                        
                        cm = confusion_matrix(labels, predictions)
                        plt.figure(figsize=(8, 6))
                        
                        if use_seaborn:
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                        else:
                            # Fall back to matplotlib if seaborn is not available
                            plt.imshow(cm, interpolation='nearest', cmap='Blues')
                            plt.colorbar()
                            # Add text annotations
                            thresh = cm.max() / 2.
                            for i in range(cm.shape[0]):
                                for j in range(cm.shape[1]):
                                    plt.text(j, i, format(cm[i, j], 'd'),
                                            ha="center", va="center",
                                            color="white" if cm[i, j] > thresh else "black")
                                            
                        plt.title('Confusion Matrix')
                        plt.xlabel('Predicted Label')
                        plt.ylabel('True Label')
                        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
                        plt.close()
                    except Exception as cm_error:
                        print(f"Error creating confusion matrix: {cm_error}")
            
            print(f"Visualizations saved to {output_dir}")
        except Exception as e:
            print(f"Error in visualization: {e}")
    
    def run_pipeline(self, input_data_path, output_dir="results"):
        """
        Run the full anomaly detection pipeline
        
        Args:
            input_data_path: Path to input data
            output_dir: Directory to save results
            
        Returns:
            Dictionary with results
        """
        print("\n" + "="*50)
        print("Starting Anomaly Detection Pipeline")
        print("="*50)
        
        # Step 1: Try to load models from MLflow
        print("\nStep 1: Loading models from MLflow (using latest runs)...")
        mlflow_success = self.load_models_from_mlflow()
        
        # Step 2: If MLflow fails, try local models
        if not mlflow_success:
            print("\nStep 2: Loading local models...")
            local_success = self.load_local_models()
            if not local_success:
                print("Failed to load models. Exiting pipeline.")
                return None
        
        # Step 3: Load model configurations
        print("\nStep 3: Loading model configurations...")
        self.load_configs()
        
        # Step 4: Load input data
        print("\nStep 4: Loading input data...")
        try:
            if isinstance(input_data_path, str):
                input_data = np.load(input_data_path)
                print(f"Loaded data: {input_data.shape}")
            else:
                # Assume input_data_path is already a numpy array
                input_data = input_data_path
                print(f"Using provided numpy array: {input_data.shape}")
        except Exception as e:
            print(f"Error loading input data: {e}")
            return None
        
        # Step 5: Process through SimCLR
        print("\nStep 5: Processing through SimCLR...")
        simclr_transformed = self.process_simclr(input_data)
        
        # Step 6: Process through VAE
        print("\nStep 6: Processing through VAE...")
        vae_transformed = self.process_vae(simclr_transformed)
        
        # Step 7: Detect anomalies
        print("\nStep 7: Detecting anomalies...")
        anomaly_scores, predictions, labels, threshold = self.detect_anomalies(vae_transformed)
        
        if anomaly_scores is None:
            print("Anomaly detection failed. Exiting pipeline.")
            return None
        
        print(f"Detected {sum(predictions)} anomalies out of {len(predictions)} samples")
        print(f"Anomaly threshold: {threshold:.4f}")
        
        # Step 8: Evaluate results
        print("\nStep 8: Evaluating results...")
        metrics = self.evaluate(predictions, labels)
        
        if metrics:
            print("Evaluation metrics:")
            for metric, value in metrics.items():
                if value is not None:
                    print(f"  {metric}: {value:.4f}")
        
        # Step 9: Visualize results
        print("\nStep 9: Visualizing results...")
        self.visualize_results(vae_transformed, anomaly_scores, predictions, labels, output_dir)
        
        # Step 10: Save results
        print("\nStep 10: Saving results...")
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save predictions
            np.save(os.path.join(output_dir, "anomaly_predictions.npy"), predictions)
            np.save(os.path.join(output_dir, "anomaly_scores.npy"), anomaly_scores)
            
            # Save transformed data
            np.save(os.path.join(output_dir, "vae_transformed.npy"), vae_transformed)
            
            # Save metrics
            if metrics:
                with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
                    json.dump({k: float(v) if v is not None else None for k, v in metrics.items()}, f, indent=2)
            
            print(f"Results saved to {output_dir}")
        except Exception as e:
            print(f"Error saving results: {e}")
        
        print("\n" + "="*50)
        print("Anomaly Detection Pipeline Complete")
        print("="*50)
        
        # Return results
        results = {
            'transformed_data': vae_transformed,
            'anomaly_scores': anomaly_scores,
            'predictions': predictions,
            'threshold': threshold,
            'metrics': metrics
        }
        
        return results


def main():
    """Main function to run the pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Anomaly Detection Pipeline')
    parser.add_argument('--input', type=str, default='data/extracted_features/train_cls_embeddings_with_label.npy',
                       help='Path to input data')
    parser.add_argument('--output', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--simclr-run-id', type=str, default=None,
                       help='MLflow run ID for SimCLR model (if omitted, the latest run will be used)')
    parser.add_argument('--vae-run-id', type=str, default=None,
                       help='MLflow run ID for VAE model (if omitted, the latest run will be used)')
    parser.add_argument('--simclr-model', type=str, default=None,
                       help='Path to local SimCLR model (if provided, MLflow will not be used)')
    parser.add_argument('--vae-model', type=str, default=None,
                       help='Path to local VAE model (if provided, MLflow will not be used)')
    parser.add_argument('--use-local', action='store_true',
                       help='Use local models instead of MLflow (looks in default data directories)')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Custom anomaly detection threshold (default: 95th percentile)')
    parser.add_argument('--install-dependencies', action='store_true',
                       help='Install missing dependencies before running')
    
    args = parser.parse_args()
    
    # Check for missing dependencies and install them if requested
    if args.install_dependencies:
        try:
            import importlib
            import subprocess
            
            # Check for seaborn
            try:
                importlib.import_module('seaborn')
                print("Seaborn is already installed.")
            except ImportError:
                print("Installing seaborn...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
                print("Seaborn installed successfully!")
            
            # Add other dependency checks as needed
            
        except Exception as e:
            print(f"Error installing dependencies: {e}")
    
    # Initialize pipeline
    pipeline = AnomalyDetectionPipeline(
        simclr_model_path=args.simclr_model if args.use_local or args.simclr_model else None,
        vae_model_path=args.vae_model if args.use_local or args.vae_model else None
    )
    
    # Run pipeline
    results = pipeline.run_pipeline(args.input, args.output)
    
    if results:
        print("\nPipeline executed successfully!")
        print(f"Check {args.output} directory for results and visualizations.")
    else:
        print("\nPipeline execution failed.")


if __name__ == "__main__":
    main()
