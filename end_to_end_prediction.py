"""
End-to-End Anomaly Detection Pipeline

This script takes raw log files and processes them through the entire pipeline:
1. Log parsing with Drain
2. Preprocessing
3. Feature extraction with CodeBERT
4. SimCLR transformation (loaded from MLflow)
5. VAE transformation (loaded from MLflow)
6. Anomaly detection and prediction
7. Results saved to CSV with original logs and predictions
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import mlflow
import dagshub
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Import model components
from models.drain import (
    load_log_file, 
    enrich_log_structure, 
    init_drain, 
    stream_templates,
    process_log_column
)
from models.preprocessing import remove_low_variance
from models.feature_extraction import generate_cls_embeddings
from models.SimCLR_utils import transform_data as simclr_transform
from models.VAE_utils import load_and_transform_with_vae

class EndToEndPipeline:
    """
    Complete end-to-end anomaly detection pipeline starting from raw logs
    """
    def __init__(self, 
                output_dir="output", 
                device=None,
                model_name="microsoft/codebert-base",
                max_len=128,
                batch_size=32,
                sample_ratio=1.0):
        """
        Initialize the pipeline
        
        Args:
            output_dir: Directory to save intermediate and final outputs
            device: Device to use for model inference (cuda or cpu)
            model_name: Transformer model to use for feature extraction
            max_len: Maximum sequence length for the transformer
            batch_size: Batch size for model inference
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for each stage
        self.drain_dir = os.path.join(output_dir, "drain")
        self.preprocess_dir = os.path.join(output_dir, "preprocessed")
        self.features_dir = os.path.join(output_dir, "features")
        self.simclr_dir = os.path.join(output_dir, "simclr")
        self.vae_dir = os.path.join(output_dir, "vae")
        self.results_dir = os.path.join(output_dir, "results")
        
        # Create all directories
        for dir_path in [self.drain_dir, self.preprocess_dir, self.features_dir, 
                        self.simclr_dir, self.vae_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        # Set device for feature extraction
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        # Feature extraction parameters
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.sample_ratio = sample_ratio
        logging.info(f"Using sample ratio: {sample_ratio * 100:.1f}%")
        
        # MLflow model paths (will be populated later)
        self.simclr_model_path = None
        self.simclr_config_path = None
        self.vae_model_path = None
        self.vae_config_path = None
        
        # Original logs and templates (for final output)
        self.original_logs = None
        self.log_templates = None
    
    def init_mlflow(self):
        """Initialize MLflow client with DAGsHub"""
        try:
            dagshub.init(repo_owner='ElkamelDyari', repo_name='AnomalyDetection', mlflow=True)
            logging.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
            return True
        except Exception as e:
            logging.error(f"Error initializing MLflow: {e}")
            return False
    
    def load_models_from_mlflow(self):
        """
        Load SimCLR and VAE models from MLflow
        Uses the latest run for each model type
        """
        self.init_mlflow()
        client = mlflow.tracking.MlflowClient()
        
        # Function to find and download model artifacts
        def find_and_download_model(model_type, search_filter):
            temp_dir = os.path.join(os.getcwd(), "temp_artifacts", model_type.lower())
            os.makedirs(temp_dir, exist_ok=True)
            
            try:
                logging.info(f"Searching for the latest {model_type} run...")
                runs = client.search_runs(
                    experiment_ids=["0"],  # Default experiment
                    filter_string=search_filter,
                    order_by=["attribute.start_time DESC"],
                    max_results=1
                )
                
                if not runs:
                    logging.warning(f"No {model_type} runs found")
                    return None, None
                
                latest_run = runs[0]
                logging.info(f"Found latest {model_type} run: {latest_run.info.run_id}")
                
                try:
                    artifact_dir = client.download_artifacts(
                        latest_run.info.run_id, "", temp_dir
                    )
                except Exception as e:
                    logging.error(f"Error downloading artifacts: {e}")
                    return None, None
                
                # Find model and config files
                model_path, config_path = None, None
                for root, _, files in os.walk(artifact_dir):
                    for file in files:
                        if file.endswith('.pth'):
                            model_path = os.path.join(root, file)
                        elif file.endswith('.json'):
                            config_path = os.path.join(root, file)
                
                if model_path:
                    logging.info(f"Found {model_type} model: {model_path}")
                if config_path:
                    logging.info(f"Found {model_type} config: {config_path}")
                
                return model_path, config_path
            
            except Exception as e:
                logging.error(f"Error finding {model_type} model: {e}")
                return None, None
        
        # Find SimCLR model
        self.simclr_model_path, self.simclr_config_path = find_and_download_model(
            "SimCLR", "tags.mlflow.runName LIKE '%SimCLR%'"
        )
        
        # Find VAE model
        self.vae_model_path, self.vae_config_path = find_and_download_model(
            "VAE", "tags.mlflow.runName LIKE '%VAE%'"
        )
        
        # Check if models were found
        if not self.simclr_model_path:
            logging.warning("SimCLR model not found in MLflow, will try to use local models")
        
        if not self.vae_model_path:
            logging.warning("VAE model not found in MLflow, will try to use local models")
        
        return self.simclr_model_path is not None and self.vae_model_path is not None
    
    def load_local_models(self):
        """Load models from local files if MLflow loading fails"""
        if not self.simclr_model_path:
            self.simclr_model_path = "data/SimCLR/simclr_head.pth"
            self.simclr_config_path = "data/SimCLR/best_arch.json"
            
            if os.path.exists(self.simclr_model_path):
                logging.info(f"Using local SimCLR model: {self.simclr_model_path}")
            else:
                logging.error(f"Local SimCLR model not found at {self.simclr_model_path}")
                self.simclr_model_path = None
                
        if not self.vae_model_path:
            self.vae_model_path = "data/VAE/final_vae_pcdarts.pth"
            self.vae_config_path = "data/VAE/best_vae_architecture.json"
            
            if os.path.exists(self.vae_model_path):
                logging.info(f"Using local VAE model: {self.vae_model_path}")
            else:
                logging.error(f"Local VAE model not found at {self.vae_model_path}")
                self.vae_model_path = None
    
    def step1_parse_logs_with_drain(self, log_file_path, max_lines=None):
        """
        Parse raw logs using Drain to extract templates
        
        Args:
            log_file_path: Path to the input log file
            
        Returns:
            Path to the CSV with extracted templates
        """
        logging.info("Step 1: Parsing logs with Drain")
        
        # Load log file
        logging.info(f"Loading log file: {log_file_path}")
        if max_lines:
            logging.info(f"Using limited sample: {max_lines} lines")
        df = load_log_file(log_file_path, max_lines=max_lines)
        self.original_logs = df.copy()  # Keep original logs for final output
        
        # Enrich log structure
        logging.info("Enriching log structure")
        df = enrich_log_structure(df)
        
        # Process logs with Drain
        templates_csv = os.path.join(self.drain_dir, "log_templates.csv")
        logging.info(f"Extracting templates with Drain to {templates_csv}")
        process_log_column(df, "Content", templates_csv)
        
        # Load templates
        self.log_templates = pd.read_csv(templates_csv)
        logging.info(f"Extracted {len(self.log_templates)} templates")
        
        return templates_csv
    
    def step2_preprocess_templates(self, templates_csv, variance_threshold=5, sample_ratio=None):
        """
        Preprocess templates by removing low variance ones
        
        Args:
            templates_csv: Path to CSV with extracted templates
            variance_threshold: Threshold for template occurrence filtering
            
        Returns:
            Path to preprocessed data
        """
        logging.info("Step 2: Preprocessing templates")
        
        # Filter out low variance templates
        logging.info(f"Removing templates with occurrence < {variance_threshold}")
        filtered_data = remove_low_variance(templates_csv, variance_threshold)
        
        # Apply sampling if needed
        sample_ratio = sample_ratio or self.sample_ratio
        if sample_ratio < 1.0:
            original_size = len(filtered_data)
            sample_size = max(int(original_size * sample_ratio), 1000)  # Ensure minimum sample size
            filtered_data = filtered_data.sample(n=sample_size, random_state=42)
            logging.info(f"Sampled data from {original_size} to {len(filtered_data)} entries ({sample_ratio*100:.1f}%)")
        
        # Save preprocessed data
        preprocessed_csv = os.path.join(self.preprocess_dir, "preprocessed_templates.csv")
        filtered_data.to_csv(preprocessed_csv, index=False)
        logging.info(f"Saved {len(filtered_data)} preprocessed templates to {preprocessed_csv}")
        
        return preprocessed_csv
    
    def step3_extract_features(self, preprocessed_csv, batch_size=None):
        """
        Extract features from templates using CodeBERT
        
        Args:
            preprocessed_csv: Path to preprocessed templates
            
        Returns:
            Path to extracted features
        """
        logging.info("Step 3: Extracting features with CodeBERT")
        
        # Load preprocessed data
        df = pd.read_csv(preprocessed_csv)
        templates = df["template"].astype(str).tolist()
        labels = df["label"].to_numpy().reshape(-1, 1)
        
        # Extract features - ensure larger batch size for GPU efficiency
        batch_size = batch_size or self.batch_size
        # Increase batch size for GPU efficiency if we have many templates
        if len(templates) > 10000 and torch.cuda.is_available():
            # Dynamically adjust batch size based on available VRAM
            available_mem = torch.cuda.get_device_properties(0).total_memory
            # Use up to 80% of available memory, estimating ~20MB per 100 samples at 128 seq length
            recommended_batch = min(256, max(32, int((available_mem * 0.8) / (2e7) * 100)))
            batch_size = max(batch_size, recommended_batch)
            logging.info(f"Adjusted batch size to {batch_size} for GPU efficiency")
            
        logging.info(f"Generating embeddings for {len(templates)} templates using batch size {batch_size}")
        # Free up CUDA memory before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        embeddings = generate_cls_embeddings(templates, batch_size=batch_size)
        
        # Combine embeddings with labels
        features_with_labels = np.hstack([embeddings, labels])
        
        # Save features
        features_path = os.path.join(self.features_dir, "extracted_features.npy")
        np.save(features_path, features_with_labels)
        logging.info(f"Saved features with shape {features_with_labels.shape} to {features_path}")
        
        return features_path
    
    def step4_transform_with_simclr(self, features_path):
        """
        Transform features using SimCLR model from MLflow
        
        Args:
            features_path: Path to extracted features
            
        Returns:
            Path to SimCLR-transformed features
        """
        logging.info("Step 4: Transforming with SimCLR")
        
        if not self.simclr_model_path:
            logging.error("SimCLR model not available")
            return features_path
        
        try:
            simclr_output_path = os.path.join(self.simclr_dir, "simclr_transformed.npy")
            
            # Transform with SimCLR
            logging.info(f"Applying SimCLR transformation using model: {self.simclr_model_path}")
            simclr_transform(
                model_path=self.simclr_model_path,
                input_npy_path=features_path,
                output_npy_path=simclr_output_path,
                arch_json_path=self.simclr_config_path
            )
            
            logging.info(f"Saved SimCLR transformed features to {simclr_output_path}")
            return simclr_output_path
        
        except Exception as e:
            logging.error(f"Error in SimCLR transformation: {e}")
            return features_path
    
    def step5_transform_with_vae(self, simclr_features_path):
        """
        Transform SimCLR features using VAE model from MLflow
        
        Args:
            simclr_features_path: Path to SimCLR-transformed features
            
        Returns:
            Path to VAE-transformed features
        """
        logging.info("Step 5: Transforming with VAE")
        
        if not self.vae_model_path:
            logging.error("VAE model not available")
            return simclr_features_path
        
        try:
            vae_output_path = os.path.join(self.vae_dir, "vae_transformed.npy")
            
            # Try to transform with VAE
            logging.info(f"Applying VAE transformation using model: {self.vae_model_path}")
            # This function returns both latent representations and labels
            latent_space, labels = load_and_transform_with_vae(
                json_path=self.vae_config_path,
                model_path=self.vae_model_path,
                data_path=simclr_features_path
            )
            
            # If we got results, combine and save
            if latent_space is not None:
                # Combine with labels if available
                if labels is not None:
                    transformed_data = np.column_stack((latent_space, labels))
                else:
                    # If no labels in VAE output, try to get from input
                    input_data = np.load(simclr_features_path)
                    if input_data.shape[1] > 1:
                        # Assume last column is label
                        labels = input_data[:, -1]
                        transformed_data = np.column_stack((latent_space, labels))
                    else:
                        transformed_data = latent_space
                
                np.save(vae_output_path, transformed_data)
                logging.info(f"Saved VAE transformed features with shape {transformed_data.shape} to {vae_output_path}")
                return vae_output_path
            else:
                logging.warning("VAE transformation failed to produce output")
                return simclr_features_path
                
        except Exception as e:
            logging.error(f"Error in VAE transformation: {e}")
            logging.warning("Using SimCLR features for anomaly detection")
            return simclr_features_path
    
    def step6_detect_anomalies(self, features_path, threshold_percentile=95):
        """
        Detect anomalies in the transformed features
        
        Args:
            features_path: Path to transformed features
            threshold_percentile: Percentile for anomaly threshold
            
        Returns:
            DataFrame with anomaly predictions and scores
        """
        logging.info("Step 6: Detecting anomalies")
        
        # Load transformed features
        data = np.load(features_path)
        logging.info(f"Loaded features with shape {data.shape}")
        
        # Extract features and labels
        if data.shape[1] > 1:
            # Assume last column is label
            features = data[:, :-1]
            true_labels = data[:, -1]
        else:
            features = data
            true_labels = None
        
        # Compute anomaly scores (distance from origin in latent space)
        anomaly_scores = np.linalg.norm(features, axis=1)
        
        # Determine threshold
        threshold = np.percentile(anomaly_scores, threshold_percentile)
        logging.info(f"Using threshold {threshold:.4f} ({threshold_percentile}th percentile)")
        
        # Get predictions
        predictions = (anomaly_scores > threshold).astype(int)
        
        # Create results DataFrame
        results = pd.DataFrame({
            "anomaly_score": anomaly_scores,
            "anomaly_prediction": predictions
        })
        
        # Add true labels if available
        if true_labels is not None:
            results["true_label"] = true_labels
            
            # Compute metrics
            accuracy = accuracy_score(true_labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, predictions, average='binary')
            
            logging.info(f"Anomaly detection metrics:")
            logging.info(f"  Accuracy: {accuracy:.4f}")
            logging.info(f"  Precision: {precision:.4f}")
            logging.info(f"  Recall: {recall:.4f}")
            logging.info(f"  F1 Score: {f1:.4f}")
        
        # Count anomalies
        anomaly_count = predictions.sum()
        logging.info(f"Detected {anomaly_count} anomalies out of {len(predictions)} samples "
                     f"({anomaly_count/len(predictions)*100:.2f}%)")
        
        return results
    
    def step7_create_final_output(self, anomaly_results):
        """
        Create final output by combining original logs with predictions
        
        Args:
            anomaly_results: DataFrame with anomaly predictions and scores
            
        Returns:
            Path to final CSV output
        """
        logging.info("Step 7: Creating final output")
        
        # Start with original logs
        if self.original_logs is None or self.log_templates is None:
            logging.error("Original logs or templates not available")
            # Save what we have
            results_path = os.path.join(self.results_dir, "anomaly_predictions.csv")
            anomaly_results.to_csv(results_path, index=False)
            return results_path
        
        # Combine original logs with templates and predictions
        combined_df = self.original_logs.copy()
        
        # Add template information
        template_df = self.log_templates.copy()
        # Make sure we have the same number of rows
        if len(combined_df) == len(template_df):
            combined_df["template"] = template_df["template"]
            combined_df["template_params"] = template_df["params"]
        else:
            logging.warning(f"Row count mismatch: {len(combined_df)} logs vs {len(template_df)} templates")
        
        # Add anomaly predictions
        if len(combined_df) == len(anomaly_results):
            combined_df["anomaly_score"] = anomaly_results["anomaly_score"].values
            combined_df["anomaly_prediction"] = anomaly_results["anomaly_prediction"].values
            
            # If we have true labels in both, check agreement
            if "true_label" in anomaly_results.columns and "label" in combined_df.columns:
                agreement = (anomaly_results["true_label"] == combined_df["label"]).mean()
                logging.info(f"Label agreement: {agreement:.4f}")
        else:
            logging.warning(f"Row count mismatch: {len(combined_df)} logs vs {len(anomaly_results)} predictions")
            # If sizes don't match, create a separate predictions file
            anomaly_results.to_csv(os.path.join(self.results_dir, "anomaly_predictions.csv"), index=False)
        
        # Save final output
        results_path = os.path.join(self.results_dir, "final_predictions.csv")
        combined_df.to_csv(results_path, index=False)
        logging.info(f"Saved final predictions to {results_path}")
        
        # Generate summary visualization
        self.create_summary_visualization(combined_df, anomaly_results)
        
        return results_path
    
    def create_summary_visualization(self, combined_df, anomaly_results):
        """Create summary visualizations for the anomaly detection results"""
        try:
            # Create visualizations directory
            vis_dir = os.path.join(self.results_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            # 1. Anomaly score distribution
            plt.figure(figsize=(10, 6))
            plt.hist(anomaly_results["anomaly_score"], bins=50, alpha=0.7)
            plt.title('Distribution of Anomaly Scores')
            plt.xlabel('Anomaly Score')
            plt.ylabel('Frequency')
            threshold = np.percentile(anomaly_results["anomaly_score"], 95)
            plt.axvline(x=threshold, color='r', linestyle='--', label='95th Percentile Threshold')
            plt.legend()
            plt.savefig(os.path.join(vis_dir, 'anomaly_score_distribution.png'))
            plt.close()
            
            # 2. Confusion matrix if we have true labels
            if "true_label" in anomaly_results.columns and "anomaly_prediction" in anomaly_results.columns:
                from sklearn.metrics import confusion_matrix
                try:
                    import seaborn as sns
                    has_seaborn = True
                except ImportError:
                    has_seaborn = False
                
                cm = confusion_matrix(anomaly_results["true_label"], anomaly_results["anomaly_prediction"])
                plt.figure(figsize=(8, 6))
                
                if has_seaborn:
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                else:
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
                plt.savefig(os.path.join(vis_dir, 'confusion_matrix.png'))
                plt.close()
                
            # 3. Template frequency for anomalies vs normal
            if "template" in combined_df.columns and "anomaly_prediction" in combined_df.columns:
                # Get most common templates for anomalies
                anomaly_templates = combined_df[combined_df["anomaly_prediction"] == 1]["template"].value_counts().head(10)
                normal_templates = combined_df[combined_df["anomaly_prediction"] == 0]["template"].value_counts().head(10)
                
                # Plot anomaly templates
                plt.figure(figsize=(12, 6))
                anomaly_templates.plot(kind='barh')
                plt.title('Most Common Templates in Anomalies')
                plt.xlabel('Count')
                plt.ylabel('Template')
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, 'anomaly_templates.png'))
                plt.close()
                
                # Plot normal templates
                plt.figure(figsize=(12, 6))
                normal_templates.plot(kind='barh')
                plt.title('Most Common Templates in Normal Logs')
                plt.xlabel('Count')
                plt.ylabel('Template')
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, 'normal_templates.png'))
                plt.close()
            
            logging.info(f"Saved visualizations to {vis_dir}")
            
        except Exception as e:
            logging.error(f"Error creating visualizations: {e}")
    
    def run_pipeline(self, log_file_path, variance_threshold=5, threshold_percentile=95, sample_ratio=None):
        """
        Run the complete pipeline from raw logs to anomaly predictions
        
        Args:
            log_file_path: Path to input log file
            variance_threshold: Threshold for template filtering
            threshold_percentile: Percentile for anomaly detection threshold
            
        Returns:
            Path to final predictions CSV
        """
        # 0. Load models from MLflow
        logging.info("Step 0: Loading models from MLflow")
        mlflow_success = self.load_models_from_mlflow()
        
        if not mlflow_success:
            logging.info("Falling back to local models")
            self.load_local_models()
        
        # Apply sampling ratio if provided
        actual_sample_ratio = sample_ratio if sample_ratio is not None else self.sample_ratio
        
        # Calculate max lines based on sample ratio if sampling is requested
        max_lines = None
        if actual_sample_ratio < 1.0:
            # Estimate total lines in file to determine sample size
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Count lines in first 1000 bytes to estimate average line length
                sample = f.read(10000)
                avg_line_length = len(sample) / (sample.count('\n') + 1)
                # Get file size and estimate total lines
                f.seek(0, 2)  # Seek to end
                file_size = f.tell()
                estimated_lines = int(file_size / avg_line_length)
                # Calculate max lines based on sample ratio
                max_lines = max(int(estimated_lines * actual_sample_ratio), 1000)
                logging.info(f"Estimated {estimated_lines} total lines, sampling {max_lines} lines ({actual_sample_ratio*100:.1f}%)")
        
        # Step 1: Parse logs with Drain
        templates_csv = self.step1_parse_logs_with_drain(log_file_path, max_lines=max_lines)
        preprocessed_csv = self.step2_preprocess_templates(templates_csv, variance_threshold)
        features_path = self.step3_extract_features(preprocessed_csv)
        simclr_features_path = self.step4_transform_with_simclr(features_path)
        final_features_path = self.step5_transform_with_vae(simclr_features_path)
        anomaly_results = self.step6_detect_anomalies(final_features_path, threshold_percentile)
        final_output_path = self.step7_create_final_output(anomaly_results)
        
        logging.info(f"Pipeline completed successfully! Final output: {final_output_path}")
        return final_output_path


def main():
    """Main function to run the pipeline from the command line"""
    parser = argparse.ArgumentParser(description='End-to-End Anomaly Detection Pipeline')
    parser.add_argument('--log-file', type=str, required=True,
                        help='Path to input log file')
    parser.add_argument('--output-dir', type=str, default='pipeline_output',
                        help='Directory to save all outputs')
    parser.add_argument('--variance-threshold', type=int, default=5,
                        help='Threshold for template occurrence filtering')
    parser.add_argument('--threshold-percentile', type=float, default=95,
                        help='Percentile threshold for anomaly detection')
    parser.add_argument('--model-name', type=str, default='microsoft/codebert-base',
                        help='Transformer model for feature extraction')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--max-len', type=int, default=128,
                        help='Maximum sequence length for feature extraction')
    parser.add_argument('--sample-ratio', type=float, default=0.1,
                        help='Ratio of data to sample (0.1 = 10%)')
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = EndToEndPipeline(
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_len=args.max_len,
        batch_size=args.batch_size,
        sample_ratio=args.sample_ratio
    )
    
    # Run the pipeline
    pipeline.run_pipeline(
        log_file_path=args.log_file,
        variance_threshold=args.variance_threshold,
        threshold_percentile=args.threshold_percentile,
        sample_ratio=args.sample_ratio
    )
    
    print(f"\nPipeline execution complete!")
    print(f"Check {args.output_dir}/results directory for final predictions and visualizations.")


if __name__ == "__main__":
    main()
