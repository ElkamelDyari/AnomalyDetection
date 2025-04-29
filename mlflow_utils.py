"""
MLflow utilities for the AnomalyDetection project
Provides common functions for tracking models with MLflow and DAGsHub
"""
import os
import json
import mlflow
import dagshub
from pathlib import Path
import numpy as np

def init_tracking():
    """Initialize DAGsHub and MLflow tracking"""
    dagshub.init(repo_owner='ElkamelDyari', repo_name='AnomalyDetection', mlflow=True)
    print("MLflow tracking URI:", mlflow.get_tracking_uri())
    return mlflow

def log_model_params_from_json(json_path):
    """Log model parameters from a JSON file"""
    if not os.path.exists(json_path):
        print(f"Warning: JSON file {json_path} not found.")
        return
        
    try:
        with open(json_path, 'r') as f:
            params = json.load(f)
        
        # Log each parameter
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        # Log the JSON file as an artifact
        mlflow.log_artifact(json_path)
        print(f"Parameters logged from {json_path}")
    except Exception as e:
        print(f"Error logging parameters from {json_path}: {e}")

def log_artifact_if_exists(artifact_path):
    """Log an artifact if it exists"""
    if not os.path.exists(artifact_path):
        print(f"Warning: Artifact {artifact_path} not found.")
        return False
        
    try:
        mlflow.log_artifact(artifact_path)
        print(f"Artifact logged: {artifact_path}")
        return True
    except Exception as e:
        print(f"Error logging artifact {artifact_path}: {e}")
        return False

def log_directory_artifacts(directory_path):
    """Log all files in a directory as artifacts"""
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        print(f"Warning: Directory {directory_path} not found.")
        return 0
        
    count = 0
    try:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                mlflow.log_artifact(file_path)
                count += 1
        print(f"Logged {count} artifacts from {directory_path}")
        return count
    except Exception as e:
        print(f"Error logging artifacts from directory {directory_path}: {e}")
        return 0

def log_numpy_array_summary(array_path, array_name="data"):
    """Log summary statistics for a numpy array"""
    if not os.path.exists(array_path):
        print(f"Warning: Numpy array file {array_path} not found.")
        return
        
    try:
        data = np.load(array_path)
        mlflow.log_metric(f"{array_name}_samples", data.shape[0])
        if len(data.shape) > 1:
            mlflow.log_metric(f"{array_name}_features", data.shape[1])
        
        # Log basic statistics if the array doesn't contain labels
        if len(data.shape) == 2 and data.shape[1] > 1:
            try:
                # Assuming last column might be labels, skip it for stats
                features = data[:, :-1]
                mlflow.log_metric(f"{array_name}_mean", float(np.mean(features)))
                mlflow.log_metric(f"{array_name}_std", float(np.std(features)))
                mlflow.log_metric(f"{array_name}_min", float(np.min(features)))
                mlflow.log_metric(f"{array_name}_max", float(np.max(features)))
            except Exception as stats_error:
                print(f"Could not compute statistics: {stats_error}")
        
        print(f"Logged numpy array summary for {array_path}")
    except Exception as e:
        print(f"Error logging numpy array summary for {array_path}: {e}")
