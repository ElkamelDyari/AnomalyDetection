# Log Anomaly Detection with Drain, SimCLR, and VAE

This project implements an end-to-end log anomaly detection system that processes raw log data through a multi-stage pipeline:

1. **Log Parsing with Drain3**: Extract structured templates from raw log messages
2. **Preprocessing**: Filter and clean log templates
3. **Feature Extraction**: Generate embeddings using CodeBERT
4. **Representation Learning**: Transform embeddings with SimCLR
5. **Dimensionality Reduction**: Transform representations with VAE
6. **Anomaly Detection**: Detect anomalies based on distance metrics

The project integrates with MLflow for experiment tracking and model management through DAGsHub: [https://dagshub.com/ElkamelDyari/AnomalyDetection.mlflow](https://dagshub.com/ElkamelDyari/AnomalyDetection.mlflow)

## Key Features

- **End-to-End Pipeline**: Process raw logs to anomaly predictions in a single workflow
- **Modular Architecture**: Each component can be used, trained, and evaluated separately
- **Neural Architecture Search**: PCDarts for optimizing SimCLR and VAE architectures
- **MLflow Integration**: Track experiments, parameters, metrics, and artifacts
- **Automated Model Loading**: Use the latest models from MLflow or fall back to local files
- **GPU Acceleration**: Utilize CUDA for faster processing
- **Sample-Based Processing**: Process large log files efficiently by sampling

## Project Structure

```
.
├── data/                    # Data directory
│   ├── raw/                # Raw log files
│   ├── preprocessed/       # Preprocessed data
│   ├── SimCLR/             # SimCLR output
│   └── extracted_features/ # Feature extraction output
├── models/                 # Model implementations
│   ├── SimCLR.py           # SimCLR implementation
│   ├── SimCLR_utils.py     # SimCLR utility functions
│   ├── VAE.py              # VAE implementation
│   ├── VAE_utils.py        # VAE utility functions
│   ├── drain.py            # Drain log parser implementation
│   ├── preprocessing.py    # Data preprocessing utilities
│   └── feature_extraction.py # Feature extraction with CodeBERT
├── mlflow_utils.py         # MLflow utility functions
├── track_models.py         # Main tracking script for all models
├── track_simclr.py         # SimCLR model tracking
├── track_vae.py            # VAE model tracking
├── track_drain.py          # Drain model tracking
├── track_simclr_pcdarts.py # SimCLR architecture search with tracking
├── track_vae_pcdarts.py    # VAE architecture search with tracking
├── train_best_simclr.py    # Train SimCLR with best architecture
├── train_best_vae.py       # Train VAE with best architecture
├── prediction_pipeline.py  # Pipeline for predictions using trained models
└── end_to_end_prediction.py # End-to-end pipeline from raw logs to anomaly predictions
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ElkamelDyari/AnomalyDetection.git
   cd AnomalyDetection
   ```

2. Create a virtual environment:
   ```bash
   python -m venv myEnv
   ```

3. Activate the virtual environment:
   ```bash
   # On Windows
   .\myEnv\Scripts\activate
   # On Linux/Mac
   source myEnv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install mlflow dagshub torch transformers scikit-learn matplotlib seaborn pandas tqdm optuna drain3 umap-learn
   ```

## Usage

### End-to-End Anomaly Detection

Process raw logs through the entire pipeline:

```bash
# Run end-to-end pipeline with sample rate to manage large files
python end_to_end_prediction.py --log-file "data/raw/logs.log" --sample-ratio 0.1
```

Options:
```
--log-file TEXT             Path to input log file [required]
--output-dir TEXT           Directory to save all outputs [default: pipeline_output]
--variance-threshold INT    Threshold for template occurrence filtering [default: 5]
--threshold-percentile FLOAT Percentile threshold for anomaly detection [default: 95]
--model-name TEXT           Transformer model for feature extraction [default: microsoft/codebert-base]
--batch-size INT            Batch size for feature extraction [default: 32]
--max-len INT               Maximum sequence length for feature extraction [default: 128]
--sample-ratio FLOAT        Ratio of data to sample (0.1 = 10%) [default: 0.1]
```

### Architecture Search with PCDarts

Optimize SimCLR architecture with PCDarts and track in MLflow:

```bash
python track_simclr_pcdarts.py --data-path "data/extracted_features/train_cls_embeddings_with_label.npy" \
                              --experiment-name "SimCLR_PCDarts_Search" \
                              --n-trials 20 --epochs-per-trial 5
```

Optimize VAE architecture with PCDarts and track in MLflow:

```bash
python track_vae_pcdarts.py --data-path "data/SimCLR/SimCLR_data.npy" \
                           --experiment-name "VAE_PCDarts_Search" \
                           --n-trials 20 --epochs-per-trial 5
```

### Training with Best Architectures

Train SimCLR with the best PCDarts architecture:

```bash
python train_best_simclr.py --data-path "data/extracted_features/train_cls_embeddings_with_label.npy" --epochs 20
```

Train VAE with the best PCDarts architecture:

```bash
python train_best_vae.py --data-path "data/SimCLR/SimCLR_data.npy" --epochs 20
```

### Tracking Models in MLflow

Log existing models to MLflow:

```bash
python track_models.py --track-all
```

Or track specific models:

```bash
python track_models.py --track-simclr --track-vae --track-drain
```

## Pipeline Workflow

### 1. Log Parsing (Drain)

Drain3 parses raw logs into structured templates, extracting constant parts (templates) and variable parts (parameters).

```python
# Example of parsed log
Original: "2023-04-28 08:15:34 ERROR Server failed to start on port 8080"
Template: "* ERROR Server failed to start on port *"
Parameters: ["2023-04-28 08:15:34", "8080"]
```

### 2. Feature Extraction (CodeBERT)

CodeBERT generates embeddings for log templates, capturing their semantic meaning.

### 3. Representation Learning (SimCLR)

SimCLR transforms the embeddings into a lower-dimensional space with better separation between normal and anomalous patterns. The architecture is optimized using PCDarts.

### 4. Dimensionality Reduction (VAE)

VAE further reduces dimensionality while preserving the structure relevant for anomaly detection. The architecture is optimized using PCDarts.

### 5. Anomaly Detection

Euclidean distances in the latent space help identify anomalies, with threshold determined by percentile of distances.

## Model Architecture

### SimCLR Projection Head

The optimized PCDarts architecture for SimCLR consists of:
- Multiple cells with mixed operations
- Contrastive learning objective
- Temperature-scaled cosine similarity loss

### VAE

The optimized PCDarts architecture for VAE consists of:
- Encoder: Multiple cells with mixed operations
- Latent space: Low-dimensional representation
- Decoder: Multiple cells with mixed operations
- Loss: Reconstruction + KL Divergence

## MLflow Integration

The project uses MLflow for experiment tracking and model versioning. All models, parameters, metrics, and artifacts are logged to MLflow and can be accessed through DAGsHub.

## Results Visualization

The pipeline generates visualizations to help understand the anomaly detection results:

- Distribution of distances and anomaly scores
- Confusion matrix (when ground truth is available)
- t-SNE and UMAP visualizations of the latent space

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Drain3](https://github.com/IBM/Drain3) for log parsing
- [CodeBERT](https://github.com/microsoft/CodeBERT) for feature extraction
- [MLflow](https://mlflow.org/) for experiment tracking
- [DAGsHub](https://dagshub.com/) for MLflow hosting
- [PyTorch](https://pytorch.org/) for deep learning models

