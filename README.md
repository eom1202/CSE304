# Multimodal Feature Representation Impact on Clustering Analysis

This repository contains the implementation of a comprehensive research project analyzing the impact of multimodal feature representation on clustering performance. The project implements 5 different feature extraction methods and evaluates them using 3 clustering algorithms with statistical validation.

## Features

- **5 Feature Extraction Methods:**
  - Text-only features (TF-IDF + SVD)
  - Image-only features (ResNet-18 + PCA)
  - Early fusion (concatenation before dimensionality reduction)
  - Late fusion (concatenation after individual processing)
  - Attention fusion (learned attention weights)

- **3 Clustering Algorithms:**
  - K-means clustering
  - Spectral clustering
  - Hierarchical clustering

- **Statistical Validation:**
  - Bootstrap experiments with confidence intervals
  - Statistical significance testing
  - Effect size analysis

- **Comprehensive Visualization:**
  - Performance comparison plots
  - Method complementarity analysis
  - t-SNE visualizations
  - Statistical analysis results

## Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Dataset Setup

This project uses the MS-COCO dataset. Follow these steps to set up the dataset:

### 1. Download COCO Dataset

Run the setup script to download and prepare the MS-COCO dataset:

```bash
python setup_coco.py
```

**Options:**
- `--data-dir`: Specify data directory (default: `./data/coco`)
- `--subset`: Choose dataset subset - `val2017` (~1GB) or `train2017` (~18GB) (default: `val2017`)
- `--create-config`: Create a sample configuration file

**Examples:**
```bash
# Download validation set (recommended for testing)
python setup_coco.py --subset val2017

# Download training set (larger dataset)
python setup_coco.py --subset train2017 --data-dir ./data/coco

# Create sample configuration
python setup_coco.py --create-config
```

### 2. Verify Dataset Structure

After setup, your data directory should look like:
```
data/coco/
├── annotations/
│   ├── instances_val2017.json
│   ├── captions_val2017.json
│   ├── instances_train2017.json
│   └── captions_train2017.json
└── val2017/
    ├── 000000000139.jpg
    ├── 000000000285.jpg
    └── ... (more images)
```

## Configuration

The experiment configuration is managed through `config.yaml`. Key settings include:

- **Dataset settings:** Sample size, categories, COCO data path
- **Feature extraction:** Dimensionality, fusion methods
- **Clustering:** Algorithms, K-range, auto-K method
- **Experiment:** Bootstrap runs, random seed, confidence level

Example configuration:
```yaml
dataset:
  source: "mscoco"
  n_samples: 500
  coco_data_dir: "./data/coco"
  
features:
  methods:
    - "text_only"
    - "image_only"
    - "early_fusion"
    - "late_fusion"
    - "attention_fusion"

clustering:
  algorithms:
    - "kmeans"
    - "spectral"
    - "hierarchical"

experiment:
  n_bootstrap_runs: 20
  random_seed: 42
```

## Running the Experiment

Once the dataset is set up, run the main experiment:

```bash
python main.py
```

The script will:
1. Load and preprocess the COCO dataset
2. Extract features using all 5 methods
3. Apply 3 clustering algorithms to each feature set
4. Perform statistical validation with bootstrap sampling
5. Generate comprehensive visualizations and analysis
6. Save results to the `outputs/` directory

## Output

The experiment generates several outputs in the `outputs/` directory:

- **Results:** JSON files with detailed experimental results
- **Visualizations:** 
  - Performance comparison plots
  - Method complementarity heatmaps
  - t-SNE visualizations
  - Statistical analysis charts
- **Logs:** Detailed execution logs
- **Statistics:** Bootstrap confidence intervals and significance tests

## Project Structure

```
├── main.py                 # Main experiment script
├── setup_coco.py          # COCO dataset setup script
├── config.yaml            # Experiment configuration
├── requirements.txt       # Python dependencies
├── src/
│   ├── data_loader.py     # Dataset loading and preprocessing
│   ├── feature_extractor.py  # Feature extraction methods
│   ├── clustering.py      # Clustering algorithms
│   ├── evaluation.py      # Performance evaluation
│   ├── statistical_validation.py  # Statistical analysis
│   ├── visualization.py   # Result visualization
│   └── utils.py          # Utility functions
└── outputs/               # Generated results and visualizations
```

### Performance Tips

- Use `val2017` subset for faster experimentation
- Reduce `n_bootstrap_runs` for quicker results
- Adjust `n_samples` based on available computational resources