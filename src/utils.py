"""
Utility functions for multimodal clustering experiments
"""

import os
import json
import yaml
import numpy as np
import random
import torch
from datetime import datetime
from typing import Dict, List, Any, Tuple
import logging

def setup_logging(log_file: str = "experiment_log.txt") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def ensure_dir(directory: str):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_json(data: Dict[str, Any], filepath: str):
    """Save data to JSON file"""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize features to unit variance"""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(features)

def calculate_optimal_k(embeddings: np.ndarray, k_range: Tuple[int, int] = (2, 10), 
                       method: str = "elbow") -> int:
    """Calculate optimal number of clusters"""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    k_min, k_max = k_range
    scores = []
    
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        if method == "elbow":
            scores.append(kmeans.inertia_)
        elif method == "silhouette":
            scores.append(silhouette_score(embeddings, labels))
    
    if method == "elbow":
        # Find elbow point using the "knee" method
        scores = np.array(scores)
        n_points = len(scores)
        all_coord = np.vstack((range(n_points), scores)).T
        first_point = all_coord[0]
        line_vec = all_coord[-1] - all_coord[0]
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
        
        vec_from_first = all_coord - first_point
        scalar_product = np.sum(vec_from_first * line_vec_norm, axis=1)
        vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
        vec_to_line = vec_from_first - vec_from_first_parallel
        dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
        
        optimal_k = k_min + np.argmax(dist_to_line)
    else:  # silhouette
        optimal_k = k_min + np.argmax(scores)
    
    return optimal_k

def create_timestamp() -> str:
    """Create timestamp string for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

class ProgressTracker:
    """Track experiment progress"""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = datetime.now()
    
    def update(self, step_name: str = ""):
        """Update progress"""
        self.current_step += 1
        elapsed = datetime.now() - self.start_time
        progress = self.current_step / self.total_steps
        
        print(f"Progress: {progress:.1%} ({self.current_step}/{self.total_steps}) - {step_name}")
        print(f"Elapsed: {elapsed}, ETA: {elapsed / progress - elapsed if progress > 0 else 'Unknown'}") 