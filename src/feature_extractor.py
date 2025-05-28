"""
Feature extraction for multimodal data
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Tuple
from PIL import Image
import re

class TextFeatureExtractor:
    """Text feature extraction using TF-IDF + SVD"""
    
    def __init__(self, dim: int = 256, max_features: int = 5000):
        """
        Initialize text feature extractor
        
        Args:
            dim: Output dimension
            max_features: Maximum number of TF-IDF features
        """
        self.dim = dim
        self.max_features = max_features
        self.vectorizer = None
        self.svd = None
        self.scaler = None
        self.fitted = False
    
    def _preprocess_text(self, texts: List[str]) -> List[str]:
        """Preprocess text data"""
        processed = []
        for text in texts:
            # Convert to lowercase
            text = text.lower()
            # Remove special characters but keep spaces
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            # Remove extra whitespace
            text = ' '.join(text.split())
            processed.append(text)
        return processed
    
    def fit(self, texts: List[str]):
        """Fit the text feature extractor"""
        processed_texts = self._preprocess_text(texts)
        
        # TF-IDF vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95  # Ignore terms that appear in more than 95% of documents
        )
        
        tfidf_features = self.vectorizer.fit_transform(processed_texts)
        
        # Dimensionality reduction with SVD
        n_components = min(self.dim, tfidf_features.shape[1], tfidf_features.shape[0])
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced_features = self.svd.fit_transform(tfidf_features)
        
        # Standardization
        self.scaler = StandardScaler()
        self.scaler.fit(reduced_features)
        
        self.fitted = True
        print(f"Text feature extractor fitted: {tfidf_features.shape[1]} -> {n_components} dimensions")
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to feature vectors"""
        if not self.fitted:
            raise ValueError("Feature extractor must be fitted first")
        
        processed_texts = self._preprocess_text(texts)
        tfidf_features = self.vectorizer.transform(processed_texts)
        reduced_features = self.svd.transform(tfidf_features)
        normalized_features = self.scaler.transform(reduced_features)
        
        # Pad or truncate to exact dimension
        if normalized_features.shape[1] < self.dim:
            padding = np.zeros((normalized_features.shape[0], self.dim - normalized_features.shape[1]))
            normalized_features = np.hstack([normalized_features, padding])
        elif normalized_features.shape[1] > self.dim:
            normalized_features = normalized_features[:, :self.dim]
        
        return normalized_features
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(texts)
        return self.transform(texts)

class ImageFeatureExtractor:
    """Image feature extraction using pre-trained ResNet"""
    
    def __init__(self, dim: int = 256, backbone: str = 'resnet18'):
        """
        Initialize image feature extractor
        
        Args:
            dim: Output dimension
            backbone: Backbone model ('resnet18', 'resnet34', 'resnet50')
        """
        self.dim = dim
        self.backbone = backbone
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained model
        if backbone == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            feature_dim = 512
        elif backbone == 'resnet34':
            self.model = models.resnet34(pretrained=True)
            feature_dim = 512
        elif backbone == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the final classification layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        # Add projection layer to match desired dimension
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.model.to(self.device)
        self.projection.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Image feature extractor initialized: {backbone} -> {dim} dimensions")
    
    def extract_features(self, images: List[Image.Image]) -> np.ndarray:
        """Extract features from images"""
        features = []
        
        with torch.no_grad():
            for i in range(0, len(images), 32):  # Process in batches
                batch_images = images[i:i+32]
                
                # Preprocess images
                batch_tensors = []
                for img in batch_images:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    tensor = self.transform(img)
                    batch_tensors.append(tensor)
                
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                
                # Extract features
                backbone_features = self.model(batch_tensor)
                backbone_features = backbone_features.view(backbone_features.size(0), -1)
                
                # Project to desired dimension
                projected_features = self.projection(backbone_features)
                
                features.append(projected_features.cpu().numpy())
        
        return np.vstack(features)

class MultimodalFusion:
    """Multimodal fusion strategies"""
    
    def __init__(self, dim: int = 256):
        """
        Initialize fusion module
        
        Args:
            dim: Output dimension
        """
        self.dim = dim
        self.fitted = False
        self.pca = None
        self.scaler = None
        
        # For attention fusion
        self.attention_weights = None
    
    def early_fusion(self, text_features: np.ndarray, image_features: np.ndarray, 
                    fit: bool = False) -> np.ndarray:
        """
        Early fusion: concatenate features and apply PCA
        
        Args:
            text_features: Text feature matrix
            image_features: Image feature matrix
            fit: Whether to fit the PCA
            
        Returns:
            Fused feature matrix
        """
        # Concatenate features
        concatenated = np.hstack([text_features, image_features])
        
        if fit:
            # Apply PCA for dimensionality reduction
            n_components = min(self.dim, concatenated.shape[1], concatenated.shape[0])
            self.pca = PCA(n_components=n_components, random_state=42)
            reduced_features = self.pca.fit_transform(concatenated)
            
            # Standardization
            self.scaler = StandardScaler()
            normalized_features = self.scaler.fit_transform(reduced_features)
            self.fitted = True
        else:
            if not self.fitted:
                raise ValueError("Fusion must be fitted first")
            reduced_features = self.pca.transform(concatenated)
            normalized_features = self.scaler.transform(reduced_features)
        
        # Pad or truncate to exact dimension
        if normalized_features.shape[1] < self.dim:
            padding = np.zeros((normalized_features.shape[0], self.dim - normalized_features.shape[1]))
            normalized_features = np.hstack([normalized_features, padding])
        elif normalized_features.shape[1] > self.dim:
            normalized_features = normalized_features[:, :self.dim]
        
        return normalized_features
    
    def late_fusion(self, text_features: np.ndarray, image_features: np.ndarray) -> np.ndarray:
        """
        Late fusion: weighted average of normalized features
        
        Args:
            text_features: Text feature matrix
            image_features: Image feature matrix
            
        Returns:
            Fused feature matrix
        """
        # Normalize features
        text_norm = text_features / (np.linalg.norm(text_features, axis=1, keepdims=True) + 1e-8)
        image_norm = image_features / (np.linalg.norm(image_features, axis=1, keepdims=True) + 1e-8)
        
        # Simple weighted average (can be learned)
        weight_text = 0.5
        weight_image = 0.5
        
        fused = weight_text * text_norm + weight_image * image_norm
        
        return fused
    
    def attention_fusion(self, text_features: np.ndarray, image_features: np.ndarray,
                        fit: bool = False) -> np.ndarray:
        """
        Attention-based fusion: learn attention weights
        
        Args:
            text_features: Text feature matrix
            image_features: Image feature matrix
            fit: Whether to fit the attention weights
            
        Returns:
            Fused feature matrix
        """
        if fit:
            # Simple attention mechanism: compute similarity-based weights
            # Normalize features
            text_norm = text_features / (np.linalg.norm(text_features, axis=1, keepdims=True) + 1e-8)
            image_norm = image_features / (np.linalg.norm(image_features, axis=1, keepdims=True) + 1e-8)
            
            # Compute cross-modal similarity
            similarity = np.sum(text_norm * image_norm, axis=1, keepdims=True)
            
            # Compute attention weights based on similarity
            attention_text = 0.5 + 0.3 * similarity  # Base weight + similarity bonus
            attention_image = 1.0 - attention_text
            
            # Store average attention weights
            self.attention_weights = {
                'text': np.mean(attention_text),
                'image': np.mean(attention_image)
            }
            
            self.fitted = True
        else:
            if not self.fitted:
                # Use default weights if not fitted
                attention_text = 0.5
                attention_image = 0.5
            else:
                attention_text = self.attention_weights['text']
                attention_image = self.attention_weights['image']
        
        # Normalize features
        text_norm = text_features / (np.linalg.norm(text_features, axis=1, keepdims=True) + 1e-8)
        image_norm = image_features / (np.linalg.norm(image_features, axis=1, keepdims=True) + 1e-8)
        
        # Apply attention weights
        if isinstance(attention_text, np.ndarray):
            fused = attention_text * text_norm + attention_image * image_norm
        else:
            fused = attention_text * text_norm + attention_image * image_norm
        
        return fused

class FeatureExtractor:
    """Main feature extraction coordinator"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature extractor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.text_dim = config.get('text_dim', 256)
        self.image_dim = config.get('image_dim', 256)
        self.fusion_dim = config.get('fusion_dim', 256)
        
        # Initialize extractors
        self.text_extractor = TextFeatureExtractor(dim=self.text_dim)
        self.image_extractor = ImageFeatureExtractor(dim=self.image_dim)
        self.fusion = MultimodalFusion(dim=self.fusion_dim)
        
        self.fitted = False
    
    def fit(self, data: List[Dict[str, Any]]):
        """Fit all feature extractors"""
        texts = [sample['caption'] for sample in data]
        
        # Fit text extractor
        self.text_extractor.fit(texts)
        
        # Image extractor doesn't need fitting (pre-trained)
        
        # Fit fusion modules
        text_features = self.text_extractor.transform(texts)
        images = [sample['image'] for sample in data]
        image_features = self.image_extractor.extract_features(images)
        
        # Fit fusion strategies
        self.fusion.early_fusion(text_features, image_features, fit=True)
        self.fusion.attention_fusion(text_features, image_features, fit=True)
        
        self.fitted = True
        print("All feature extractors fitted successfully")
    
    def extract_features(self, data: List[Dict[str, Any]], method: str) -> np.ndarray:
        """
        Extract features using specified method
        
        Args:
            data: List of data samples
            method: Feature extraction method
            
        Returns:
            Feature matrix
        """
        if not self.fitted and method != 'image_only':
            raise ValueError("Feature extractors must be fitted first")
        
        texts = [sample['caption'] for sample in data]
        images = [sample['image'] for sample in data]
        
        if method == 'text_only':
            return self.text_extractor.transform(texts)
        
        elif method == 'image_only':
            return self.image_extractor.extract_features(images)
        
        elif method == 'early_fusion':
            text_features = self.text_extractor.transform(texts)
            image_features = self.image_extractor.extract_features(images)
            return self.fusion.early_fusion(text_features, image_features, fit=False)
        
        elif method == 'late_fusion':
            text_features = self.text_extractor.transform(texts)
            image_features = self.image_extractor.extract_features(images)
            return self.fusion.late_fusion(text_features, image_features)
        
        elif method == 'attention_fusion':
            text_features = self.text_extractor.transform(texts)
            image_features = self.image_extractor.extract_features(images)
            return self.fusion.attention_fusion(text_features, image_features, fit=False)
        
        else:
            raise ValueError(f"Unsupported feature extraction method: {method}")
    
    def get_all_features(self, data: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Extract features using all methods"""
        methods = ['text_only', 'image_only', 'early_fusion', 'late_fusion', 'attention_fusion']
        features = {}
        
        for method in methods:
            print(f"Extracting {method} features...")
            features[method] = self.extract_features(data, method)
        
        return features 