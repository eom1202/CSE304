"""
Clustering algorithms and optimization
"""

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdaptiveClusteringFramework:
    """Adaptive clustering framework with multiple algorithms"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize clustering framework
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.algorithms = config.get('algorithms', ['kmeans', 'spectral', 'hierarchical'])
        self.k_range = config.get('k_range', [3, 10])
        self.auto_k_method = config.get('auto_k_method', 'elbow')
        self.random_state = config.get('random_seed', 42)
    
    def determine_optimal_k(self, embeddings: np.ndarray, method: str = 'elbow') -> int:
        """
        Determine optimal number of clusters
        
        Args:
            embeddings: Feature embeddings
            method: Method for determining k ('elbow', 'silhouette')
            
        Returns:
            Optimal number of clusters
        """
        k_min, k_max = self.k_range
        scores = []
        k_values = list(range(k_min, k_max + 1))
        
        for k in k_values:
            try:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                
                if method == 'elbow':
                    scores.append(kmeans.inertia_)
                elif method == 'silhouette':
                    if k > 1:  # Silhouette score requires at least 2 clusters
                        score = silhouette_score(embeddings, labels)
                        scores.append(score)
                    else:
                        scores.append(0)
                        
            except Exception as e:
                print(f"Warning: Failed to compute score for k={k}: {e}")
                scores.append(0 if method == 'silhouette' else float('inf'))
        
        if method == 'elbow':
            # Find elbow point using the "knee" method
            optimal_k = self._find_elbow_point(k_values, scores)
        elif method == 'silhouette':
            # Find k with maximum silhouette score
            optimal_k = k_values[np.argmax(scores)]
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return optimal_k
    
    def _find_elbow_point(self, k_values: List[int], scores: List[float]) -> int:
        """Find elbow point in the scores curve"""
        scores = np.array(scores)
        n_points = len(scores)
        
        # Create coordinate arrays
        all_coord = np.vstack((range(n_points), scores)).T
        
        # Calculate the line from first to last point
        first_point = all_coord[0]
        line_vec = all_coord[-1] - all_coord[0]
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
        
        # Calculate distance from each point to the line
        vec_from_first = all_coord - first_point
        scalar_product = np.sum(vec_from_first * line_vec_norm, axis=1)
        vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
        vec_to_line = vec_from_first - vec_from_first_parallel
        dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
        
        # Find the point with maximum distance to the line (elbow)
        elbow_idx = np.argmax(dist_to_line)
        optimal_k = k_values[elbow_idx]
        
        return optimal_k
    
    def cluster_with_algorithm(self, embeddings: np.ndarray, algorithm: str, 
                              k: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform clustering with specified algorithm
        
        Args:
            embeddings: Feature embeddings
            algorithm: Clustering algorithm ('kmeans', 'spectral', 'hierarchical')
            k: Number of clusters (if None, will be determined automatically)
            
        Returns:
            Tuple of (cluster_labels, metadata)
        """
        if k is None:
            k = self.determine_optimal_k(embeddings, self.auto_k_method)
        
        metadata = {'algorithm': algorithm, 'k': k, 'n_samples': len(embeddings)}
        
        try:
            if algorithm == 'kmeans':
                clusterer = KMeans(
                    n_clusters=k, 
                    random_state=self.random_state,
                    n_init=10,
                    max_iter=300
                )
                labels = clusterer.fit_predict(embeddings)
                metadata['inertia'] = clusterer.inertia_
                metadata['n_iter'] = clusterer.n_iter_
                
            elif algorithm == 'spectral':
                clusterer = SpectralClustering(
                    n_clusters=k,
                    random_state=self.random_state,
                    affinity='rbf',
                    gamma=1.0
                )
                labels = clusterer.fit_predict(embeddings)
                metadata['affinity'] = 'rbf'
                
            elif algorithm == 'hierarchical':
                clusterer = AgglomerativeClustering(
                    n_clusters=k,
                    linkage='ward'
                )
                labels = clusterer.fit_predict(embeddings)
                metadata['linkage'] = 'ward'
                
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            metadata['success'] = True
            
        except Exception as e:
            print(f"Error in {algorithm} clustering: {e}")
            # Fallback to random labels
            labels = np.random.randint(0, k, size=len(embeddings))
            metadata['success'] = False
            metadata['error'] = str(e)
        
        return labels, metadata
    
    def multi_algorithm_clustering(self, embeddings: np.ndarray, 
                                  k: int = None) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Perform clustering with multiple algorithms
        
        Args:
            embeddings: Feature embeddings
            k: Number of clusters (if None, will be determined for each algorithm)
            
        Returns:
            Dictionary mapping algorithm names to (labels, metadata) tuples
        """
        results = {}
        
        for algorithm in self.algorithms:
            print(f"Running {algorithm} clustering...")
            labels, metadata = self.cluster_with_algorithm(embeddings, algorithm, k)
            results[algorithm] = (labels, metadata)
        
        return results
    
    def evaluate_clustering_quality(self, embeddings: np.ndarray, 
                                   labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate clustering quality using multiple metrics
        
        Args:
            embeddings: Feature embeddings
            labels: Cluster labels
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        try:
            # Silhouette Score
            if len(np.unique(labels)) > 1:
                metrics['silhouette_score'] = silhouette_score(embeddings, labels)
            else:
                metrics['silhouette_score'] = 0.0
            
            # Calinski-Harabasz Index
            from sklearn.metrics import calinski_harabasz_score
            if len(np.unique(labels)) > 1:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(embeddings, labels)
            else:
                metrics['calinski_harabasz_score'] = 0.0
            
            # Davies-Bouldin Index
            from sklearn.metrics import davies_bouldin_score
            if len(np.unique(labels)) > 1:
                metrics['davies_bouldin_score'] = davies_bouldin_score(embeddings, labels)
            else:
                metrics['davies_bouldin_score'] = float('inf')
            
            # Inertia (for K-means like evaluation)
            metrics['inertia'] = self._calculate_inertia(embeddings, labels)
            
            # Composite score (normalized combination)
            metrics['composite_score'] = self._calculate_composite_score(metrics)
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            metrics = {
                'silhouette_score': 0.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_score': float('inf'),
                'inertia': float('inf'),
                'composite_score': 0.0
            }
        
        return metrics
    
    def _calculate_inertia(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Calculate inertia (within-cluster sum of squares)"""
        inertia = 0.0
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            cluster_points = embeddings[labels == label]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                inertia += np.sum((cluster_points - centroid) ** 2)
        
        return inertia
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate composite score from multiple metrics
        
        Higher is better for this composite score
        """
        # Normalize silhouette score (already in [-1, 1], shift to [0, 2])
        silhouette_norm = (metrics['silhouette_score'] + 1) / 2
        
        # Normalize Calinski-Harabasz (log scale, then normalize)
        ch_score = metrics['calinski_harabasz_score']
        ch_norm = np.log(ch_score + 1) / 10  # Rough normalization
        ch_norm = min(ch_norm, 1.0)  # Cap at 1
        
        # Normalize Davies-Bouldin (invert since lower is better)
        db_score = metrics['davies_bouldin_score']
        if db_score == float('inf'):
            db_norm = 0.0
        else:
            db_norm = 1 / (1 + db_score)  # Invert and normalize
        
        # Weighted combination
        composite = 0.5 * silhouette_norm + 0.3 * ch_norm + 0.2 * db_norm
        
        return composite
    
    def run_comprehensive_clustering(self, embeddings: np.ndarray, 
                                   method_name: str = "unknown") -> Dict[str, Any]:
        """
        Run comprehensive clustering analysis
        
        Args:
            embeddings: Feature embeddings
            method_name: Name of the feature extraction method
            
        Returns:
            Comprehensive clustering results
        """
        print(f"\nRunning comprehensive clustering for {method_name}...")
        
        # Determine optimal k
        optimal_k = self.determine_optimal_k(embeddings, self.auto_k_method)
        print(f"Optimal k determined: {optimal_k}")
        
        # Run multiple algorithms
        clustering_results = self.multi_algorithm_clustering(embeddings, optimal_k)
        
        # Evaluate each algorithm
        evaluation_results = {}
        for algorithm, (labels, metadata) in clustering_results.items():
            metrics = self.evaluate_clustering_quality(embeddings, labels)
            evaluation_results[algorithm] = {
                'labels': labels,
                'metadata': metadata,
                'metrics': metrics
            }
        
        # Find best algorithm based on composite score
        best_algorithm = max(evaluation_results.keys(), 
                           key=lambda alg: evaluation_results[alg]['metrics']['composite_score'])
        
        results = {
            'method_name': method_name,
            'optimal_k': optimal_k,
            'algorithms': evaluation_results,
            'best_algorithm': best_algorithm,
            'best_score': evaluation_results[best_algorithm]['metrics']['composite_score'],
            'best_labels': evaluation_results[best_algorithm]['labels'],
            'embeddings_shape': embeddings.shape
        }
        
        print(f"Best algorithm for {method_name}: {best_algorithm} "
              f"(score: {results['best_score']:.3f})")
        
        return results 