"""
Clustering evaluation metrics and analysis
"""

import numpy as np
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, v_measure_score
)
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class ClusteringEvaluator:
    """Comprehensive clustering evaluation"""
    
    def __init__(self):
        """Initialize evaluator"""
        pass
    
    def intrinsic_evaluation(self, embeddings: np.ndarray, 
                           cluster_labels: np.ndarray) -> Dict[str, float]:
        """
        Intrinsic evaluation (no ground truth required)
        
        Args:
            embeddings: Feature embeddings
            cluster_labels: Predicted cluster labels
            
        Returns:
            Dictionary of intrinsic metrics
        """
        metrics = {}
        
        try:
            n_clusters = len(np.unique(cluster_labels))
            
            if n_clusters > 1 and n_clusters < len(embeddings):
                # Silhouette Score: [-1, 1], higher is better
                metrics['silhouette_score'] = silhouette_score(embeddings, cluster_labels)
                
                # Calinski-Harabasz Index: positive, higher is better
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(embeddings, cluster_labels)
                
                # Davies-Bouldin Index: positive, lower is better
                metrics['davies_bouldin_score'] = davies_bouldin_score(embeddings, cluster_labels)
            else:
                # Edge cases: all points in one cluster or each point is its own cluster
                metrics['silhouette_score'] = 0.0
                metrics['calinski_harabasz_score'] = 0.0
                metrics['davies_bouldin_score'] = float('inf')
            
            # Additional metrics
            metrics['inertia'] = self._calculate_inertia(embeddings, cluster_labels)
            metrics['n_clusters'] = n_clusters
            metrics['cluster_sizes'] = self._calculate_cluster_sizes(cluster_labels)
            
            # Composite score
            metrics['composite_score'] = self._calculate_composite_score(metrics)
            
        except Exception as e:
            print(f"Error in intrinsic evaluation: {e}")
            metrics = self._get_default_metrics()
        
        return metrics
    
    def extrinsic_evaluation(self, true_labels: np.ndarray, 
                           predicted_labels: np.ndarray) -> Dict[str, float]:
        """
        Extrinsic evaluation (requires ground truth)
        
        Args:
            true_labels: Ground truth labels
            predicted_labels: Predicted cluster labels
            
        Returns:
            Dictionary of extrinsic metrics
        """
        metrics = {}
        
        try:
            # Adjusted Rand Index: [-1, 1], higher is better
            metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, predicted_labels)
            
            # Normalized Mutual Information: [0, 1], higher is better
            metrics['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, predicted_labels)
            
            # V-measure: [0, 1], higher is better (harmonic mean of homogeneity and completeness)
            metrics['v_measure'] = v_measure_score(true_labels, predicted_labels)
            
            # Homogeneity and Completeness separately
            from sklearn.metrics import homogeneity_score, completeness_score
            metrics['homogeneity'] = homogeneity_score(true_labels, predicted_labels)
            metrics['completeness'] = completeness_score(true_labels, predicted_labels)
            
            # Accuracy-like metric (best matching between clusters and true labels)
            metrics['cluster_accuracy'] = self._calculate_cluster_accuracy(true_labels, predicted_labels)
            
        except Exception as e:
            print(f"Error in extrinsic evaluation: {e}")
            metrics = {
                'adjusted_rand_score': 0.0,
                'normalized_mutual_info': 0.0,
                'v_measure': 0.0,
                'homogeneity': 0.0,
                'completeness': 0.0,
                'cluster_accuracy': 0.0
            }
        
        return metrics
    
    def comprehensive_evaluation(self, embeddings: np.ndarray, 
                               predicted_labels: np.ndarray,
                               true_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation combining intrinsic and extrinsic metrics
        
        Args:
            embeddings: Feature embeddings
            predicted_labels: Predicted cluster labels
            true_labels: Ground truth labels (optional)
            
        Returns:
            Dictionary of all evaluation metrics
        """
        results = {}
        
        # Intrinsic evaluation
        intrinsic_metrics = self.intrinsic_evaluation(embeddings, predicted_labels)
        results['intrinsic'] = intrinsic_metrics
        
        # Extrinsic evaluation (if ground truth available)
        if true_labels is not None:
            extrinsic_metrics = self.extrinsic_evaluation(true_labels, predicted_labels)
            results['extrinsic'] = extrinsic_metrics
        else:
            results['extrinsic'] = None
        
        # Overall quality score
        results['overall_score'] = intrinsic_metrics['composite_score']
        
        return results
    
    def _calculate_inertia(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Calculate within-cluster sum of squares (inertia)"""
        inertia = 0.0
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            cluster_points = embeddings[labels == label]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                inertia += np.sum((cluster_points - centroid) ** 2)
        
        return inertia
    
    def _calculate_cluster_sizes(self, labels: np.ndarray) -> Dict[str, Any]:
        """Calculate cluster size statistics"""
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        return {
            'mean_size': np.mean(counts),
            'std_size': np.std(counts),
            'min_size': np.min(counts),
            'max_size': np.max(counts),
            'size_distribution': dict(zip(unique_labels.astype(str), counts.tolist()))
        }
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate composite score from intrinsic metrics
        Higher is better
        """
        # Normalize silhouette score (from [-1, 1] to [0, 1])
        silhouette_norm = (metrics['silhouette_score'] + 1) / 2
        
        # Normalize Calinski-Harabasz (log scale)
        ch_score = metrics['calinski_harabasz_score']
        ch_norm = np.log(ch_score + 1) / 10
        ch_norm = min(ch_norm, 1.0)  # Cap at 1
        
        # Normalize Davies-Bouldin (invert since lower is better)
        db_score = metrics['davies_bouldin_score']
        if db_score == float('inf') or db_score == 0:
            db_norm = 0.0
        else:
            db_norm = 1 / (1 + db_score)
        
        # Weighted combination
        composite = 0.5 * silhouette_norm + 0.3 * ch_norm + 0.2 * db_norm
        
        return composite
    
    def _calculate_cluster_accuracy(self, true_labels: np.ndarray, 
                                  predicted_labels: np.ndarray) -> float:
        """
        Calculate cluster accuracy using optimal assignment
        """
        from scipy.optimize import linear_sum_assignment
        
        # Create confusion matrix
        true_unique = np.unique(true_labels)
        pred_unique = np.unique(predicted_labels)
        
        confusion_matrix = np.zeros((len(true_unique), len(pred_unique)))
        
        for i, true_label in enumerate(true_unique):
            for j, pred_label in enumerate(pred_unique):
                confusion_matrix[i, j] = np.sum((true_labels == true_label) & 
                                              (predicted_labels == pred_label))
        
        # Find optimal assignment
        row_indices, col_indices = linear_sum_assignment(-confusion_matrix)
        
        # Calculate accuracy
        total_correct = confusion_matrix[row_indices, col_indices].sum()
        accuracy = total_correct / len(true_labels)
        
        return accuracy
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """Get default metrics for error cases"""
        return {
            'silhouette_score': 0.0,
            'calinski_harabasz_score': 0.0,
            'davies_bouldin_score': float('inf'),
            'inertia': float('inf'),
            'n_clusters': 1,
            'cluster_sizes': {'mean_size': 0, 'std_size': 0, 'min_size': 0, 'max_size': 0},
            'composite_score': 0.0
        }

class ComplementarityAnalyzer:
    """Analyze complementarity between different feature extraction methods"""
    
    def __init__(self):
        """Initialize analyzer"""
        pass
    
    def analyze_method_complementarity(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze how different methods complement each other
        
        Args:
            results: Dictionary mapping method names to clustering results
            
        Returns:
            Complementarity analysis results
        """
        method_names = list(results.keys())
        n_methods = len(method_names)
        
        # Initialize matrices
        agreement_matrix = np.zeros((n_methods, n_methods))
        performance_matrix = np.zeros((n_methods, n_methods))
        
        # Calculate pairwise agreements and performance differences
        for i, method1 in enumerate(method_names):
            for j, method2 in enumerate(method_names):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                    performance_matrix[i, j] = 0.0
                else:
                    # Calculate agreement (Adjusted Rand Index)
                    labels1 = results[method1]['best_labels']
                    labels2 = results[method2]['best_labels']
                    agreement = adjusted_rand_score(labels1, labels2)
                    agreement_matrix[i, j] = agreement
                    
                    # Calculate performance difference
                    score1 = results[method1]['best_score']
                    score2 = results[method2]['best_score']
                    performance_matrix[i, j] = score1 - score2
        
        # Find most complementary pairs (low agreement, high combined performance)
        complementarity_scores = {}
        for i, method1 in enumerate(method_names):
            for j, method2 in enumerate(method_names):
                if i < j:  # Avoid duplicates
                    agreement = agreement_matrix[i, j]
                    avg_performance = (results[method1]['best_score'] + 
                                     results[method2]['best_score']) / 2
                    
                    # Complementarity: high performance, low agreement
                    complementarity = avg_performance * (1 - agreement)
                    complementarity_scores[f"{method1}_vs_{method2}"] = {
                        'agreement': agreement,
                        'avg_performance': avg_performance,
                        'complementarity': complementarity
                    }
        
        # Rank methods by individual performance
        method_rankings = sorted(method_names, 
                               key=lambda m: results[m]['best_score'], 
                               reverse=True)
        
        analysis_results = {
            'agreement_matrix': agreement_matrix,
            'performance_matrix': performance_matrix,
            'method_names': method_names,
            'complementarity_scores': complementarity_scores,
            'method_rankings': method_rankings,
            'best_method': method_rankings[0],
            'most_complementary_pair': max(complementarity_scores.keys(), 
                                         key=lambda k: complementarity_scores[k]['complementarity'])
        }
        
        return analysis_results
    
    def generate_insights(self, complementarity_results: Dict[str, Any], 
                         method_results: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Generate human-readable insights from complementarity analysis
        
        Args:
            complementarity_results: Results from analyze_method_complementarity
            method_results: Original clustering results
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Best performing method
        best_method = complementarity_results['best_method']
        best_score = method_results[best_method]['best_score']
        insights.append(f"Best performing method: {best_method} (score: {best_score:.3f})")
        
        # Most complementary pair
        most_comp_pair = complementarity_results['most_complementary_pair']
        comp_info = complementarity_results['complementarity_scores'][most_comp_pair]
        insights.append(f"Most complementary methods: {most_comp_pair.replace('_vs_', ' and ')} "
                       f"(complementarity: {comp_info['complementarity']:.3f})")
        
        # Performance gaps
        rankings = complementarity_results['method_rankings']
        if len(rankings) > 1:
            gap = method_results[rankings[0]]['best_score'] - method_results[rankings[-1]]['best_score']
            insights.append(f"Performance gap between best and worst: {gap:.3f}")
        
        # Agreement patterns
        agreement_matrix = complementarity_results['agreement_matrix']
        method_names = complementarity_results['method_names']
        
        # Find most similar methods
        max_agreement = 0
        most_similar_pair = None
        for i in range(len(method_names)):
            for j in range(i+1, len(method_names)):
                if agreement_matrix[i, j] > max_agreement:
                    max_agreement = agreement_matrix[i, j]
                    most_similar_pair = (method_names[i], method_names[j])
        
        if most_similar_pair:
            insights.append(f"Most similar methods: {most_similar_pair[0]} and {most_similar_pair[1]} "
                           f"(agreement: {max_agreement:.3f})")
        
        return insights 