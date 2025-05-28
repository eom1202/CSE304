"""
Statistical validation and bootstrap experiments
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class StatisticalValidator:
    """Statistical validation for clustering experiments"""
    
    def __init__(self, alpha: float = 0.05, confidence_level: float = 0.95):
        """
        Initialize statistical validator
        
        Args:
            alpha: Significance level for hypothesis testing
            confidence_level: Confidence level for intervals
        """
        self.alpha = alpha
        self.confidence_level = confidence_level
    
    def bootstrap_experiment(self, data: List[Dict[str, Any]], 
                           feature_extractor, clustering_framework,
                           n_runs: int = 20) -> Dict[str, Any]:
        """
        Run bootstrap experiments for robust statistical analysis
        
        Args:
            data: Original dataset
            feature_extractor: Feature extraction object
            clustering_framework: Clustering framework object
            n_runs: Number of bootstrap runs
            
        Returns:
            Bootstrap experiment results
        """
        print(f"Running {n_runs} bootstrap experiments...")
        
        methods = ['text_only', 'image_only', 'early_fusion', 'late_fusion', 'attention_fusion']
        bootstrap_results = {method: [] for method in methods}
        
        for run in range(n_runs):
            print(f"Bootstrap run {run + 1}/{n_runs}")
            
            # Bootstrap sampling
            bootstrap_data = self._bootstrap_sample(data)
            
            # Extract features for all methods
            try:
                all_features = feature_extractor.get_all_features(bootstrap_data)
                
                # Run clustering for each method
                for method in methods:
                    embeddings = all_features[method]
                    results = clustering_framework.run_comprehensive_clustering(embeddings, method)
                    
                    # Store key metrics
                    bootstrap_results[method].append({
                        'silhouette_score': results['algorithms'][results['best_algorithm']]['metrics']['silhouette_score'],
                        'composite_score': results['best_score'],
                        'optimal_k': results['optimal_k'],
                        'best_algorithm': results['best_algorithm']
                    })
                    
            except Exception as e:
                print(f"Error in bootstrap run {run + 1}: {e}")
                # Add default values for failed runs
                for method in methods:
                    bootstrap_results[method].append({
                        'silhouette_score': 0.0,
                        'composite_score': 0.0,
                        'optimal_k': 3,
                        'best_algorithm': 'kmeans'
                    })
        
        # Aggregate results
        aggregated_results = self._aggregate_bootstrap_results(bootstrap_results)
        
        return aggregated_results
    
    def _bootstrap_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create bootstrap sample from data"""
        n_samples = len(data)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return [data[i] for i in indices]
    
    def _aggregate_bootstrap_results(self, bootstrap_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Aggregate bootstrap results with statistics"""
        aggregated = {}
        
        for method, runs in bootstrap_results.items():
            if not runs:  # Skip if no results
                continue
                
            # Extract metrics across runs
            silhouette_scores = [run['silhouette_score'] for run in runs]
            composite_scores = [run['composite_score'] for run in runs]
            optimal_ks = [run['optimal_k'] for run in runs]
            
            # Calculate statistics
            aggregated[method] = {
                'silhouette_score': {
                    'mean': np.mean(silhouette_scores),
                    'std': np.std(silhouette_scores),
                    'median': np.median(silhouette_scores),
                    'confidence_interval': self._calculate_confidence_interval(silhouette_scores)
                },
                'composite_score': {
                    'mean': np.mean(composite_scores),
                    'std': np.std(composite_scores),
                    'median': np.median(composite_scores),
                    'confidence_interval': self._calculate_confidence_interval(composite_scores)
                },
                'optimal_k': {
                    'mean': np.mean(optimal_ks),
                    'std': np.std(optimal_ks),
                    'mode': stats.mode(optimal_ks, keepdims=False).mode if optimal_ks else 3
                },
                'n_runs': len(runs),
                'raw_scores': {
                    'silhouette': silhouette_scores,
                    'composite': composite_scores
                }
            }
        
        return aggregated
    
    def _calculate_confidence_interval(self, values: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for a list of values"""
        if not values:
            return (0.0, 0.0)
        
        values = np.array(values)
        n = len(values)
        
        if n < 2:
            return (values[0], values[0])
        
        # Use t-distribution for small samples
        mean = np.mean(values)
        std_err = stats.sem(values)  # Standard error of the mean
        
        # Calculate confidence interval
        confidence = self.confidence_level
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin_error = t_value * std_err
        
        return (mean - margin_error, mean + margin_error)
    
    def paired_significance_test(self, method_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform paired significance tests between methods
        
        Args:
            method_results: Bootstrap results for all methods
            
        Returns:
            Statistical test results
        """
        methods = list(method_results.keys())
        n_methods = len(methods)
        
        # Initialize results
        test_results = {
            'pairwise_tests': {},
            'method_rankings': [],
            'significant_differences': []
        }
        
        # Perform pairwise tests
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i < j:  # Avoid duplicates
                    pair_key = f"{method1}_vs_{method2}"
                    
                    # Get scores for both methods - fix the data structure access
                    scores1 = method_results[method1]['raw_scores']['composite']
                    scores2 = method_results[method2]['raw_scores']['composite']
                    
                    # Perform paired t-test
                    test_result = self._perform_paired_test(scores1, scores2, method1, method2)
                    test_results['pairwise_tests'][pair_key] = test_result
                    
                    # Check for significant differences
                    if test_result['significant']:
                        test_results['significant_differences'].append({
                            'methods': (method1, method2),
                            'p_value': test_result['p_value'],
                            'effect_size': test_result['effect_size'],
                            'better_method': test_result['better_method']
                        })
        
        # Apply multiple comparison correction
        test_results = self._apply_multiple_comparison_correction(test_results)
        
        # Rank methods by performance
        method_means = {method: results['composite_score']['mean'] 
                       for method, results in method_results.items()}
        test_results['method_rankings'] = sorted(method_means.items(), 
                                                key=lambda x: x[1], reverse=True)
        
        return test_results
    
    def _perform_paired_test(self, scores1: List[float], scores2: List[float], 
                           method1: str, method2: str) -> Dict[str, Any]:
        """Perform paired statistical test between two methods"""
        scores1 = np.array(scores1)
        scores2 = np.array(scores2)
        
        # Check normality (Shapiro-Wilk test)
        _, p_norm1 = stats.shapiro(scores1) if len(scores1) > 3 else (None, 0.05)
        _, p_norm2 = stats.shapiro(scores2) if len(scores2) > 3 else (None, 0.05)
        
        # Choose appropriate test
        if p_norm1 > 0.05 and p_norm2 > 0.05:
            # Both distributions are normal, use paired t-test
            statistic, p_value = stats.ttest_rel(scores1, scores2)
            test_type = 'paired_t_test'
        else:
            # Non-normal distributions, use Wilcoxon signed-rank test
            statistic, p_value = stats.wilcoxon(scores1, scores2, alternative='two-sided')
            test_type = 'wilcoxon_signed_rank'
        
        # Calculate effect size (Cohen's d for paired samples)
        effect_size = self._calculate_cohens_d_paired(scores1, scores2)
        
        # Determine which method is better
        mean1, mean2 = np.mean(scores1), np.mean(scores2)
        better_method = method1 if mean1 > mean2 else method2
        
        return {
            'test_type': test_type,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'effect_size': effect_size,
            'effect_size_interpretation': self._interpret_effect_size(effect_size),
            'better_method': better_method,
            'mean_difference': mean1 - mean2,
            'normality_p_values': (p_norm1, p_norm2)
        }
    
    def _calculate_cohens_d_paired(self, scores1: np.ndarray, scores2: np.ndarray) -> float:
        """Calculate Cohen's d for paired samples"""
        differences = scores1 - scores2
        return np.mean(differences) / np.std(differences, ddof=1)
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"
    
    def _apply_multiple_comparison_correction(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Bonferroni correction for multiple comparisons"""
        pairwise_tests = test_results['pairwise_tests']
        n_tests = len(pairwise_tests)
        
        if n_tests == 0:
            return test_results
        
        # Bonferroni correction
        corrected_alpha = self.alpha / n_tests
        
        # Update significance based on corrected alpha
        for pair_key, test_result in pairwise_tests.items():
            test_result['corrected_p_value'] = test_result['p_value'] * n_tests
            test_result['significant_corrected'] = test_result['p_value'] < corrected_alpha
        
        # Update significant differences list
        test_results['significant_differences'] = [
            diff for diff in test_results['significant_differences']
            if pairwise_tests[f"{diff['methods'][0]}_vs_{diff['methods'][1]}"]['significant_corrected']
        ]
        
        test_results['correction_info'] = {
            'method': 'bonferroni',
            'n_tests': n_tests,
            'corrected_alpha': corrected_alpha,
            'original_alpha': self.alpha
        }
        
        return test_results
    
    def effect_size_analysis(self, method_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze effect sizes between methods
        
        Args:
            method_results: Bootstrap results for all methods
            
        Returns:
            Effect size analysis results
        """
        methods = list(method_results.keys())
        effect_sizes = {}
        
        # Calculate effect sizes for all pairs
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i < j:
                    scores1 = method_results[method1]['raw_scores']['composite']
                    scores2 = method_results[method2]['raw_scores']['composite']
                    
                    effect_size = self._calculate_cohens_d_paired(np.array(scores1), np.array(scores2))
                    
                    effect_sizes[f"{method1}_vs_{method2}"] = {
                        'effect_size': effect_size,
                        'interpretation': self._interpret_effect_size(effect_size),
                        'practical_significance': abs(effect_size) >= 0.5  # Medium or larger
                    }
        
        # Find largest effect sizes
        largest_effects = sorted(effect_sizes.items(), 
                               key=lambda x: abs(x[1]['effect_size']), 
                               reverse=True)
        
        return {
            'effect_sizes': effect_sizes,
            'largest_effects': largest_effects[:3],  # Top 3
            'practically_significant': [k for k, v in effect_sizes.items() 
                                      if v['practical_significance']]
        }
    
    def confidence_intervals(self, method_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate and analyze confidence intervals
        
        Args:
            method_results: Bootstrap results for all methods
            
        Returns:
            Confidence interval analysis
        """
        intervals = {}
        overlaps = {}
        
        # Extract confidence intervals
        for method, results in method_results.items():
            intervals[method] = {
                'silhouette': results['silhouette_score']['confidence_interval'],
                'composite': results['composite_score']['confidence_interval']
            }
        
        # Check for overlapping intervals
        methods = list(method_results.keys())
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i < j:
                    pair_key = f"{method1}_vs_{method2}"
                    
                    # Check overlap for composite scores
                    ci1 = intervals[method1]['composite']
                    ci2 = intervals[method2]['composite']
                    
                    overlap = not (ci1[1] < ci2[0] or ci2[1] < ci1[0])
                    
                    overlaps[pair_key] = {
                        'overlap': overlap,
                        'ci1': ci1,
                        'ci2': ci2,
                        'gap': min(abs(ci1[1] - ci2[0]), abs(ci2[1] - ci1[0])) if not overlap else 0
                    }
        
        return {
            'intervals': intervals,
            'overlaps': overlaps,
            'non_overlapping_pairs': [k for k, v in overlaps.items() if not v['overlap']]
        } 