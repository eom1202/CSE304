"""
Visualization and analysis for multimodal clustering experiments
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from typing import Dict, List, Any, Tuple
import os
import pandas as pd
import json

class ResultAnalyzer:
    """Result analysis and visualization"""
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize result analyzer
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def create_performance_comparison(self, bootstrap_results: Dict[str, Dict[str, Any]], 
                                    statistical_results: Dict[str, Any],
                                    save_path: str = None) -> str:
        """
        Create performance comparison visualization
        
        Args:
            bootstrap_results: Bootstrap experiment results
            statistical_results: Statistical test results
            save_path: Path to save the figure
            
        Returns:
            Path to saved figure
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "performance_comparison.png")
        
        # Prepare data
        methods = list(bootstrap_results.keys())
        means = [bootstrap_results[method]['composite_score']['mean'] for method in methods]
        stds = [bootstrap_results[method]['composite_score']['std'] for method in methods]
        cis = [bootstrap_results[method]['composite_score']['confidence_interval'] for method in methods]
        
        # Calculate error bars (confidence intervals)
        ci_lower = [ci[0] for ci in cis]
        ci_upper = [ci[1] for ci in cis]
        errors = [[mean - lower for mean, lower in zip(means, ci_lower)],
                 [upper - mean for mean, upper in zip(means, ci_upper)]]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create bar plot
        bars = ax.bar(methods, means, yerr=errors, capsize=5, 
                     alpha=0.8, edgecolor='black', linewidth=1)
        
        # Color bars based on performance ranking
        colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add significance annotations
        self._add_significance_annotations(ax, methods, means, statistical_results)
        
        # Customize plot
        ax.set_ylabel('Composite Score', fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature Extraction Method', fontsize=12, fontweight='bold')
        ax.set_title('Performance Comparison of Multimodal Feature Extraction Methods\n'
                    '(with 95% Confidence Intervals)', fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + errors[1][i] + 0.01,
                   f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance comparison saved to: {save_path}")
        return save_path
    
    def _add_significance_annotations(self, ax, methods: List[str], means: List[float], 
                                    statistical_results: Dict[str, Any]):
        """Add significance annotations to the plot"""
        if 'significant_differences' not in statistical_results:
            return
        
        significant_pairs = statistical_results['significant_differences']
        
        # Add significance markers
        y_max = max(means) * 1.2
        y_offset = y_max * 0.05
        
        for i, sig_diff in enumerate(significant_pairs[:3]):  # Show top 3 significant differences
            method1, method2 = sig_diff['methods']
            if method1 in methods and method2 in methods:
                idx1, idx2 = methods.index(method1), methods.index(method2)
                
                # Draw significance line
                y_pos = y_max + i * y_offset
                ax.plot([idx1, idx2], [y_pos, y_pos], 'k-', linewidth=1)
                ax.plot([idx1, idx1], [y_pos - y_offset*0.1, y_pos], 'k-', linewidth=1)
                ax.plot([idx2, idx2], [y_pos - y_offset*0.1, y_pos], 'k-', linewidth=1)
                
                # Add significance marker
                p_val = sig_diff['p_value']
                if p_val < 0.001:
                    marker = '***'
                elif p_val < 0.01:
                    marker = '**'
                else:
                    marker = '*'
                
                ax.text((idx1 + idx2) / 2, y_pos + y_offset*0.1, marker, 
                       ha='center', va='bottom', fontweight='bold')
    
    def create_tsne_visualization(self, all_features: Dict[str, np.ndarray], 
                                all_results: Dict[str, Dict[str, Any]],
                                true_labels: List[str] = None,
                                save_path: str = None) -> str:
        """
        Create t-SNE visualization for all feature extraction methods
        
        Args:
            all_features: Dictionary mapping method names to feature matrices
            all_results: Clustering results for all methods
            true_labels: True category labels (optional)
            save_path: Path to save the figure
            
        Returns:
            Path to saved figure
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "tsne_visualization.png")
        
        methods = list(all_features.keys())
        n_methods = len(methods)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, method in enumerate(methods):
            ax = axes[i]
            
            # Get features and labels
            features = all_features[method]
            cluster_labels = all_results[method]['best_labels']
            
            # Perform t-SNE
            print(f"Computing t-SNE for {method}...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
            features_2d = tsne.fit_transform(features)
            
            # Create scatter plot
            scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], 
                               c=cluster_labels, cmap='tab10', alpha=0.7, s=50)
            
            # Customize subplot
            ax.set_title(f'{method.replace("_", " ").title()}\n'
                        f'Score: {all_results[method]["best_score"]:.3f}', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Cluster')
        
        # Remove empty subplot
        if n_methods < 6:
            axes[5].remove()
        
        plt.suptitle('t-SNE Visualization of Different Feature Extraction Methods', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"t-SNE visualization saved to: {save_path}")
        return save_path
    
    def create_complementarity_heatmap(self, complementarity_results: Dict[str, Any],
                                     save_path: str = None) -> str:
        """
        Create complementarity analysis heatmap
        
        Args:
            complementarity_results: Results from complementarity analysis
            save_path: Path to save the figure
            
        Returns:
            Path to saved figure
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "complementarity_heatmap.png")
        
        # Get agreement matrix and method names
        agreement_matrix = complementarity_results['agreement_matrix']
        method_names = complementarity_results['method_names']
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Agreement heatmap
        sns.heatmap(agreement_matrix, 
                   xticklabels=[name.replace('_', ' ').title() for name in method_names],
                   yticklabels=[name.replace('_', ' ').title() for name in method_names],
                   annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   center=0.5, ax=ax1, cbar_kws={'label': 'Agreement (ARI)'})
        ax1.set_title('Method Agreement Matrix\n(Adjusted Rand Index)', 
                     fontsize=12, fontweight='bold')
        
        # Complementarity scores
        comp_scores = complementarity_results['complementarity_scores']
        pairs = list(comp_scores.keys())
        scores = [comp_scores[pair]['complementarity'] for pair in pairs]
        
        # Create complementarity bar plot
        pair_labels = [pair.replace('_vs_', ' vs ').replace('_', ' ').title() for pair in pairs]
        bars = ax2.bar(range(len(pairs)), scores, alpha=0.8)
        
        # Color bars based on score
        colors = plt.cm.viridis(np.array(scores) / max(scores))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax2.set_xlabel('Method Pairs')
        ax2.set_ylabel('Complementarity Score')
        ax2.set_title('Method Complementarity Scores\n(Higher = More Complementary)', 
                     fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(pairs)))
        ax2.set_xticklabels(pair_labels, rotation=45, ha='right')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Complementarity heatmap saved to: {save_path}")
        return save_path
    
    def create_statistical_summary(self, bootstrap_results: Dict[str, Dict[str, Any]],
                                 statistical_results: Dict[str, Any],
                                 save_path: str = None) -> str:
        """
        Create statistical summary visualization
        
        Args:
            bootstrap_results: Bootstrap experiment results
            statistical_results: Statistical test results
            save_path: Path to save the figure
            
        Returns:
            Path to saved figure
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "statistical_summary.png")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Box plot of bootstrap distributions
        methods = list(bootstrap_results.keys())
        data_for_boxplot = []
        labels_for_boxplot = []
        
        for method in methods:
            scores = bootstrap_results[method]['raw_scores']['composite']
            data_for_boxplot.append(scores)
            labels_for_boxplot.append(method.replace('_', ' ').title())
        
        box_plot = ax1.boxplot(data_for_boxplot, labels=labels_for_boxplot, patch_artist=True)
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax1.set_title('Distribution of Bootstrap Scores', fontweight='bold')
        ax1.set_ylabel('Composite Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. Confidence intervals
        means = [bootstrap_results[method]['composite_score']['mean'] for method in methods]
        cis = [bootstrap_results[method]['composite_score']['confidence_interval'] for method in methods]
        
        ci_lower = [ci[0] for ci in cis]
        ci_upper = [ci[1] for ci in cis]
        
        y_pos = np.arange(len(methods))
        ax2.errorbar(means, y_pos, xerr=[np.array(means) - np.array(ci_lower), 
                                        np.array(ci_upper) - np.array(means)], 
                    fmt='o', capsize=5, capthick=2, markersize=8)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels_for_boxplot)
        ax2.set_xlabel('Composite Score')
        ax2.set_title('95% Confidence Intervals', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Effect sizes
        if 'pairwise_tests' in statistical_results:
            effect_sizes = []
            pair_names = []
            
            for pair, test_result in statistical_results['pairwise_tests'].items():
                effect_sizes.append(abs(test_result['effect_size']))
                pair_names.append(pair.replace('_vs_', ' vs ').replace('_', ' '))
            
            bars = ax3.bar(range(len(effect_sizes)), effect_sizes, alpha=0.8)
            
            # Color by effect size magnitude
            colors = ['red' if es >= 0.8 else 'orange' if es >= 0.5 else 'yellow' if es >= 0.2 else 'lightgray' 
                     for es in effect_sizes]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax3.set_xticks(range(len(pair_names)))
            ax3.set_xticklabels(pair_names, rotation=45, ha='right')
            ax3.set_ylabel('|Effect Size| (Cohen\'s d)')
            ax3.set_title('Effect Sizes Between Methods', fontweight='bold')
            
            # Add effect size interpretation lines
            ax3.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small')
            ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium')
            ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large')
            ax3.legend()
        
        # 4. Method rankings
        if 'method_rankings' in statistical_results:
            rankings = statistical_results['method_rankings']
            method_names = [ranking[0].replace('_', ' ').title() for ranking in rankings]
            scores = [ranking[1] for ranking in rankings]
            
            bars = ax4.barh(range(len(method_names)), scores, alpha=0.8)
            
            # Color by rank
            colors = plt.cm.RdYlGn(np.linspace(0.3, 1, len(method_names)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax4.set_yticks(range(len(method_names)))
            ax4.set_yticklabels(method_names)
            ax4.set_xlabel('Mean Composite Score')
            ax4.set_title('Method Rankings', fontweight='bold')
            
            # Add value labels
            for i, (bar, score) in enumerate(zip(bars, scores)):
                ax4.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{score:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Statistical summary saved to: {save_path}")
        return save_path
    
    def generate_practical_guidelines(self, bootstrap_results: Dict[str, Dict[str, Any]],
                                    statistical_results: Dict[str, Any],
                                    complementarity_results: Dict[str, Any],
                                    save_path: str = None) -> str:
        """
        Generate practical guidelines document
        
        Args:
            bootstrap_results: Bootstrap experiment results
            statistical_results: Statistical test results
            complementarity_results: Complementarity analysis results
            save_path: Path to save the guidelines
            
        Returns:
            Path to saved guidelines
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "practical_guidelines.md")
        
        # Get best method
        best_method = complementarity_results['best_method']
        best_score = bootstrap_results[best_method]['composite_score']['mean']
        
        # Get most complementary pair
        most_comp_pair = complementarity_results['most_complementary_pair']
        comp_info = complementarity_results['complementarity_scores'][most_comp_pair]
        
        # Generate guidelines
        guidelines = f"""# Practical Guidelines for Multimodal Clustering

## Executive Summary

Based on comprehensive statistical analysis of {len(bootstrap_results)} feature extraction methods across {bootstrap_results[list(bootstrap_results.keys())[0]]['n_runs']} bootstrap experiments, this document provides evidence-based recommendations for multimodal clustering tasks.

## Key Findings

### Best Overall Method
**{best_method.replace('_', ' ').title()}** achieved the highest performance with a mean composite score of **{best_score:.3f}**.

### Most Complementary Methods
**{most_comp_pair.replace('_vs_', ' and ').replace('_', ' ').title()}** showed the highest complementarity (score: {comp_info['complementarity']:.3f}), suggesting they capture different aspects of the data structure.

## Method-Specific Recommendations

"""
        
        # Add method-specific recommendations
        method_rankings = statistical_results['method_rankings']
        
        for rank, (method, score) in enumerate(method_rankings, 1):
            method_name = method.replace('_', ' ').title()
            ci = bootstrap_results[method]['composite_score']['confidence_interval']
            
            guidelines += f"""### {rank}. {method_name}
- **Performance**: {score:.3f} (95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])
- **Recommended when**: """
            
            if method == 'text_only':
                guidelines += """Text data is rich and well-structured, images are low quality or noisy, computational resources are limited."""
            elif method == 'image_only':
                guidelines += """Visual patterns are more informative than text, text descriptions are sparse or unreliable, domain expertise suggests visual features are key."""
            elif method == 'early_fusion':
                guidelines += """Both modalities are high quality, computational efficiency is important, simple fusion is preferred over complex methods."""
            elif method == 'late_fusion':
                guidelines += """Modalities have different scales or distributions, you want to preserve individual modality characteristics, interpretability is important."""
            elif method == 'attention_fusion':
                guidelines += """Adaptive weighting between modalities is needed, modalities have varying relevance across samples, maximum performance is prioritized over simplicity."""
            
            guidelines += "\n\n"
        
        # Add statistical confidence section
        guidelines += f"""## Statistical Confidence

All recommendations are based on rigorous statistical analysis:

- **Significance Level**: α = {statistical_results.get('correction_info', {}).get('original_alpha', 0.05)}
- **Multiple Comparison Correction**: {statistical_results.get('correction_info', {}).get('method', 'Bonferroni').title()}
- **Confidence Intervals**: 95% using t-distribution
- **Effect Size Threshold**: Medium (Cohen's d ≥ 0.5) for practical significance

### Significant Differences Found

"""
        
        if statistical_results['significant_differences']:
            for diff in statistical_results['significant_differences']:
                method1, method2 = diff['methods']
                guidelines += f"- **{method1.replace('_', ' ').title()}** vs **{method2.replace('_', ' ').title()}**: "
                guidelines += f"p = {diff['p_value']:.4f}, effect size = {diff['effect_size']:.3f} ({statistical_results['pairwise_tests'][f'{method1}_vs_{method2}']['effect_size_interpretation']})\n"
        else:
            guidelines += "No statistically significant differences found after multiple comparison correction.\n"
        
        # Add decision tree
        guidelines += f"""
## Decision Framework

```
Is computational efficiency critical?
├─ YES → Use Text-Only or Image-Only (whichever domain is richer)
└─ NO → Continue to next question

Are both text and images high quality?
├─ YES → Continue to next question
└─ NO → Use the higher quality modality only

Do you need interpretable feature contributions?
├─ YES → Use Late Fusion
└─ NO → Use {best_method.replace('_', ' ').title()} (best overall performance)

Is adaptive weighting between modalities important?
├─ YES → Use Attention Fusion
└─ NO → Use Early Fusion (simpler, often sufficient)
```

## Implementation Notes

1. **Data Quality Assessment**: Always evaluate the quality of both text and image modalities before method selection.

2. **Computational Constraints**: Consider your computational budget. Fusion methods require processing both modalities.

3. **Domain Expertise**: Leverage domain knowledge about which modality typically contains more discriminative information.

4. **Validation Strategy**: Use cross-validation or bootstrap sampling to validate method choice on your specific dataset.

5. **Hyperparameter Tuning**: All methods benefit from proper hyperparameter optimization, especially clustering algorithm selection and number of clusters.

## Confidence and Limitations

- These guidelines are based on synthetic data experiments and may not generalize to all real-world scenarios.
- Performance differences may vary significantly based on dataset characteristics.
- Always validate method choice on your specific use case with appropriate evaluation metrics.

---
*Generated from multimodal clustering analysis with {bootstrap_results[list(bootstrap_results.keys())[0]]['n_runs']} bootstrap experiments*
"""
        
        # Save guidelines
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(guidelines)
        
        print(f"Practical guidelines saved to: {save_path}")
        return save_path
    
    def export_numerical_results(self, bootstrap_results: Dict[str, Dict[str, Any]],
                                statistical_results: Dict[str, Any],
                                complementarity_results: Dict[str, Any],
                                timestamp: str) -> Dict[str, str]:
        """
        Export numerical results as CSV and text files
        
        Args:
            bootstrap_results: Bootstrap experiment results
            statistical_results: Statistical test results
            complementarity_results: Complementarity analysis results
            timestamp: Timestamp for file naming
            
        Returns:
            Dictionary of exported file paths
        """
        exported_files = {}
        
        # 1. Performance summary CSV
        performance_csv_path = os.path.join(self.output_dir, f"performance_summary_{timestamp}.csv")
        performance_data = []
        
        for method, results in bootstrap_results.items():
            performance_data.append({
                'Method': method.replace('_', ' ').title(),
                'Mean_Score': results['composite_score']['mean'],
                'Std_Score': results['composite_score']['std'],
                'Median_Score': results['composite_score']['median'],
                'CI_Lower': results['composite_score']['confidence_interval'][0],
                'CI_Upper': results['composite_score']['confidence_interval'][1],
                'Mean_Silhouette': results['silhouette_score']['mean'],
                'Std_Silhouette': results['silhouette_score']['std'],
                'Optimal_K_Mean': results['optimal_k']['mean'],
                'Optimal_K_Mode': results['optimal_k']['mode'],
                'N_Runs': results['n_runs']
            })
        
        performance_df = pd.DataFrame(performance_data)
        performance_df = performance_df.sort_values('Mean_Score', ascending=False)
        performance_df.to_csv(performance_csv_path, index=False)
        exported_files['performance_summary'] = performance_csv_path
        
        # 2. Statistical tests CSV
        if 'pairwise_tests' in statistical_results:
            stats_csv_path = os.path.join(self.output_dir, f"statistical_tests_{timestamp}.csv")
            stats_data = []
            
            for pair, test_result in statistical_results['pairwise_tests'].items():
                method1, method2 = pair.split('_vs_')
                stats_data.append({
                    'Method_1': method1.replace('_', ' ').title(),
                    'Method_2': method2.replace('_', ' ').title(),
                    'Test_Type': test_result['test_type'],
                    'P_Value': test_result['p_value'],
                    'Corrected_P_Value': test_result.get('corrected_p_value', 'N/A'),
                    'Significant': test_result['significant'],
                    'Significant_Corrected': test_result.get('significant_corrected', 'N/A'),
                    'Effect_Size': test_result['effect_size'],
                    'Effect_Size_Interpretation': test_result['effect_size_interpretation'],
                    'Better_Method': test_result['better_method'].replace('_', ' ').title(),
                    'Mean_Difference': test_result['mean_difference']
                })
            
            stats_df = pd.DataFrame(stats_data)
            stats_df = stats_df.sort_values('Effect_Size', key=abs, ascending=False)
            stats_df.to_csv(stats_csv_path, index=False)
            exported_files['statistical_tests'] = stats_csv_path
        
        # 3. Raw bootstrap scores CSV
        raw_scores_csv_path = os.path.join(self.output_dir, f"raw_bootstrap_scores_{timestamp}.csv")
        raw_scores_data = {}
        
        for method, results in bootstrap_results.items():
            method_name = method.replace('_', ' ').title()
            raw_scores_data[f'{method_name}_Composite'] = results['raw_scores']['composite']
            raw_scores_data[f'{method_name}_Silhouette'] = results['raw_scores']['silhouette']
        
        # Pad shorter lists with NaN
        max_length = max(len(scores) for scores in raw_scores_data.values())
        for key, scores in raw_scores_data.items():
            if len(scores) < max_length:
                raw_scores_data[key] = scores + [np.nan] * (max_length - len(scores))
        
        raw_scores_df = pd.DataFrame(raw_scores_data)
        raw_scores_df.to_csv(raw_scores_csv_path, index=False)
        exported_files['raw_bootstrap_scores'] = raw_scores_csv_path
        
        # 4. Complementarity analysis CSV
        comp_csv_path = os.path.join(self.output_dir, f"complementarity_analysis_{timestamp}.csv")
        comp_data = []
        
        for pair, comp_info in complementarity_results['complementarity_scores'].items():
            method1, method2 = pair.split('_vs_')
            comp_data.append({
                'Method_1': method1.replace('_', ' ').title(),
                'Method_2': method2.replace('_', ' ').title(),
                'Agreement_ARI': comp_info['agreement'],
                'Avg_Performance': comp_info['avg_performance'],
                'Complementarity_Score': comp_info['complementarity']
            })
        
        comp_df = pd.DataFrame(comp_data)
        comp_df = comp_df.sort_values('Complementarity_Score', ascending=False)
        comp_df.to_csv(comp_csv_path, index=False)
        exported_files['complementarity_analysis'] = comp_csv_path
        
        return exported_files
    
    def generate_detailed_report(self, bootstrap_results: Dict[str, Dict[str, Any]],
                               statistical_results: Dict[str, Any],
                               complementarity_results: Dict[str, Any],
                               timestamp: str) -> str:
        """
        Generate detailed numerical report as text file
        
        Args:
            bootstrap_results: Bootstrap experiment results
            statistical_results: Statistical test results
            complementarity_results: Complementarity analysis results
            timestamp: Timestamp for file naming
            
        Returns:
            Path to generated report
        """
        report_path = os.path.join(self.output_dir, f"detailed_report_{timestamp}.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("MULTIMODAL CLUSTERING EXPERIMENT - DETAILED NUMERICAL REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # 1. Executive Summary
            f.write("1. EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            best_method = complementarity_results['best_method']
            best_score = bootstrap_results[best_method]['composite_score']['mean']
            n_runs = bootstrap_results[best_method]['n_runs']
            
            f.write(f"Best performing method: {best_method.replace('_', ' ').title()}\n")
            f.write(f"Best composite score: {best_score:.4f}\n")
            f.write(f"Number of bootstrap runs: {n_runs}\n")
            f.write(f"Total methods compared: {len(bootstrap_results)}\n\n")
            
            # 2. Performance Rankings
            f.write("2. PERFORMANCE RANKINGS\n")
            f.write("-" * 40 + "\n")
            
            method_rankings = statistical_results['method_rankings']
            for rank, (method, score) in enumerate(method_rankings, 1):
                method_results = bootstrap_results[method]
                ci = method_results['composite_score']['confidence_interval']
                std = method_results['composite_score']['std']
                
                f.write(f"{rank}. {method.replace('_', ' ').title()}\n")
                f.write(f"   Mean Score: {score:.4f} ± {std:.4f}\n")
                f.write(f"   95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]\n")
                f.write(f"   Silhouette: {method_results['silhouette_score']['mean']:.4f}\n")
                f.write(f"   Optimal K: {method_results['optimal_k']['mode']}\n\n")
            
            # 3. Statistical Significance
            f.write("3. STATISTICAL SIGNIFICANCE TESTS\n")
            f.write("-" * 40 + "\n")
            
            if statistical_results['significant_differences']:
                f.write("Significant differences found:\n\n")
                for diff in statistical_results['significant_differences']:
                    method1, method2 = diff['methods']
                    pair_key = f"{method1}_vs_{method2}"
                    test_info = statistical_results['pairwise_tests'][pair_key]
                    
                    f.write(f"{method1.replace('_', ' ').title()} vs {method2.replace('_', ' ').title()}:\n")
                    f.write(f"   Test: {test_info['test_type']}\n")
                    f.write(f"   P-value: {test_info['p_value']:.6f}\n")
                    f.write(f"   Corrected P-value: {test_info.get('corrected_p_value', 'N/A'):.6f}\n")
                    f.write(f"   Effect size: {test_info['effect_size']:.4f} ({test_info['effect_size_interpretation']})\n")
                    f.write(f"   Better method: {test_info['better_method'].replace('_', ' ').title()}\n\n")
            else:
                f.write("No statistically significant differences found after correction.\n\n")
            
            # 4. Complementarity Analysis
            f.write("4. COMPLEMENTARITY ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            most_comp_pair = complementarity_results['most_complementary_pair']
            comp_info = complementarity_results['complementarity_scores'][most_comp_pair]
            
            f.write(f"Most complementary pair: {most_comp_pair.replace('_vs_', ' and ').replace('_', ' ').title()}\n")
            f.write(f"Complementarity score: {comp_info['complementarity']:.4f}\n")
            f.write(f"Agreement (ARI): {comp_info['agreement']:.4f}\n")
            f.write(f"Average performance: {comp_info['avg_performance']:.4f}\n\n")
            
            f.write("All method pairs (sorted by complementarity):\n")
            sorted_pairs = sorted(complementarity_results['complementarity_scores'].items(),
                                key=lambda x: x[1]['complementarity'], reverse=True)
            
            for pair, info in sorted_pairs:
                f.write(f"   {pair.replace('_vs_', ' vs ').replace('_', ' ')}: {info['complementarity']:.4f}\n")
            
            # 5. Raw Data Summary
            f.write(f"\n5. RAW DATA SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            for method, results in bootstrap_results.items():
                f.write(f"{method.replace('_', ' ').title()}:\n")
                composite_scores = results['raw_scores']['composite']
                silhouette_scores = results['raw_scores']['silhouette']
                
                f.write(f"   Composite scores: {composite_scores}\n")
                f.write(f"   Silhouette scores: {silhouette_scores}\n")
                f.write(f"   Min composite: {min(composite_scores):.4f}\n")
                f.write(f"   Max composite: {max(composite_scores):.4f}\n")
                f.write(f"   Range: {max(composite_scores) - min(composite_scores):.4f}\n\n")
        
        return report_path 