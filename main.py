"""
Main execution script for multimodal clustering experiments
"""

import os
import sys
import time
from datetime import datetime

# Add src to path
sys.path.append('src')

from utils import load_config, set_random_seeds, setup_logging, save_json, create_timestamp
from data_loader import MultimodalDataset
from feature_extractor import FeatureExtractor
from clustering import AdaptiveClusteringFramework
from evaluation import ClusteringEvaluator, ComplementarityAnalyzer
from statistical_validation import StatisticalValidator
from visualization import ResultAnalyzer

def main():
    """Main experiment execution"""
    
    # Setup
    print("=" * 80)
    print("MULTIMODAL FEATURE REPRESENTATION CLUSTERING ANALYSIS")
    print("=" * 80)
    
    start_time = time.time()
    timestamp = create_timestamp()
    
    # Load configuration
    config = load_config()
    print(f"Configuration loaded: {config}")
    
    # Create timestamped output directory
    base_output_dir = config['output']['save_dir']
    timestamped_output_dir = os.path.join(base_output_dir, f"experiment_{timestamp}")
    os.makedirs(timestamped_output_dir, exist_ok=True)
    
    # Update config to use timestamped directory
    config['output']['save_dir'] = timestamped_output_dir
    
    # Setup logging
    log_file = os.path.join(timestamped_output_dir, f"experiment_log_{timestamp}.txt")
    logger = setup_logging(log_file)
    logger.info("Starting multimodal clustering experiment")
    logger.info(f"Output directory: {timestamped_output_dir}")
    
    # Set random seeds for reproducibility
    set_random_seeds(config['experiment']['random_seed'])
    logger.info(f"Random seed set to: {config['experiment']['random_seed']}")
    
    # Initialize components
    print("\n1. Initializing components...")
    
    # Data loader
    dataset = MultimodalDataset(
        source=config['dataset']['source'],
        size=config['dataset']['n_samples'],
        config=config['dataset']
    )
    
    # Feature extractor
    feature_extractor = FeatureExtractor(config['features'])
    
    # Clustering framework
    clustering_framework = AdaptiveClusteringFramework(config['clustering'])
    
    # Evaluators
    evaluator = ClusteringEvaluator()
    complementarity_analyzer = ComplementarityAnalyzer()
    statistical_validator = StatisticalValidator(
        alpha=config['experiment']['alpha'],
        confidence_level=config['experiment']['confidence_level']
    )
    
    # Result analyzer
    result_analyzer = ResultAnalyzer(timestamped_output_dir)
    
    print("✓ All components initialized successfully")
    
    # Load and prepare data
    print("\n2. Loading and preparing data...")
    
    data = dataset.load_data()
    data_stats = dataset.get_data_statistics()
    logger.info(f"Dataset statistics: {data_stats}")
    print(f"✓ Loaded {len(data)} samples across {len(data_stats['categories'])} categories")
    
    # Split data
    data_splits = dataset.get_balanced_split({
        'train': config['dataset']['train_ratio'],
        'test': config['dataset']['test_ratio']
    })
    
    train_data = data_splits['train']
    test_data = data_splits['test']
    
    print(f"✓ Data split: {len(train_data)} train, {len(test_data)} test samples")
    
    # Fit feature extractors
    print("\n3. Fitting feature extractors...")
    feature_extractor.fit(train_data)
    print("✓ Feature extractors fitted successfully")
    
    # Single run experiment (for initial analysis)
    print("\n4. Running single experiment...")
    
    # Extract features for all methods
    all_features = feature_extractor.get_all_features(test_data)
    print(f"✓ Features extracted for {len(all_features)} methods")
    
    # Run clustering for each method
    single_run_results = {}
    for method, features in all_features.items():
        print(f"  Running clustering for {method}...")
        results = clustering_framework.run_comprehensive_clustering(features, method)
        single_run_results[method] = results
    
    print("✓ Single run clustering completed")
    
    # Bootstrap experiments
    print(f"\n5. Running bootstrap experiments ({config['experiment']['n_bootstrap_runs']} runs)...")
    
    bootstrap_results = statistical_validator.bootstrap_experiment(
        data=test_data,
        feature_extractor=feature_extractor,
        clustering_framework=clustering_framework,
        n_runs=config['experiment']['n_bootstrap_runs']
    )
    
    print("✓ Bootstrap experiments completed")
    
    # Statistical analysis
    print("\n6. Performing statistical analysis...")
    
    # Significance tests
    statistical_results = statistical_validator.paired_significance_test(bootstrap_results)
    
    # Effect size analysis
    effect_size_results = statistical_validator.effect_size_analysis(bootstrap_results)
    
    # Confidence intervals
    confidence_results = statistical_validator.confidence_intervals(bootstrap_results)
    
    print("✓ Statistical analysis completed")
    
    # Complementarity analysis
    print("\n7. Analyzing method complementarity...")
    
    complementarity_results = complementarity_analyzer.analyze_method_complementarity(single_run_results)
    insights = complementarity_analyzer.generate_insights(complementarity_results, single_run_results)
    
    print("✓ Complementarity analysis completed")
    print("\nKey Insights:")
    for insight in insights:
        print(f"  • {insight}")
    
    # Create visualizations
    print("\n8. Creating visualizations...")
    
    # Performance comparison
    perf_comparison_path = result_analyzer.create_performance_comparison(
        bootstrap_results, statistical_results
    )
    
    # t-SNE visualization
    tsne_path = result_analyzer.create_tsne_visualization(
        all_features, single_run_results
    )
    
    # Complementarity heatmap
    heatmap_path = result_analyzer.create_complementarity_heatmap(
        complementarity_results
    )
    
    # Statistical summary
    stats_summary_path = result_analyzer.create_statistical_summary(
        bootstrap_results, statistical_results
    )
    
    print("✓ All visualizations created")
    
    # Export numerical results
    print("\n9. Exporting numerical results...")
    
    # Export CSV files and detailed report
    exported_files = result_analyzer.export_numerical_results(
        bootstrap_results, statistical_results, complementarity_results, timestamp
    )
    
    # Generate detailed text report
    detailed_report_path = result_analyzer.generate_detailed_report(
        bootstrap_results, statistical_results, complementarity_results, timestamp
    )
    
    print("✓ Numerical results exported:")
    for file_type, file_path in exported_files.items():
        print(f"  • {file_type}: {os.path.basename(file_path)}")
    print(f"  • detailed_report: {os.path.basename(detailed_report_path)}")
    
    # Generate practical guidelines
    print("\n10. Generating practical guidelines...")
    
    guidelines_path = result_analyzer.generate_practical_guidelines(
        bootstrap_results, statistical_results, complementarity_results
    )
    
    print("✓ Practical guidelines generated")
    
    # Save comprehensive results
    print("\n11. Saving comprehensive results...")
    
    # Compile all results
    comprehensive_results = {
        'metadata': {
            'timestamp': timestamp,
            'config': config,
            'dataset_stats': data_stats,
            'execution_time_seconds': time.time() - start_time
        },
        'single_run_results': single_run_results,
        'bootstrap_results': bootstrap_results,
        'statistical_results': statistical_results,
        'effect_size_results': effect_size_results,
        'confidence_results': confidence_results,
        'complementarity_results': complementarity_results,
        'insights': insights,
        'file_paths': {
            'performance_comparison': perf_comparison_path,
            'tsne_visualization': tsne_path,
            'complementarity_heatmap': heatmap_path,
            'statistical_summary': stats_summary_path,
            'practical_guidelines': guidelines_path,
            'detailed_report': detailed_report_path,
            'experiment_log': log_file,
            **exported_files  # Add all CSV files
        }
    }
    
    # Save main results file
    results_path = os.path.join(timestamped_output_dir, f"experiment_results_{timestamp}.json")
    save_json(comprehensive_results, results_path)
    
    # Save statistical analysis separately
    stats_path = os.path.join(timestamped_output_dir, f"statistical_analysis_{timestamp}.json")
    save_json({
        'bootstrap_results': bootstrap_results,
        'statistical_results': statistical_results,
        'effect_size_results': effect_size_results,
        'confidence_results': confidence_results
    }, stats_path)
    
    print(f"✓ Results saved to: {results_path}")
    print(f"✓ Statistical analysis saved to: {stats_path}")
    
    # Final summary
    execution_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.1f} minutes)")
    print(f"Results saved in: {timestamped_output_dir}")
    
    # Print key findings
    best_method = complementarity_results['best_method']
    best_score = bootstrap_results[best_method]['composite_score']['mean']
    
    print(f"\nKEY FINDINGS:")
    print(f"• Best performing method: {best_method.replace('_', ' ').title()} (score: {best_score:.3f})")
    print(f"• Most complementary methods: {complementarity_results['most_complementary_pair'].replace('_vs_', ' and ').replace('_', ' ').title()}")
    print(f"• Significant differences found: {len(statistical_results['significant_differences'])}")
    print(f"• Practical guidelines available in: {guidelines_path}")
    
    # Print detailed numerical summary to console
    print(f"\nDETAILED PERFORMANCE SUMMARY:")
    print("-" * 60)
    method_rankings = statistical_results['method_rankings']
    for rank, (method, score) in enumerate(method_rankings, 1):
        method_results = bootstrap_results[method]
        ci = method_results['composite_score']['confidence_interval']
        std = method_results['composite_score']['std']
        silhouette = method_results['silhouette_score']['mean']
        
        print(f"{rank}. {method.replace('_', ' ').title()}")
        print(f"   Score: {score:.4f} ± {std:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
        print(f"   Silhouette: {silhouette:.4f}, Optimal K: {method_results['optimal_k']['mode']}")
    
    # Print statistical significance summary
    if statistical_results['significant_differences']:
        print(f"\nSTATISTICAL SIGNIFICANCE:")
        print("-" * 60)
        for diff in statistical_results['significant_differences']:
            method1, method2 = diff['methods']
            pair_key = f"{method1}_vs_{method2}"
            test_info = statistical_results['pairwise_tests'][pair_key]
            
            print(f"• {method1.replace('_', ' ').title()} vs {method2.replace('_', ' ').title()}")
            print(f"  P-value: {test_info['p_value']:.6f}, Effect size: {test_info['effect_size']:.4f} ({test_info['effect_size_interpretation']})")
            print(f"  Better: {test_info['better_method'].replace('_', ' ').title()}")
    else:
        print(f"\nNo statistically significant differences found after multiple comparison correction.")
    
    # Print file summary
    print(f"\nOUTPUT FILES GENERATED:")
    print("-" * 60)
    print(f"📊 Numerical Results:")
    for file_type, file_path in exported_files.items():
        print(f"   • {file_type.replace('_', ' ').title()}: {os.path.basename(file_path)}")
    print(f"   • Detailed Report: {os.path.basename(detailed_report_path)}")
    
    print(f"\n📈 Visualizations:")
    print(f"   • Performance Comparison: {os.path.basename(perf_comparison_path)}")
    print(f"   • t-SNE Visualization: {os.path.basename(tsne_path)}")
    print(f"   • Complementarity Heatmap: {os.path.basename(heatmap_path)}")
    print(f"   • Statistical Summary: {os.path.basename(stats_summary_path)}")
    
    print(f"\n📋 Documentation:")
    print(f"   • Practical Guidelines: {os.path.basename(guidelines_path)}")
    print(f"   • Experiment Log: {os.path.basename(log_file)}")
    print(f"   • Complete Results (JSON): {os.path.basename(results_path)}")
    
    logger.info("Experiment completed successfully")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 