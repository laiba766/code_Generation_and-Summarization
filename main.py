"""
Main Pipeline Script
Orchestrates the entire Phase 3 workflow
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append('src')

from data_collector import DataCollector
from feature_extractor import FeatureExtractor
from clustering_models import ClusteringPipeline
from prediction_models import RandomForestPipeline, LSTMPipeline
from visualization import ClusteringVisualizer, MetricsVisualizer
import pandas as pd
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path='config/config.yaml'):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def collect_data(config, source_dir):
    """Step 1: Collect code snippets"""
    logger.info("=" * 80)
    logger.info("STEP 1: DATA COLLECTION")
    logger.info("=" * 80)

    collector = DataCollector(config)
    functions = collector.collect_from_directory(source_dir)
    collector.save_dataset(functions, 'data/raw')

    logger.info("Data collection completed successfully")
    return functions


def extract_features(config):
    """Step 2: Extract features from code snippets"""
    logger.info("=" * 80)
    logger.info("STEP 2: FEATURE EXTRACTION")
    logger.info("=" * 80)

    extractor = FeatureExtractor()

    # Process each language
    all_dfs = []
    for language in config['data']['languages']:
        lang_lower = language.lower()
        input_file = f'data/raw/{lang_lower}_functions.jsonl'

        if not os.path.exists(input_file):
            logger.warning(f"File not found: {input_file}, skipping {language}")
            continue

        output_file = f'data/processed/{lang_lower}_features.csv'
        logger.info(f"Processing {language}...")

        df = extractor.process_dataset(input_file, output_file)
        all_dfs.append(df)

    # Combine all datasets
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv('data/processed/all_features.csv', index=False)
        logger.info(f"Combined dataset shape: {combined_df.shape}")
    else:
        logger.error("No data to combine!")
        return None

    logger.info("Feature extraction completed successfully")
    return combined_df


def run_clustering(config):
    """Step 3: Run clustering experiments"""
    logger.info("=" * 80)
    logger.info("STEP 3: CLUSTERING EXPERIMENTS")
    logger.info("=" * 80)

    # Load features
    df = pd.read_csv('data/processed/all_features.csv')
    logger.info(f"Loaded dataset: {df.shape}")

    # Initialize pipeline
    pipeline = ClusteringPipeline(config)
    visualizer = ClusteringVisualizer(config)

    # Prepare data
    X_scaled, feature_df = pipeline.prepare_data(df)
    
    # Adjust PCA components based on available samples
    n_samples = X_scaled.shape[0]
    n_components = min(50, n_samples - 1)  # PCA components must be less than n_samples
    X_pca = pipeline.apply_pca(X_scaled, n_components=n_components)

    # Run all clustering algorithms
    results = pipeline.run_all_clustering(X_pca)

    # Save results
    pipeline.save_results(results, 'results/metrics/clustering_results.json')

    # Find best method
    best_method = max(
        [(name, res) for name, res in results.items() if res],
        key=lambda x: x[1]['best_score']
    )
    logger.info(f"Best clustering method: {best_method[0].upper()}")
    logger.info(f"Best silhouette score: {best_method[1]['best_score']:.4f}")

    # Visualizations
    logger.info("Generating visualizations...")

    best_labels = best_method[1]['best_labels']
    languages = df['language'].values

    visualizer.plot_tsne(X_pca, best_labels, languages)
    visualizer.plot_pca(X_pca, best_labels, languages)
    visualizer.plot_cluster_distribution(best_labels, languages)
    visualizer.plot_silhouette_comparison(results)

    # Plot elbow curve for K-Means
    if 'kmeans' in results and results['kmeans']:
        visualizer.plot_elbow_curve(results['kmeans']['all_results'])

    logger.info("Clustering experiments completed successfully")
    return results


def run_prediction(config):
    """Step 4: Run prediction experiments"""
    logger.info("=" * 80)
    logger.info("STEP 4: PREDICTION EXPERIMENTS")
    logger.info("=" * 80)

    # Load features
    df = pd.read_csv('data/processed/all_features.csv')

    # Select complexity features
    complexity_features = [
        'ast_node_count', 'ast_depth', 'ast_leaf_count', 'ast_branching_factor_avg',
        'ast_distinct_node_types', 'cc_mccabe', 'halstead_volume', 'halstead_difficulty',
        'halstead_effort', 'num_if', 'num_for', 'num_while', 'loc'
    ]

    # Filter features that exist in the dataframe
    complexity_features = [f for f in complexity_features if f in df.columns]
    logger.info(f"Using {len(complexity_features)} complexity features")

    target_col = 'language'

    # Random Forest
    logger.info("Training Random Forest...")
    rf_pipeline = RandomForestPipeline(config, task='classification')
    X_train, X_test, y_train, y_test = rf_pipeline.prepare_data(
        df, target_col, feature_cols=complexity_features
    )

    rf_train_results = rf_pipeline.train(X_train, y_train)
    rf_test_results = rf_pipeline.evaluate(X_test, y_test)

    logger.info(f"Random Forest Test Accuracy: {rf_test_results['accuracy']:.4f}")
    logger.info(f"Random Forest Test F1-Score: {rf_test_results['f1_score']:.4f}")

    # LSTM
    logger.info("Training LSTM...")
    lstm_pipeline = LSTMPipeline(config)
    X_train, X_test, y_train, y_test, input_size, num_classes = lstm_pipeline.prepare_data(
        df, target_col, feature_cols=complexity_features
    )

    lstm_history = lstm_pipeline.train(X_train, y_train, input_size, num_classes)
    lstm_results = lstm_pipeline.evaluate(X_test, y_test)

    logger.info(f"LSTM Test Accuracy: {lstm_results['accuracy']:.4f}")
    logger.info(f"LSTM Test F1-Score: {lstm_results['f1_score']:.4f}")

    # Save results
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        """Convert numpy types to native Python types"""
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    all_results = {
        'random_forest': {
            'train': convert_to_native(rf_train_results),
            'test': {k: convert_to_native(v) for k, v in rf_test_results.items()
                     if k != 'classification_report'}
        },
        'lstm': {
            'history': {k: [float(x) for x in v] for k, v in lstm_history.items()},
            'test': convert_to_native(lstm_results)
        }
    }

    with open('results/metrics/prediction_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Model comparison
    comparison_df = pd.DataFrame({
        'Model': ['Random Forest', 'LSTM'],
        'Accuracy': [rf_test_results['accuracy'], lstm_results['accuracy']],
        'Precision': [rf_test_results['precision'], lstm_results['precision']],
        'Recall': [rf_test_results['recall'], lstm_results['recall']],
        'F1-Score': [rf_test_results['f1_score'], lstm_results['f1_score']]
    })

    comparison_df.to_csv('results/metrics/model_comparison.csv', index=False)
    logger.info("\nModel Comparison:")
    logger.info(comparison_df.to_string())

    logger.info("Prediction experiments completed successfully")
    return all_results


def generate_summary_report(config):
    """Step 5: Generate summary report"""
    logger.info("=" * 80)
    logger.info("STEP 5: GENERATING SUMMARY REPORT")
    logger.info("=" * 80)

    report = []
    report.append("=" * 80)
    report.append("PHASE 3: MODEL IMPLEMENTATION & EXPERIMENTATION")
    report.append("Code Summarization and Generation Project")
    report.append("=" * 80)
    report.append("")
    report.append("Author: Laiba Akram")
    report.append("Student ID: 42943")
    report.append("Course: BSCS-7A - Theory of Programming Languages")
    report.append("")

    # Clustering results
    if os.path.exists('results/metrics/clustering_results.json'):
        with open('results/metrics/clustering_results.json', 'r') as f:
            clustering_results = json.load(f)

        report.append("CLUSTERING RESULTS:")
        report.append("-" * 40)
        for method, results in clustering_results.items():
            if results:
                report.append(f"\n{method.upper()}:")
                report.append(f"  Best Parameters: {results['best_params']}")
                report.append(f"  Silhouette Score: {results['best_score']:.4f}")

    # Prediction results
    if os.path.exists('results/metrics/model_comparison.csv'):
        comparison_df = pd.read_csv('results/metrics/model_comparison.csv')

        report.append("\n\nPREDICTION RESULTS:")
        report.append("-" * 40)
        report.append(comparison_df.to_string(index=False))

    report.append("\n\n" + "=" * 80)
    report.append("HYPOTHESIS TESTING:")
    report.append("-" * 40)
    report.append("H1: Statically typed, memory-safe languages exhibit higher")
    report.append("    syntax complexity than dynamically typed languages.")
    report.append("")
    report.append("See clustering visualizations and detailed analysis in notebooks.")
    report.append("=" * 80)

    # Save report
    report_text = "\n".join(report)
    with open('results/PHASE3_SUMMARY_REPORT.txt', 'w') as f:
        f.write(report_text)

    logger.info("\n" + report_text)
    logger.info("\nSummary report saved to: results/PHASE3_SUMMARY_REPORT.txt")


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(
        description='Phase 3: Model Implementation & Experimentation'
    )
    parser.add_argument(
        '--step',
        type=str,
        choices=['collect', 'extract', 'cluster', 'predict', 'all'],
        default='all',
        help='Which step to run (default: all)'
    )
    parser.add_argument(
        '--source-dir',
        type=str,
        help='Source directory for code collection'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('results/visualizations', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    logger.info("Starting Phase 3 Pipeline...")
    logger.info(f"Step: {args.step}")

    try:
        if args.step in ['collect', 'all']:
            if args.source_dir:
                collect_data(config, args.source_dir)
            else:
                logger.warning("Skipping data collection (no --source-dir provided)")

        if args.step in ['extract', 'all']:
            extract_features(config)

        if args.step in ['cluster', 'all']:
            run_clustering(config)

        if args.step in ['predict', 'all']:
            run_prediction(config)

        if args.step == 'all':
            generate_summary_report(config)

        logger.info("=" * 80)
        logger.info("PHASE 3 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info("\nResults saved in:")
        logger.info("  - results/visualizations/")
        logger.info("  - results/metrics/")
        logger.info("\nNext steps:")
        logger.info("  1. Review Jupyter notebooks in notebooks/")
        logger.info("  2. Analyze results in results/PHASE3_SUMMARY_REPORT.txt")
        logger.info("  3. Generate final report for submission")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
