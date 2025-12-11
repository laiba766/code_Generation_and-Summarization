"""
Clustering Models Module
Implements K-Means, DBSCAN, and Hierarchical Clustering
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Dict, List, Tuple, Any
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusteringPipeline:
    """Pipeline for clustering experiments"""

    def __init__(self, config: Dict):
        self.config = config
        self.scaler = StandardScaler()
        self.pca = None
        self.models = {}
        self.results = {}

    def prepare_data(self, df: pd.DataFrame, feature_cols: List[str] = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """Prepare data for clustering"""
        logger.info("Preparing data for clustering")

        # Select numeric features
        if feature_cols is None:
            # Auto-select numeric columns, exclude metadata
            exclude_cols = ['function_id', 'function_name', 'file_path', 'repo_name',
                            'language', 'start_line', 'end_line', 'code', 'operators', 'operands']
            feature_cols = [col for col in df.columns
                            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

        logger.info(f"Using {len(feature_cols)} features for clustering")

        # Handle missing values
        X = df[feature_cols].fillna(0)

        # Normalize features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, df[feature_cols]

    def apply_pca(self, X: np.ndarray, n_components: int = None) -> np.ndarray:
        """Apply PCA for dimensionality reduction"""
        if n_components is None:
            # Explained variance threshold
            self.pca = PCA(n_components=0.95)
        else:
            self.pca = PCA(n_components=n_components)

        X_pca = self.pca.fit_transform(X)
        logger.info(f"PCA: reduced from {X.shape[1]} to {X_pca.shape[1]} dimensions")
        logger.info(f"Explained variance: {self.pca.explained_variance_ratio_.sum():.4f}")

        return X_pca

    def kmeans_clustering(self, X: np.ndarray, param_grid: Dict = None) -> Dict[str, Any]:
        """K-Means clustering with hyperparameter tuning"""
        logger.info("Running K-Means clustering")

        if param_grid is None:
            param_grid = self.config['models']['clustering']['kmeans']

        best_score = -1
        best_params = None
        best_labels = None
        results = []

        for n_clusters in param_grid['n_clusters']:
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=param_grid['random_state'],
                max_iter=param_grid['max_iter']
            )

            labels = kmeans.fit_predict(X)

            # Calculate metrics
            silhouette = silhouette_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            calinski_harabasz = calinski_harabasz_score(X, labels)

            results.append({
                'n_clusters': n_clusters,
                'silhouette_score': silhouette,
                'davies_bouldin_score': davies_bouldin,
                'calinski_harabasz_score': calinski_harabasz,
                'inertia': kmeans.inertia_
            })

            logger.info(f"  n_clusters={n_clusters}: silhouette={silhouette:.4f}, DB={davies_bouldin:.4f}")

            if silhouette > best_score:
                best_score = silhouette
                best_params = {'n_clusters': n_clusters}
                best_labels = labels

        return {
            'method': 'KMeans',
            'best_params': best_params,
            'best_score': best_score,
            'best_labels': best_labels,
            'all_results': results
        }

    def dbscan_clustering(self, X: np.ndarray, param_grid: Dict = None) -> Dict[str, Any]:
        """DBSCAN clustering with hyperparameter tuning"""
        logger.info("Running DBSCAN clustering")

        if param_grid is None:
            param_grid = self.config['models']['clustering']['dbscan']

        best_score = -1
        best_params = None
        best_labels = None
        results = []

        for eps in param_grid['eps']:
            for min_samples in param_grid['min_samples']:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X)

                # Check if valid clustering (more than 1 cluster, not all noise)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)

                if n_clusters < 2 or n_noise > len(labels) * 0.5:
                    logger.info(f"  eps={eps}, min_samples={min_samples}: Invalid clustering (clusters={n_clusters}, noise={n_noise})")
                    continue

                # Calculate metrics (excluding noise points)
                valid_indices = labels != -1
                if sum(valid_indices) > 0:
                    silhouette = silhouette_score(X[valid_indices], labels[valid_indices])

                    results.append({
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'silhouette_score': silhouette
                    })

                    logger.info(f"  eps={eps}, min_samples={min_samples}: silhouette={silhouette:.4f}, clusters={n_clusters}")

                    if silhouette > best_score:
                        best_score = silhouette
                        best_params = {'eps': eps, 'min_samples': min_samples}
                        best_labels = labels

        if best_labels is None:
            logger.warning("No valid DBSCAN clustering found")
            return None

        return {
            'method': 'DBSCAN',
            'best_params': best_params,
            'best_score': best_score,
            'best_labels': best_labels,
            'all_results': results
        }

    def hierarchical_clustering(self, X: np.ndarray, param_grid: Dict = None) -> Dict[str, Any]:
        """Hierarchical clustering with hyperparameter tuning"""
        logger.info("Running Hierarchical clustering")

        if param_grid is None:
            param_grid = self.config['models']['clustering']['hierarchical']

        best_score = -1
        best_params = None
        best_labels = None
        results = []

        for n_clusters in param_grid['n_clusters']:
            for linkage in param_grid['linkage']:
                hierarchical = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=linkage
                )

                labels = hierarchical.fit_predict(X)

                # Calculate metrics
                silhouette = silhouette_score(X, labels)
                davies_bouldin = davies_bouldin_score(X, labels)
                calinski_harabasz = calinski_harabasz_score(X, labels)

                results.append({
                    'n_clusters': n_clusters,
                    'linkage': linkage,
                    'silhouette_score': silhouette,
                    'davies_bouldin_score': davies_bouldin,
                    'calinski_harabasz_score': calinski_harabasz
                })

                logger.info(f"  n_clusters={n_clusters}, linkage={linkage}: silhouette={silhouette:.4f}")

                if silhouette > best_score:
                    best_score = silhouette
                    best_params = {'n_clusters': n_clusters, 'linkage': linkage}
                    best_labels = labels

        return {
            'method': 'Hierarchical',
            'best_params': best_params,
            'best_score': best_score,
            'best_labels': best_labels,
            'all_results': results
        }

    def run_all_clustering(self, X: np.ndarray) -> Dict[str, Any]:
        """Run all clustering algorithms"""
        results = {}

        # K-Means
        results['kmeans'] = self.kmeans_clustering(X)

        # DBSCAN
        dbscan_result = self.dbscan_clustering(X)
        if dbscan_result:
            results['dbscan'] = dbscan_result

        # Hierarchical
        results['hierarchical'] = self.hierarchical_clustering(X)

        return results

    def analyze_clusters(self, df: pd.DataFrame, labels: np.ndarray, method: str) -> Dict[str, Any]:
        """Analyze cluster composition"""
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = labels

        analysis = {
            'method': method,
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'cluster_sizes': {},
            'language_distribution': {}
        }

        # Cluster sizes
        for cluster_id in set(labels):
            if cluster_id == -1:  # Noise in DBSCAN
                continue
            cluster_size = sum(labels == cluster_id)
            analysis['cluster_sizes'][int(cluster_id)] = cluster_size

        # Language distribution per cluster
        if 'language' in df.columns:
            for cluster_id in set(labels):
                if cluster_id == -1:
                    continue

                cluster_mask = labels == cluster_id
                lang_dist = df[cluster_mask]['language'].value_counts().to_dict()
                analysis['language_distribution'][int(cluster_id)] = lang_dist

        return analysis

    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save clustering results"""
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj

        results_serializable = convert_types(results)

        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        logger.info(f"Saved clustering results to {output_file}")


if __name__ == "__main__":
    import yaml

    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Example usage
    # Load feature data
    # df = pd.read_csv('data/processed/features.csv')

    # pipeline = ClusteringPipeline(config)
    # X_scaled, feature_df = pipeline.prepare_data(df)

    # Apply PCA
    # X_pca = pipeline.apply_pca(X_scaled)

    # Run clustering
    # results = pipeline.run_all_clustering(X_pca)

    # Analyze results
    # for method, result in results.items():
    #     if result:
    #         analysis = pipeline.analyze_clusters(df, result['best_labels'], method)
    #         print(f"\n{method} Analysis:")
    #         print(json.dumps(analysis, indent=2))

    # Save results
    # pipeline.save_results(results, 'results/clustering_results.json')
