"""
Visualization Module
Creates visualizations for clustering results, t-SNE, and metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 300


class ClusteringVisualizer:
    """Visualize clustering results"""

    def __init__(self, config: Dict, output_dir: str = 'results/visualizations'):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_tsne(self, X: np.ndarray, labels: np.ndarray, languages: np.ndarray = None,
                  title: str = 't-SNE Visualization', save_path: str = None):
        """Create t-SNE visualization"""
        logger.info("Creating t-SNE visualization")

        tsne_config = self.config['visualization']['tsne']
        
        # Adjust perplexity based on sample size (must be less than n_samples)
        n_samples = X.shape[0]
        perplexity = min(tsne_config['perplexity'], n_samples - 1)

        # Apply t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=tsne_config.get('n_iter', 1000),  # max_iter is the correct parameter name
            random_state=tsne_config['random_state']
        )
        X_tsne = tsne.fit_transform(X)

        # Create plot
        fig, axes = plt.subplots(1, 2 if languages is not None else 1, figsize=(20, 8))

        if languages is None:
            ax = axes if isinstance(axes, plt.Axes) else axes[0]
        else:
            ax = axes[0]

        # Plot by cluster
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10',
                             alpha=0.6, s=50)
        ax.set_title(f'{title} - Colored by Cluster', fontsize=14)
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        plt.colorbar(scatter, ax=ax, label='Cluster')

        # Plot by language if available
        if languages is not None:
            ax = axes[1]
            unique_langs = np.unique(languages)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_langs)))

            for i, lang in enumerate(unique_langs):
                mask = languages == lang
                ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                           c=[colors[i]], label=lang, alpha=0.6, s=50)

            ax.set_title(f'{title} - Colored by Language', fontsize=14)
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
            ax.legend()

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'tsne_plot.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved t-SNE plot to {save_path}")
        plt.close()

    def plot_pca(self, X: np.ndarray, labels: np.ndarray, languages: np.ndarray = None,
                 title: str = 'PCA Visualization', save_path: str = None):
        """Create PCA visualization"""
        logger.info("Creating PCA visualization")

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        fig, axes = plt.subplots(1, 2 if languages is not None else 1, figsize=(20, 8))

        if languages is None:
            ax = axes if isinstance(axes, plt.Axes) else axes[0]
        else:
            ax = axes[0]

        # Plot by cluster
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10',
                             alpha=0.6, s=50)
        ax.set_title(f'{title} - Colored by Cluster', fontsize=14)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.colorbar(scatter, ax=ax, label='Cluster')

        # Plot by language if available
        if languages is not None:
            ax = axes[1]
            unique_langs = np.unique(languages)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_langs)))

            for i, lang in enumerate(unique_langs):
                mask = languages == lang
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                           c=[colors[i]], label=lang, alpha=0.6, s=50)

            ax.set_title(f'{title} - Colored by Language', fontsize=14)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            ax.legend()

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'pca_plot.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved PCA plot to {save_path}")
        plt.close()

    def plot_cluster_distribution(self, labels: np.ndarray, languages: np.ndarray,
                                   save_path: str = None):
        """Plot language distribution across clusters"""
        logger.info("Creating cluster distribution plot")

        df = pd.DataFrame({'cluster': labels, 'language': languages})

        # Count distribution
        distribution = df.groupby(['cluster', 'language']).size().unstack(fill_value=0)

        # Create stacked bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        distribution.plot(kind='bar', stacked=True, ax=ax, colormap='Set3')

        ax.set_title('Language Distribution Across Clusters', fontsize=14)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Functions')
        ax.legend(title='Language')
        plt.xticks(rotation=0)
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'cluster_distribution.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved cluster distribution plot to {save_path}")
        plt.close()

    def plot_silhouette_comparison(self, results: Dict[str, Any], save_path: str = None):
        """Compare silhouette scores across methods"""
        logger.info("Creating silhouette score comparison")

        methods = []
        scores = []

        for method, result in results.items():
            if result and 'best_score' in result:
                methods.append(method.upper())
                scores.append(result['best_score'])

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(methods, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

        ax.set_title('Clustering Method Comparison (Silhouette Score)', fontsize=14)
        ax.set_xlabel('Method')
        ax.set_ylabel('Silhouette Score')
        ax.set_ylim([0, 1])

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.4f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'silhouette_comparison.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved silhouette comparison to {save_path}")
        plt.close()

    def plot_elbow_curve(self, kmeans_results: List[Dict], save_path: str = None):
        """Plot elbow curve for K-Means"""
        logger.info("Creating elbow curve")

        n_clusters = [r['n_clusters'] for r in kmeans_results]
        inertias = [r['inertia'] for r in kmeans_results]
        silhouette_scores = [r['silhouette_score'] for r in kmeans_results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Elbow curve (Inertia)
        ax1.plot(n_clusters, inertias, 'bo-')
        ax1.set_title('Elbow Method (Inertia)', fontsize=14)
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Inertia')
        ax1.grid(True)

        # Silhouette scores
        ax2.plot(n_clusters, silhouette_scores, 'ro-')
        ax2.set_title('Silhouette Score vs Number of Clusters', fontsize=14)
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.grid(True)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'elbow_curve.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved elbow curve to {save_path}")
        plt.close()

    def plot_feature_importance(self, feature_names: List[str], importances: np.ndarray,
                                 top_k: int = 20, save_path: str = None):
        """Plot feature importance from Random Forest"""
        logger.info("Creating feature importance plot")

        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_k]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(range(top_k), importances[indices], color='steelblue')
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_k} Most Important Features', fontsize=14)
        ax.invert_yaxis()

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'feature_importance.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {save_path}")
        plt.close()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                               labels: List[str], save_path: str = None):
        """Plot confusion matrix"""
        from sklearn.metrics import confusion_matrix

        logger.info("Creating confusion matrix")

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_title('Confusion Matrix', fontsize=14)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'confusion_matrix.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
        plt.close()


class MetricsVisualizer:
    """Visualize complexity metrics across languages"""

    def __init__(self, output_dir: str = 'results/visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_metric_distribution(self, df: pd.DataFrame, metric_col: str,
                                  group_by: str = 'language', save_path: str = None):
        """Plot metric distribution across groups"""
        logger.info(f"Creating distribution plot for {metric_col}")

        fig, ax = plt.subplots(figsize=(12, 6))

        df.boxplot(column=metric_col, by=group_by, ax=ax)
        ax.set_title(f'{metric_col} Distribution by {group_by}', fontsize=14)
        ax.set_xlabel(group_by.capitalize())
        ax.set_ylabel(metric_col)

        plt.suptitle('')  # Remove default title
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, f'{metric_col}_distribution.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved distribution plot to {save_path}")
        plt.close()

    def plot_correlation_heatmap(self, df: pd.DataFrame, features: List[str] = None,
                                  save_path: str = None):
        """Plot correlation heatmap of features"""
        logger.info("Creating correlation heatmap")

        if features is None:
            # Select numeric columns
            features = df.select_dtypes(include=[np.number]).columns.tolist()

        corr_matrix = df[features].corr()

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, ax=ax)
        ax.set_title('Feature Correlation Heatmap', fontsize=14)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'correlation_heatmap.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved correlation heatmap to {save_path}")
        plt.close()

    def plot_language_comparison(self, df: pd.DataFrame, metrics: List[str],
                                  save_path: str = None):
        """Compare multiple metrics across languages"""
        logger.info("Creating language comparison plot")

        languages = df['language'].unique()
        n_metrics = len(metrics)

        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))

        if n_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            df.boxplot(column=metric, by='language', ax=ax)
            ax.set_title(f'{metric} by Language', fontsize=12)
            ax.set_xlabel('Language')
            ax.set_ylabel(metric)

        plt.suptitle('')
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'language_comparison.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved language comparison to {save_path}")
        plt.close()


if __name__ == "__main__":
    import yaml

    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    logger.info("Visualization module loaded")
