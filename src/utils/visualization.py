import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import logging
from collections import defaultdict
from src.data.bpg import BehaviorProductGraph

class EmbeddingVisualizer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def visualize_embeddings(
        self,
        embeddings_dict: Dict[str, torch.Tensor],
        bpg: 'BehaviorProductGraph',
        save_path: str,
        n_components: int = 2,
        perplexity: int = 30,
        random_state: int = 42
    ) -> None:
        """Visualize product embeddings using t-SNE"""
        self.logger.info("Starting embedding visualization...")
        
        # Convert embeddings to numpy array
        product_ids = list(embeddings_dict.keys())
        embeddings = np.stack([embeddings_dict[pid].numpy() for pid in product_ids])
        
        # Apply t-SNE
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state
        )
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Get categories and types for coloring
        categories = [bpg.nodes[pid]['category'] for pid in product_ids]
        types = [bpg.nodes[pid]['type'] for pid in product_ids]
        
        # Create visualization plots
        self._plot_by_category(embeddings_2d, categories, save_path + '_category.png')
        self._plot_by_type(embeddings_2d, types, save_path + '_type.png')
        self._plot_similarity_pairs(embeddings_2d, product_ids, bpg, save_path + '_similarity.png')
        self._plot_complementary_pairs(embeddings_2d, product_ids, bpg, save_path + '_complementary.png')
        
        self.logger.info("Embedding visualization completed")
        
    def _plot_by_category(
        self,
        embeddings_2d: np.ndarray,
        categories: List[str],
        save_path: str
    ) -> None:
        """Plot embeddings colored by product category"""
        plt.figure(figsize=(12, 8))
        
        # Create color mapping
        unique_categories = list(set(categories))
        colors = sns.color_palette('husl', n_colors=len(unique_categories))
        cat_to_color = dict(zip(unique_categories, colors))
        
        # Plot points
        for cat in unique_categories:
            mask = [c == cat for c in categories]
            points = embeddings_2d[mask]
            plt.scatter(
                points[:, 0],
                points[:, 1],
                c=[cat_to_color[cat]],
                label=cat,
                alpha=0.6
            )
            
        plt.title('Product Embeddings by Category')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    def _plot_by_type(
        self,
        embeddings_2d: np.ndarray,
        types: List[str],
        save_path: str
    ) -> None:
        """Plot embeddings colored by product type"""
        # Group types by category for better visualization
        type_by_category = defaultdict(list)
        for t in types:
            category = t.split('_')[2]  # Assuming format: type_X_category
            type_by_category[category].append(t)
            
        plt.figure(figsize=(12, 8))
        
        # Plot each category's types with similar colors
        for idx, (category, category_types) in enumerate(type_by_category.items()):
            base_color = sns.color_palette('husl', n_colors=len(type_by_category))[idx]
            type_colors = sns.light_palette(base_color, n_colors=len(category_types) + 2)[2:]
            
            for type_idx, prod_type in enumerate(category_types):
                mask = [t == prod_type for t in types]
                points = embeddings_2d[mask]
                plt.scatter(
                    points[:, 0],
                    points[:, 1],
                    c=[type_colors[type_idx]],
                    label=prod_type,
                    alpha=0.6
                )
                
        plt.title('Product Embeddings by Type')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    def _plot_similarity_pairs(
        self,
        embeddings_2d: np.ndarray,
        product_ids: List[str],
        bpg: 'BehaviorProductGraph',
        save_path: str
    ) -> None:
        """Plot embeddings with similarity pairs connected"""
        plt.figure(figsize=(12, 8))
        
        # Plot all points in grey
        plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c='lightgrey',
            alpha=0.5
        )
        
        # Create index mapping
        pid_to_idx = {pid: idx for idx, pid in enumerate(product_ids)}
        
        # Plot similarity pairs
        for p1, p2 in bpg.similarity_pairs:
            if p1 in pid_to_idx and p2 in pid_to_idx:
                idx1, idx2 = pid_to_idx[p1], pid_to_idx[p2]
                plt.plot(
                    [embeddings_2d[idx1, 0], embeddings_2d[idx2, 0]],
                    [embeddings_2d[idx1, 1], embeddings_2d[idx2, 1]],
                    'b-',
                    alpha=0.2
                )
                
        plt.title('Product Embeddings with Similarity Pairs')
        plt.savefig(save_path)
        plt.close()
        
    def _plot_complementary_pairs(
        self,
        embeddings_2d: np.ndarray,
        product_ids: List[str],
        bpg: 'BehaviorProductGraph',
        save_path: str
    ) -> None:
        """Plot embeddings with complementary pairs connected"""
        plt.figure(figsize=(12, 8))
        
        # Plot all points in grey
        plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c='lightgrey',
            alpha=0.5
        )
        
        # Create index mapping
        pid_to_idx = {pid: idx for idx, pid in enumerate(product_ids)}
        
        # Plot complementary pairs
        for p1, p2 in bpg.complementary_pairs:
            if p1 in pid_to_idx and p2 in pid_to_idx:
                idx1, idx2 = pid_to_idx[p1], pid_to_idx[p2]
                plt.plot(
                    [embeddings_2d[idx1, 0], embeddings_2d[idx2, 0]],
                    [embeddings_2d[idx1, 1], embeddings_2d[idx2, 1]],
                    'r-',
                    alpha=0.2
                )
                
        plt.title('Product Embeddings with Complementary Pairs')
        plt.savefig(save_path)
        plt.close()