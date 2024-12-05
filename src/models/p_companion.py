import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .type_transition import ComplementaryTypeTransition
from .item_prediction import ComplementaryItemPrediction

class PCompanion(nn.Module):
    def __init__(self, config, pretrained_embeddings: Dict[str, torch.Tensor]):
        super().__init__()
        self.config = config
        
        # Create product mapping and embedding matrix
        product_ids = list(pretrained_embeddings.keys())
        self.product_to_idx = {pid: idx for idx, pid in enumerate(product_ids)}
        
        # Create embedding matrix from pretrained embeddings
        embedding_dim = next(iter(pretrained_embeddings.values())).size(0)
        embedding_matrix = torch.zeros(len(product_ids), embedding_dim)
        
        for pid, idx in self.product_to_idx.items():
            embedding_matrix[idx] = pretrained_embeddings[pid]
        
        # Create frozen embedding layer with pretrained embeddings
        self.product_embeddings = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=True  # Keep pretrained embeddings fixed
        )
        
        # Initialize other components
        self.type_transition = ComplementaryTypeTransition(config)
        self.item_prediction = ComplementaryItemPrediction(config)
        
        # Type embeddings
        self.query_type_embeddings = nn.Embedding(
            config.NUM_TYPES,
            config.TYPE_EMB_DIM
        )
        self.complementary_type_embeddings = nn.Embedding(
            config.NUM_TYPES,
            config.TYPE_EMB_DIM
        )
        
    def forward(self, batch):
        # Get product embeddings using pretrained embeddings
        query_indices = torch.tensor([
            self.product_to_idx[pid] for pid in batch['query_ids']
        ]).to(self.config.DEVICE)
        
        query_embeddings = self.product_embeddings(query_indices)
        
        # Get type embeddings
        query_type_emb = self.query_type_embeddings(batch['query_types'])
        
        # Predict complementary types
        comp_base = self.type_transition(query_type_emb)
        
        # Get top-k complementary types
        similarities = torch.matmul(
            comp_base,
            self.complementary_type_embeddings.weight.T
        )
        top_k_types = torch.topk(similarities, k=self.config.NUM_COMP_TYPES, dim=1)
        comp_type_embeddings = self.complementary_type_embeddings(top_k_types.indices)
        
        # Predict complementary items
        projected_embeddings = self.item_prediction(
            query_embeddings,
            comp_type_embeddings
        )
        
        return {
            'projected_embeddings': projected_embeddings,
            'complementary_types': top_k_types.indices,
            'type_similarities': similarities
        }

    def compute_loss(self, batch, outputs):
        """Compute combined loss for type transition and item prediction"""
        type_loss = self._compute_type_loss(
            outputs['type_similarities'],
            batch['positive_types'].squeeze(-1),
            batch['negative_types'].squeeze(-1)
        )
        
        item_loss = self._compute_item_loss(
            outputs['projected_embeddings'],
            batch['positive_items'],
            batch['negative_items']
        )
        
        return self.config.ALPHA * item_loss + (1 - self.config.ALPHA) * type_loss

    def _compute_type_loss(self, type_similarities, positive_types, negative_types):
        positive_scores = type_similarities[torch.arange(type_similarities.size(0)), positive_types]
        negative_scores = type_similarities[torch.arange(type_similarities.size(0)), negative_types]
        
        loss = torch.mean(torch.clamp(
            self.config.MARGIN - positive_scores + negative_scores,
            min=0.0
        ))
        return loss

    def _compute_item_loss(self, projected_embeddings, positive_items, negative_items):
        pos_distances = torch.norm(
            projected_embeddings - positive_items.unsqueeze(1),
            dim=-1
        )
        neg_distances = torch.norm(
            projected_embeddings - negative_items.unsqueeze(1),
            dim=-1
        )
        
        loss = torch.mean(torch.clamp(
            self.config.MARGIN - pos_distances + neg_distances,
            min=0.0
        ))
        return loss