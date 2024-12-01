import torch
import torch.nn as nn
import torch.nn.functional as F

from .product2vec import Product2Vec
from .type_transition import ComplementaryTypeTransition
from .item_prediction import ComplementaryItemPrediction

class PCompanion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize all components
        self.product2vec = Product2Vec(config)
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
        # Get product embeddings
        query_embeddings = self.product2vec(
            batch['query_features'],
            batch.get('query_neighbor_features', None)
        )
        
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
            'type_similarities': top_k_types.values
        }

    def compute_loss(self, batch, outputs):
        """
        Compute combined loss for type transition and item prediction
        """
        type_loss = self._compute_type_loss(
            outputs['type_similarities'],
            batch['positive_types'],
            batch['negative_types']
        )
        
        item_loss = self._compute_item_loss(
            outputs['projected_embeddings'],
            batch['positive_items'],
            batch['negative_items']
        )
        
        return self.config.ALPHA * item_loss + (1 - self.config.ALPHA) * type_loss

    def _compute_type_loss(self, type_similarities, positive_types, negative_types):
        positive_scores = type_similarities.gather(1, positive_types)
        negative_scores = type_similarities.gather(1, negative_types)
        
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