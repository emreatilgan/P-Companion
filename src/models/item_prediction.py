import torch
import torch.nn as nn
import torch.nn.functional as Fa

class ComplementaryItemPrediction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Projection network for each complementary type
        self.type_projection = nn.Linear(
            config.TYPE_EMB_DIM,
            config.PRODUCT_EMB_DIM
        )
        
        # Transformation matrix for item embedding projection
        self.item_projection = nn.Linear(
            config.PRODUCT_EMB_DIM,
            config.PRODUCT_EMB_DIM
        )
        
    def forward(self, query_item_embedding, complementary_type_embeddings):
        """
        Args:
            query_item_embedding: Tensor of shape [batch_size, product_emb_dim]
            complementary_type_embeddings: Tensor of shape [batch_size, num_types, type_emb_dim]
        Returns:
            projected_embeddings: Tensor of shape [batch_size, num_types, product_emb_dim]
        """
        # Project item embedding
        projected_item = self.item_projection(query_item_embedding)
        
        # Project each complementary type
        batch_size, num_types, _ = complementary_type_embeddings.shape
        type_projections = self.type_projection(complementary_type_embeddings)
        
        # Combine item and type projections
        projected_embeddings = projected_item.unsqueeze(1) * type_projections
        
        return projected_embeddings