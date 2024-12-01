import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.layer1(x))
        if x.dim() > 2:
            # Handle batched sequences
            batch_size, seq_len, hidden_dim = x.size()
            x = x.view(-1, hidden_dim)
            x = self.batch_norm(x)
            x = x.view(batch_size, seq_len, hidden_dim)
        else:
            x = self.batch_norm(x)
        x = torch.tanh(self.layer2(x))
        x = self.layer3(x)
        return x

class Product2Vec(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Feed-forward network for initial embeddings
        self.ffn = FeedForwardNetwork(
            input_dim=config.PRODUCT_EMB_DIM,
            hidden_dim=config.HIDDEN_SIZE,
            output_dim=config.PRODUCT_EMB_DIM
        )
        
        # Graph attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=config.PRODUCT_EMB_DIM,
            num_heads=config.NUM_ATTENTION_HEADS,
            dropout=config.DROPOUT,
            batch_first=True
        )
        
    def forward(self, catalog_features: torch.Tensor, 
                neighbor_features: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass through Product2Vec model
        
        Args:
            catalog_features: Tensor of shape [batch_size, product_emb_dim]
            neighbor_features: Optional list of neighbor feature tensors
            
        Returns:
            Tensor of shape [batch_size, product_emb_dim]
        """
        # Generate initial embeddings
        embeddings = self.ffn(catalog_features)
        
        if neighbor_features is not None and len(neighbor_features) > 0:
            # Process each batch item separately since they might have different numbers of neighbors
            batch_size = catalog_features.size(0)
            processed_embeddings = []
            
            for i in range(batch_size):
                if isinstance(neighbor_features[i], torch.Tensor):
                    # Get neighbor embeddings for current item
                    current_neighbors = neighbor_features[i]  # [num_neighbors, emb_dim]
                    
                    if current_neighbors.dim() == 1:
                        # Single neighbor case
                        current_neighbors = current_neighbors.unsqueeze(0)
                    
                    # Apply attention
                    query = embeddings[i].unsqueeze(0)  # [1, emb_dim]
                    key = current_neighbors
                    value = current_neighbors
                    
                    attn_output, _ = self.attention(
                        query.unsqueeze(0),
                        key.unsqueeze(0),
                        value.unsqueeze(0)
                    )
                    
                    processed_embeddings.append(attn_output.squeeze(0))
                else:
                    # No neighbors case - use original embedding
                    processed_embeddings.append(embeddings[i].unsqueeze(0))
            
            # Combine processed embeddings
            embeddings = torch.cat(processed_embeddings, dim=0)
            
        return embeddings