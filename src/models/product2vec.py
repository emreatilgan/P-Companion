import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.tanh(self.layer1(x))
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
            dropout=config.DROPOUT
        )
        
    def forward(self, catalog_features, neighbor_features=None):
        # Generate initial embeddings
        embeddings = self.ffn(catalog_features)
        
        if neighbor_features is not None:
            # Apply attention over neighbors
            attn_output, attention_weights = self.attention(
                embeddings.unsqueeze(0),
                neighbor_features.unsqueeze(0),
                neighbor_features.unsqueeze(0)
            )
            embeddings = attn_output.squeeze(0)
            
        return embeddings