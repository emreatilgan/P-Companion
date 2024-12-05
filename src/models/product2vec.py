import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging
from typing import Dict, Optional, Tuple

class Product2Vec(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Feed-forward network for initial embeddings (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(config.PRODUCT_EMB_DIM, config.HIDDEN_SIZE),
            nn.BatchNorm1d(config.HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(config.HIDDEN_SIZE, config.PRODUCT_EMB_DIM)
        )
        
        # Graph attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=config.PRODUCT_EMB_DIM,
            num_heads=config.NUM_ATTENTION_HEADS,
            dropout=config.DROPOUT,
            batch_first=True
        )
        
    def get_initial_embedding(self, features: torch.Tensor) -> torch.Tensor:
        """Get initial embedding through FFN"""
        if features.dim() == 1:
            # Single feature vector: [D] -> [1, D] -> [D]
            return self.ffn(features.unsqueeze(0)).squeeze(0)
        elif features.dim() == 2:
            # Batch of feature vectors: [B, D] -> [B, D]
            return self.ffn(features)
        elif features.dim() == 3:
            # Batch of multiple vectors: [B, N, D] -> [B*N, D] -> [B, N, D]
            B, N, D = features.shape
            features = features.reshape(-1, D)
            embeddings = self.ffn(features)
            return embeddings.reshape(B, N, -1)
        else:
            raise ValueError(f"Unexpected input dimension: {features.dim()}")
    
    def apply_attention(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """Apply attention mechanism with proper reshaping"""
        # Make sure inputs are 3D: [batch_size, seq_len, embed_dim]
        if query.dim() == 1:
            query = query.unsqueeze(0).unsqueeze(0)  # [D] -> [1, 1, D]
        elif query.dim() == 2:
            query = query.unsqueeze(1)  # [B, D] -> [B, 1, D]
            
        if key_value.dim() == 2:
            key_value = key_value.unsqueeze(0)  # [N, D] -> [1, N, D]
            
        # Apply attention
        attn_output, _ = self.attention(query, key_value, key_value)
        
        # Remove extra dimensions if they were added
        if query.size(0) == 1 and query.size(1) == 1:
            attn_output = attn_output.squeeze(0).squeeze(0)
        elif query.size(1) == 1:
            attn_output = attn_output.squeeze(1)
            
        return attn_output
    
    def forward(self, features: torch.Tensor, neighbors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through Product2Vec model"""
        # Get initial embeddings through FFN
        embeddings = self.get_initial_embedding(features)
        
        # Apply attention if neighbors are provided
        if neighbors is not None and neighbors.size(0) > 0:
            # Get initial embeddings for neighbors
            neighbor_embeddings = self.get_initial_embedding(neighbors)
            embeddings = self.apply_attention(embeddings, neighbor_embeddings)
        
        return embeddings
    
    def generate_all_embeddings(self, bpg) -> Dict[str, torch.Tensor]:
        """Generate embeddings for all products in the BPG"""
        self.eval()
        embeddings_dict = {}
        
        with torch.no_grad():
            # First pass: Get initial embeddings
            for product_id, product_data in tqdm(bpg.nodes.items(), desc="Generating initial embeddings"):
                features = product_data['features'].to(self.config.DEVICE)
                initial_emb = self.get_initial_embedding(features)
                embeddings_dict[product_id] = initial_emb.cpu()
            
            # Second pass: Update with neighbor information
            for product_id in tqdm(bpg.nodes.keys(), desc="Aggregating neighbor information"):
                neighbors = bpg.get_neighbors(product_id, edge_type='co_view')
                if neighbors:
                    # Get neighbor features and embeddings
                    neighbor_features = torch.stack([
                        bpg.nodes[n_id]['features'] for n_id in neighbors
                    ]).to(self.config.DEVICE)
                    
                    # Get current embedding
                    emb = embeddings_dict[product_id].to(self.config.DEVICE)
                    
                    # Update embedding with neighbor information
                    updated_emb = self(emb, neighbor_features)
                    embeddings_dict[product_id] = updated_emb.cpu()
        
        return embeddings_dict
    
    def train_model(self, train_loader, optimizer, num_epochs=10) -> Dict[str, torch.Tensor]:
        """Train Product2Vec model and generate embeddings for all products"""
        device = self.config.DEVICE
        logger = logging.getLogger(__name__)
        self.to(device)
        
        # Training loop
        for epoch in range(num_epochs):
            self.train()
            total_loss = 0.0
            num_batches = 0
            
            with tqdm(train_loader, desc=f'Product2Vec Epoch {epoch+1}/{num_epochs}') as pbar:
                for batch in pbar:
                    # Move batch to device
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Get embeddings
                    anchor_emb = self(batch['anchor'], batch.get('anchor_neighbors'))
                    positive_emb = self(batch['positive'])
                    negative_emb = self(batch['negative'])
                    
                    # Compute positive distances
                    pos_distance = F.pairwise_distance(anchor_emb, positive_emb)
                    
                    # Handle multiple negative samples
                    if negative_emb.dim() == 3:
                        anchor_expanded = anchor_emb.unsqueeze(1).expand(-1, negative_emb.size(1), -1)
                        neg_distance = torch.mean(
                            F.pairwise_distance(
                                anchor_expanded,
                                negative_emb,
                                p=2
                            ),
                            dim=1
                        )
                    else:
                        neg_distance = F.pairwise_distance(anchor_emb, negative_emb)
                    
                    # Compute triplet loss
                    loss = F.relu(self.config.MARGIN - pos_distance + neg_distance).mean()
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Update progress
                    total_loss += loss.item()
                    num_batches += 1
                    pbar.set_postfix({'loss': total_loss / num_batches})
            
            logger.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/num_batches:.4f}')
        
        # After training, generate embeddings for all products
        self.eval()
        return self.generate_all_embeddings(train_loader.dataset.bpg)