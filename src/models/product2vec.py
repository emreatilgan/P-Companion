import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging
from typing import Dict, Optional

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
        original_shape = features.shape
        
        if features.dim() == 1:
            # Single feature vector
            features = features.unsqueeze(0)
            embeddings = self.ffn(features).squeeze(0)
        elif features.dim() == 2:
            # Batch of feature vectors
            embeddings = self.ffn(features)
        elif features.dim() == 3:
            # Batch of multiple feature vectors per sample
            batch_size, num_samples, feat_dim = features.shape
            features = features.reshape(-1, feat_dim)
            embeddings = self.ffn(features)
            embeddings = embeddings.reshape(batch_size, num_samples, -1)
        else:
            raise ValueError(f"Unexpected input dimension: {features.dim()}")
            
        return embeddings
    
    def _ensure_batch_sequence_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor has batch and sequence dimensions"""
        if tensor.dim() == 1:
            # [emb_dim] -> [1, 1, emb_dim]
            return tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 2:
            # [batch, emb_dim] -> [batch, 1, emb_dim]
            return tensor.unsqueeze(1)
        return tensor
    
    def forward(self, features: torch.Tensor, neighbors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through Product2Vec model"""
        # Get initial embeddings through FFN
        embeddings = self.get_initial_embedding(features)
        
        if neighbors is not None and neighbors.size(0) > 0:
            # Ensure proper dimensions for attention
            # [batch_size, seq_len, emb_dim] for both query and key/value
            embeddings = self._ensure_batch_sequence_dim(embeddings)
            neighbors = self._ensure_batch_sequence_dim(neighbors)
            
            # Apply attention
            attn_output, _ = self.attention(embeddings, neighbors, neighbors)
            
            # Remove sequence dimension if it was added
            if embeddings.dim() > neighbors.dim():
                attn_output = attn_output.squeeze(1)
            
            embeddings = attn_output
        
        return embeddings
    
    def generate_all_embeddings(self, bpg) -> Dict[str, torch.Tensor]:
        """Generate embeddings for all products in the BPG"""
        self.eval()
        embeddings_dict = {}
        
        with torch.no_grad():
            # First, get initial embeddings for all products through FFN
            for product_id, product_data in tqdm(bpg.nodes.items(), desc="Generating initial embeddings"):
                features = product_data['features'].to(self.config.DEVICE)
                initial_emb = self.get_initial_embedding(features)
                embeddings_dict[product_id] = initial_emb.cpu()
            
            # Then, update embeddings with neighbor information where applicable
            for product_id in tqdm(bpg.nodes.keys(), desc="Aggregating neighbor information"):
                neighbors = bpg.get_neighbors(product_id, edge_type='co_view')
                if neighbors:
                    neighbor_features = torch.stack([
                        bpg.nodes[n_id]['features'] for n_id in neighbors
                    ]).to(self.config.DEVICE)
                    
                    # Get updated embedding with neighbor information
                    emb = embeddings_dict[product_id].to(self.config.DEVICE)
                    
                    # Add batch and sequence dimensions for attention
                    emb = self._ensure_batch_sequence_dim(emb)
                    neighbor_features = self._ensure_batch_sequence_dim(neighbor_features)
                    
                    updated_emb = self(emb, neighbor_features)
                    embeddings_dict[product_id] = updated_emb.squeeze().cpu()
        
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
                    
                    # Ensure correct dimensions
                    if anchor_emb.dim() > 2:
                        anchor_emb = anchor_emb.squeeze(1)
                    if positive_emb.dim() > 2:
                        positive_emb = positive_emb.squeeze(1)
                    
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