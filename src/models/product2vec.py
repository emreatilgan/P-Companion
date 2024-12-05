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
        if features.dim() == 1:
            # Single feature vector
            features = features.unsqueeze(0)  # Add batch dimension
            return self.ffn(features).squeeze(0)  # Remove batch dimension
        elif features.dim() == 2:
            # Batch of feature vectors
            return self.ffn(features)
        else:
            raise ValueError(f"Unexpected input dimension: {features.dim()}")
    
    def forward(self, features: torch.Tensor, neighbors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through Product2Vec model"""
        # Get initial embeddings through FFN
        embeddings = self.get_initial_embedding(features)
        
        if neighbors is not None and neighbors.size(0) > 0:
            # Make sure embeddings have batch dimension
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)
            
            # Add sequence dimension if needed
            if embeddings.dim() == 2:
                embeddings = embeddings.unsqueeze(1)
            
            # Apply attention
            attn_output, _ = self.attention(
                embeddings,
                neighbors.unsqueeze(0) if neighbors.dim() == 2 else neighbors,
                neighbors.unsqueeze(0) if neighbors.dim() == 2 else neighbors
            )
            
            # Remove extra dimensions
            if attn_output.size(0) == 1:
                attn_output = attn_output.squeeze(0)
            if attn_output.size(0) == 1:
                attn_output = attn_output.squeeze(0)
                
            embeddings = attn_output
        
        return embeddings
    
    def _compute_loss(self, anchor_emb: torch.Tensor, 
                     positive_emb: torch.Tensor,
                     negative_emb: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss for similar/dissimilar items"""
        # Expand anchor to match negative samples
        anchor_emb = anchor_emb.unsqueeze(1).expand(-1, negative_emb.size(1), -1)
        positive_emb = positive_emb.unsqueeze(1).expand(-1, negative_emb.size(1), -1)
        
        # Compute distances
        pos_distance = F.pairwise_distance(anchor_emb, positive_emb, p=2, keepdim=True)
        neg_distance = F.pairwise_distance(anchor_emb, negative_emb, p=2, keepdim=True)
        
        # Compute triplet loss
        loss = F.relu(pos_distance - neg_distance + self.config.MARGIN)
        return loss.mean()
    
    def generate_all_embeddings(self, bpg) -> Dict[str, torch.Tensor]:
        """Generate embeddings for all products in the BPG"""
        self.eval()
        embeddings_dict = {}
        
        with torch.no_grad():
            # First, get initial embeddings for all products through FFN
            for product_id, product_data in tqdm(bpg.nodes.items(), desc="Generating initial embeddings"):
                features = product_data['features'].to(self.config.DEVICE)
                initial_emb = self.get_initial_embedding(features)
                embeddings_dict[product_id] = initial_emb.cpu()  # Store on CPU to save memory
            
            # Then, update embeddings with neighbor information where applicable
            for product_id in tqdm(bpg.nodes.keys(), desc="Aggregating neighbor information"):
                neighbors = bpg.get_neighbors(product_id, edge_type='co_view')
                if neighbors:
                    neighbor_features = torch.stack([
                        bpg.nodes[n_id]['features'] for n_id in neighbors
                    ]).to(self.config.DEVICE)
                    
                    # Get updated embedding with neighbor information
                    emb = embeddings_dict[product_id].to(self.config.DEVICE)
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
                    
                    # Compute loss
                    pos_distance = F.pairwise_distance(
                        anchor_emb,
                        positive_emb
                    )
                    
                    # Handle multiple negative samples
                    if negative_emb.dim() == 3:
                        # Expand anchor to match negative samples
                        anchor_emb = anchor_emb.unsqueeze(1).expand(-1, negative_emb.size(1), -1)
                    
                    neg_distance = F.pairwise_distance(
                        anchor_emb,
                        negative_emb,
                        keepdim=True if negative_emb.dim() == 3 else False
                    ).mean(dim=1) if negative_emb.dim() == 3 else F.pairwise_distance(anchor_emb, negative_emb)
                    
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