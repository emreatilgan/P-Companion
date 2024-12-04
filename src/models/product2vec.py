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
        
        # Feed-forward network for initial embeddings
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
        
    def forward(self, x: torch.Tensor, neighbors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate embeddings for input features"""
        batch_size = x.size(0)
        
        # Initial embedding
        if x.dim() == 2:
            embeddings = self.ffn(x)
        else:
            # Reshape for batch norm
            embeddings = self.ffn(x.view(-1, x.size(-1))).view(batch_size, -1, self.config.PRODUCT_EMB_DIM)
        
        if neighbors is not None and neighbors.size(0) > 0:
            # Apply attention over neighbors
            attn_output, _ = self.attention(
                embeddings.unsqueeze(1) if embeddings.dim() == 2 else embeddings,
                neighbors,
                neighbors
            )
            embeddings = attn_output.squeeze(1) if embeddings.dim() == 2 else attn_output
        
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
    
    def train_model(self, train_loader: torch.utils.data.DataLoader,
                   optimizer: torch.optim.Optimizer,
                   num_epochs: int = 10) -> Dict[str, torch.Tensor]:
        """Train Product2Vec model and return learned embeddings"""
        device = self.config.DEVICE
        self.to(device)
        
        logger = logging.getLogger(__name__)
        embeddings_dict = {}
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            with tqdm(train_loader, desc=f'Product2Vec Epoch {epoch+1}/{num_epochs}') as pbar:
                for batch in pbar:
                    # Move batch to device
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Forward pass
                    anchor_emb = self(batch['anchor'], 
                                    batch.get('anchor_neighbors', None))
                    positive_emb = self(batch['positive'])
                    negative_emb = self(batch['negative'])
                    
                    # Compute loss
                    loss = self._compute_loss(anchor_emb, positive_emb, negative_emb)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    pbar.set_postfix({'loss': total_loss / num_batches})
                    
                    # Store embeddings
                    for idx, pid in enumerate(batch['anchor_ids']):
                        embeddings_dict[pid] = anchor_emb[idx].detach().cpu()
            
            logger.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/num_batches:.4f}')
        
        return embeddings_dict