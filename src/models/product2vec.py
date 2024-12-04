import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import logging
import random
from tqdm import tqdm
from src.data.bpg import BehaviorProductGraph
from config import Config

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
        
        # Graph attention layer for neighbor aggregation
        self.attention = nn.MultiheadAttention(
            embed_dim=config.PRODUCT_EMB_DIM,
            num_heads=config.NUM_ATTENTION_HEADS,
            dropout=config.DROPOUT,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor, neighbors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate embeddings for input features"""
        # Initial embedding
        embeddings = self.ffn(x)
        
        if neighbors is not None:
            # Apply attention over neighbors
            attn_output, _ = self.attention(
                embeddings.unsqueeze(1),
                neighbors,
                neighbors
            )
            embeddings = attn_output.squeeze(1)
        
        return embeddings
    
    def _compute_loss(self, 
                     anchor_emb: torch.Tensor, 
                     positive_emb: torch.Tensor, 
                     negative_emb: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss for similar/dissimilar items"""
        pos_distance = F.pairwise_distance(anchor_emb, positive_emb)
        neg_distance = F.pairwise_distance(anchor_emb, negative_emb)
        
        loss = F.relu(pos_distance - neg_distance + self.config.MARGIN)
        return loss.mean()
    
    def train_model(self, 
                   train_loader: DataLoader,
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
                    anchor = batch['anchor'].to(device)
                    positive = batch['positive'].to(device)
                    negative = batch['negative'].to(device)
                    anchor_neighbors = batch.get('anchor_neighbors')
                    
                    if anchor_neighbors is not None:
                        anchor_neighbors = anchor_neighbors.to(device)
                    
                    # Forward pass
                    anchor_emb = self(anchor, anchor_neighbors)
                    positive_emb = self(positive)
                    negative_emb = self(negative)
                    
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

class SimilarityDataset(Dataset):
    """Dataset for training Product2Vec with similar/dissimilar pairs"""
    
    def __init__(self, bpg: 'BehaviorProductGraph', config: 'Config'):
        self.bpg = bpg
        self.config = config
        
        # Get similarity pairs from BPG
        # (Bcv ∩ Bpv) - Bcp as mentioned in the paper
        self.similar_pairs = self._get_similarity_pairs()
        
    def _get_similarity_pairs(self) -> List[Tuple[str, str]]:
        """Extract similarity pairs from BPG"""
        co_view = set(self.bpg.edges['co_view'])
        purchase_after_view = set(self.bpg.edges['purchase_after_view'])
        co_purchase = set(self.bpg.edges['co_purchase'])
        
        # Get pairs that are in both co_view and purchase_after_view but not in co_purchase
        similarity_pairs = list((co_view & purchase_after_view) - co_purchase)
        return similarity_pairs
    
    def _get_negative_sample(self, anchor_id: str) -> str:
        """Get a random negative sample for the anchor"""
        while True:
            neg_id = random.choice(list(self.bpg.nodes.keys()))
            if neg_id != anchor_id and (anchor_id, neg_id) not in set(self.similar_pairs):
                return neg_id
    
    def __len__(self) -> int:
        return len(self.similar_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        anchor_id, positive_id = self.similar_pairs[idx]
        negative_id = self._get_negative_sample(anchor_id)
        
        sample = {
            'anchor_ids': anchor_id,
            'anchor': self.bpg.nodes[anchor_id]['features'],
            'positive': self.bpg.nodes[positive_id]['features'],
            'negative': self.bpg.nodes[negative_id]['features'],
            'anchor_neighbors': self._get_neighbor_features(anchor_id)
        }
        
        return sample
    
    def _get_neighbor_features(self, product_id: str) -> Optional[torch.Tensor]:
        """Get neighbor features for a product"""
        neighbors = self.bpg.get_neighbors(product_id)
        if not neighbors:
            return None
            
        neighbor_features = []
        for n_id in neighbors:
            if n_id in self.bpg.nodes:
                neighbor_features.append(self.bpg.nodes[n_id]['features'])
                
        if neighbor_features:
            return torch.stack(neighbor_features)
        return None