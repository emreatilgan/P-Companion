import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from src.data.bpg import BehaviorProductGraph
from config import Config

class SimilarityDataset(Dataset):
    """Dataset for training Product2Vec with similar/dissimilar pairs"""
    
    def __init__(self, bpg: 'BehaviorProductGraph', config: 'Config'):
        self.bpg = bpg
        self.config = config
        
        # Get similarity pairs from BPG
        self.similar_pairs = self._get_similarity_pairs()
        if len(self.similar_pairs) == 0:
            raise ValueError("No similarity pairs found in BPG. Check data generation.")
            
        print(f"Generated {len(self.similar_pairs)} similarity pairs for training.")
        
    def _get_similarity_pairs(self) -> List[Tuple[str, str]]:
        """Extract similarity pairs from BPG based on co-view and purchase-after-view"""
        co_view_pairs = set(self.bpg.edges['co_view'])
        pav_pairs = set(self.bpg.edges['purchase_after_view'])
        co_purchase = set(self.bpg.edges['co_purchase'])
        
        # Get pairs that are in both co_view and purchase_after_view but not in co_purchase
        similarity_pairs = list((co_view_pairs & pav_pairs) - co_purchase)
        
        # If we don't have enough similarity pairs, include all co-view pairs
        if len(similarity_pairs) < 100:  # Minimum threshold
            similarity_pairs = list(co_view_pairs - co_purchase)
            
        return similarity_pairs
    
    def _get_negative_samples(self, anchor_id: str, k: int = 5) -> List[str]:
        """Get multiple negative samples for the anchor"""
        neg_ids = []
        all_products = list(self.bpg.nodes.keys())
        similar_product_set = {pair[1] for pair in self.similar_pairs if pair[0] == anchor_id}
        
        while len(neg_ids) < k:
            neg_id = random.choice(all_products)
            if (neg_id != anchor_id and 
                neg_id not in similar_product_set and 
                neg_id not in neg_ids):
                neg_ids.append(neg_id)
                
        return neg_ids
    
    def __len__(self) -> int:
        return len(self.similar_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        anchor_id, positive_id = self.similar_pairs[idx]
        negative_ids = self._get_negative_samples(anchor_id)
        
        # Get features
        anchor_features = self.bpg.nodes[anchor_id]['features']
        positive_features = self.bpg.nodes[positive_id]['features']
        negative_features = torch.stack([
            self.bpg.nodes[neg_id]['features'] 
            for neg_id in negative_ids
        ])
        
        sample = {
            'anchor_ids': anchor_id,
            'anchor': anchor_features,
            'positive': positive_features,
            'negative': negative_features,
            'positive_id': positive_id,
            'negative_ids': negative_ids
        }
        
        # Add neighbor features if available
        anchor_neighbors = self._get_neighbor_features(anchor_id)
        if anchor_neighbors is not None:
            sample['anchor_neighbors'] = anchor_neighbors
            
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