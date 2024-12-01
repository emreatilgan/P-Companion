import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from src.data.synthetic_data import SyntheticDataGenerator
from src.data.bpg import BehaviorProductGraph
from config import Config

class BPGDataset(Dataset):
    def __init__(self, bpg: BehaviorProductGraph, config: 'Config', mode: str = 'train'):
        self.bpg = bpg
        self.config = config
        self.mode = mode
        
        # Create product pairs from behavioral data
        self.pairs = self._create_product_pairs()
        
        # Create type mappings
        self.type_to_idx = {t: i for i, t in enumerate(bpg.get_all_types())}
        self.idx_to_type = {i: t for t, i in self.type_to_idx.items()}
        
    def _create_product_pairs(self) -> List[Tuple[str, str, int]]:
        """Create training pairs based on behavior data"""
        pairs = []
        
        # Positive pairs from co-purchase data (excluding co-view intersection)
        positive_pairs = self.bpg.get_exclusive_co_purchase_pairs()
        
        # Negative pairs from co-view intersection
        negative_pairs = self.bpg.get_co_view_intersection_pairs()
        
        # Combine positive and negative pairs
        all_pairs = positive_pairs + negative_pairs
        
        # Split data based on mode
        if self.mode == 'train':
            pairs = all_pairs[:int(len(all_pairs) * 0.8)]
        elif self.mode == 'val':
            pairs = all_pairs[int(len(all_pairs) * 0.8):int(len(all_pairs) * 0.9)]
        else:  # test
            pairs = all_pairs[int(len(all_pairs) * 0.9):]
        
        return pairs
    
    def _get_neighbor_features(self, product_id: str) -> Optional[torch.Tensor]:
        """Get aggregated neighbor features for a product"""
        neighbors = self.bpg.get_neighbors(product_id)
        if not neighbors:
            return None
        
        neighbor_features = []
        for n_id in neighbors:
            if n_id in self.bpg.nodes:
                neighbor_features.append(self.bpg.nodes[n_id]['features'])
        
        if neighbor_features:
            # Stack neighbor features into a single tensor
            return torch.stack(neighbor_features)
        return None
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        query_id, target_id, label = self.pairs[idx]
        
        # Get product features
        query_features = self.bpg.nodes[query_id]['features']
        target_features = self.bpg.nodes[target_id]['features']
        
        # Get product types
        query_type = self.type_to_idx[self.bpg.nodes[query_id]['type']]
        target_type = self.type_to_idx[self.bpg.nodes[target_id]['type']]
        
        # Create sample with proper dimensions
        sample = {
            'query_features': query_features,
            'target_features': target_features,
            'query_types': torch.tensor(query_type),  # [1]
            'positive_types': torch.tensor(target_type if label == 1 else 0),  # [1]
            'negative_types': torch.tensor(target_type if label == -1 else 
                                        min(target_type + 1, len(self.type_to_idx) - 1)),  # [1]
            'positive_items': target_features if label == 1 else torch.randn_like(target_features),
            'negative_items': target_features if label == -1 else torch.randn_like(target_features),
            'label': torch.tensor(label)
        }
        
        # Get neighbor features
        query_neighbors = self._get_neighbor_features(query_id)
        if query_neighbors is not None:
            sample['query_neighbor_features'] = query_neighbors
            
        return sample

class SyntheticBPGDataset(BPGDataset):
    def __init__(self, config: 'Config', mode: str = 'train'):
        # Generate synthetic data
        generator = SyntheticDataGenerator(config)
        bpg = generator.generate_bpg()
        
        super().__init__(bpg, config, mode)

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function to handle variable-sized neighbor features"""
    batch_dict = defaultdict(list)
    
    for sample in batch:
        for key, value in sample.items():
            # Special handling for neighbor features
            if key == 'query_neighbor_features':
                batch_dict[key].append(value if value is not None else torch.tensor([]))
            else:
                batch_dict[key].append(value)
    
    # Stack tensors
    for key in batch_dict:
        if key != 'query_neighbor_features':
            batch_dict[key] = torch.stack(batch_dict[key])
    
    return dict(batch_dict)