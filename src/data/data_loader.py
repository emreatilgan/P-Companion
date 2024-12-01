import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict

class BPGDataset(Dataset):
    def __init__(self, bpg, config, mode='train'):
        self.bpg = bpg
        self.config = config
        self.mode = mode
        
        # Create product pairs from behavioral data
        self.pairs = self._create_product_pairs()
        
        # Create type mappings
        self.type_to_idx = {t: i for i, t in enumerate(bpg.get_all_types())}
        self.idx_to_type = {i: t for t, i in self.type_to_idx.items()}
        
    def _create_product_pairs(self):
        """Create training pairs based on behavior data"""
        pairs = []
        
        # Positive pairs from co-purchase data (excluding co-view intersection)
        positive_pairs = self.bpg.get_exclusive_co_purchase_pairs()
        
        # Negative pairs from co-view intersection
        negative_pairs = self.bpg.get_co_view_intersection_pairs()
        
        for query_id, target_id, label in positive_pairs + negative_pairs:
            if query_id in self.bpg.nodes and target_id in self.bpg.nodes:
                pairs.append((query_id, target_id, label))
        
        return pairs
    
    def _get_neighbor_features(self, product_id):
        """Get aggregated neighbor features for a product"""
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
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        query_id, target_id, label = self.pairs[idx]
        
        # Get product features
        query_features = self.bpg.nodes[query_id]['features']
        target_features = self.bpg.nodes[target_id]['features']
        
        # Get product types
        query_type = self.type_to_idx[self.bpg.nodes[query_id]['type']]
        target_type = self.type_to_idx[self.bpg.nodes[target_id]['type']]
        
        # Get neighbor features
        query_neighbors = self._get_neighbor_features(query_id)
        
        sample = {
            'query_features': query_features,
            'target_features': target_features,
            'query_type': query_type,
            'target_type': target_type,
            'label': label
        }
        
        if query_neighbors is not None:
            sample['query_neighbor_features'] = query_neighbors
            
        return sample

def collate_fn(batch):
    """Custom collate function to handle variable-sized neighbor features"""
    batch_dict = defaultdict(list)
    
    for sample in batch:
        for key, value in sample.items():
            batch_dict[key].append(value)
    
    # Stack tensors
    for key in batch_dict:
        if key != 'query_neighbor_features':
            batch_dict[key] = torch.stack(batch_dict[key])
    
    return batch_dict