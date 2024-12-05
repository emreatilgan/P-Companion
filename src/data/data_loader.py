# src/data/data_loader.py
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

class ComplementaryDataset(Dataset):
    """Dataset for training P-Companion with complementary product pairs"""
    
    def __init__(self, bpg: 'BehaviorProductGraph', config: 'Config', mode: str = 'train'):
        self.bpg = bpg
        self.config = config
        self.mode = mode
        
        # Get complementary pairs from BPG
        self.pairs = self._create_product_pairs()
        
        # Create type mappings
        self.type_to_idx = {t: i for i, t in enumerate(bpg.get_all_types())}
        self.idx_to_type = {i: t for t, i in self.type_to_idx.items()}
        
    def _create_product_pairs(self) -> List[Tuple[str, str, int]]:
        """Create training pairs based on behavior data"""
        all_pairs = []
        
        # Positive pairs from exclusive co-purchase data
        positive_pairs = [(src, tgt, 1) for src, tgt in self.bpg.edges['co_purchase']
                         if (src, tgt) not in self.bpg.edges['co_view']]
        
        # Negative pairs from co-view ∩ purchase-after-view
        co_view = set(self.bpg.edges['co_view'])
        pav = set(self.bpg.edges['purchase_after_view'])
        negative_pairs = [(src, tgt, -1) for src, tgt in (co_view & pav)]
        
        all_pairs.extend(positive_pairs)
        all_pairs.extend(negative_pairs)
        random.shuffle(all_pairs)
        
        # Split based on mode
        total = len(all_pairs)
        if self.mode == 'train':
            return all_pairs[:int(0.8 * total)]
        elif self.mode == 'val':
            return all_pairs[int(0.8 * total):int(0.9 * total)]
        else:  # test
            return all_pairs[int(0.9 * total):]
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict:
        query_id, target_id, label = self.pairs[idx]
        
        # Get product features and types
        query_type = self.type_to_idx[self.bpg.nodes[query_id]['type']]
        target_type = self.type_to_idx[self.bpg.nodes[target_id]['type']]
        
        # Convert features to tensors
        query_features = torch.tensor(self.bpg.nodes[query_id]['features'], dtype=torch.float)
        target_features = torch.tensor(self.bpg.nodes[target_id]['features'], dtype=torch.float)
        
        # Create sample
        sample = {
            'query_ids': query_id,  # Keep as string
            'query_features': query_features,
            'target_features': target_features,
            'query_types': torch.tensor(query_type),
            'positive_types': torch.tensor([target_type if label == 1 else 0]),
            'negative_types': torch.tensor([target_type if label == -1 else 
                                          (target_type + 1) % len(self.type_to_idx)]),
            'positive_items': target_features if label == 1 else torch.randn_like(target_features),
            'negative_items': target_features if label == -1 else torch.randn_like(target_features),
            'label': torch.tensor(label)
        }
        
        return sample

class SyntheticBPGDataset(ComplementaryDataset):
    """Dataset that uses synthetic BPG for training"""
    
    def __init__(self, config: 'Config', mode: str = 'train'):
        from .synthetic_data import SyntheticDataGenerator
        
        # Generate synthetic data
        generator = SyntheticDataGenerator(config)
        bpg = generator.generate_bpg()
        
        super().__init__(bpg, config, mode)

def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function to handle variable-sized data"""
    batch_dict = defaultdict(list)
    
    # First, collect all items
    for sample in batch:
        for key, value in sample.items():
            batch_dict[key].append(value)
    
    # Process each key appropriately
    result_dict = {}
    for key, values in batch_dict.items():
        if key in ['anchor_ids', 'positive_id', 'negative_ids', 'query_ids']:
            # Keep string IDs as lists
            result_dict[key] = values
        elif key == 'anchor_neighbors' and values[0] is not None:
            # Handle neighbor features with padding
            max_neighbors = max(v.size(0) for v in values if v is not None)
            padded_values = []
            for v in values:
                if v is None:
                    # Create zero tensor if no neighbors
                    v = torch.zeros(1, values[0].size(1))
                if v.size(0) < max_neighbors:
                    padding = torch.zeros(max_neighbors - v.size(0), v.size(1))
                    v = torch.cat([v, padding], dim=0)
                padded_values.append(v)
            result_dict[key] = torch.stack(padded_values)
        elif isinstance(values[0], torch.Tensor):
            # Stack other tensors normally
            result_dict[key] = torch.stack(values)
        else:
            # Keep other types as is
            result_dict[key] = values
            
    return result_dict