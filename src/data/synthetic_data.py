import torch
import random
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict
from src.data.data_loader import BehaviorProductGraph

class SyntheticDataGenerator:
    def __init__(self, config):
        self.config = config
        
        # Synthetic data parameters
        self.num_products = config.SYNTHETIC_NUM_PRODUCTS
        self.num_types = 50
        self.vocab_size = 1000
        self.title_max_len = 10
        
        # Behavior generation parameters
        self.avg_neighbors = 5  # Average number of neighbors per product
        self.co_view_ratio = 0.4  # Ratio of product pairs that are co-viewed
        self.pav_ratio = 0.3  # Ratio of co-viewed products that are purchased after viewing
        self.co_purchase_ratio = 0.2  # Ratio of product pairs that are co-purchased
        
        # Generate data
        self.products = self._generate_products()
        self.behavior_graph = self._generate_behavior_graph()
        
    def _generate_products(self) -> Dict[str, Dict]:
        """Generate synthetic product catalog"""
        products = {}
        
        # Create product types with semantic meanings
        product_types = [
            f"type_{i}_{category}"
            for category in ['electronics', 'clothing', 'sports', 'home', 'office']
            for i in range(self.num_types // 5)
        ]
        
        # Generate products
        for pid in range(self.num_products):
            product_id = f"P{str(pid).zfill(6)}"
            
            # Assign product type
            product_type = random.choice(product_types)
            category = product_type.split('_')[2]
            
            # Generate product features
            features = torch.randn(self.config.PRODUCT_EMB_DIM)
            # Add some category-based bias to features
            category_idx = ['electronics', 'clothing', 'sports', 'home', 'office'].index(category)
            features[category_idx * 20:(category_idx + 1) * 20] += 1.0
            
            products[product_id] = {
                'title': self._generate_title(category),
                'type': product_type,
                'category': category,
                'features': features,
            }
        
        return products
    
    def _generate_title(self, category: str) -> str:
        """Generate a category-influenced product title"""
        category_words = {
            'electronics': ['device', 'gadget', 'tech', 'smart', 'digital'],
            'clothing': ['shirt', 'pants', 'jacket', 'dress', 'wear'],
            'sports': ['gear', 'equipment', 'training', 'athletic', 'sport'],
            'home': ['decor', 'furniture', 'home', 'living', 'comfort'],
            'office': ['desk', 'chair', 'office', 'work', 'professional']
        }
        
        # Generate title with category-specific words
        words = [random.choice(category_words[category])]
        words.extend([f'word_{random.randint(0, self.vocab_size-1)}'
                     for _ in range(random.randint(2, self.title_max_len-1))])
        return ' '.join(words)
    
    def _generate_behavior_graph(self) -> Dict[str, List[Tuple[str, str]]]:
        """Generate synthetic behavioral data ensuring sufficient similarity pairs"""
        behaviors = {
            'co_view': [],
            'purchase_after_view': [],
            'co_purchase': []
        }
        
        all_products = list(self.products.keys())
        
        # Generate co-view relationships
        num_co_views = int(self.num_products * self.avg_neighbors * self.co_view_ratio)
        for _ in range(num_co_views):
            source = random.choice(all_products)
            target = random.choice([p for p in all_products if p != source])
            
            if (source, target) not in behaviors['co_view']:
                behaviors['co_view'].append((source, target))
                
                # Some co-views lead to purchase-after-view
                if random.random() < self.pav_ratio:
                    behaviors['purchase_after_view'].append((source, target))
                    
                    # Some purchase-after-views become co-purchases
                    if random.random() < self.co_purchase_ratio:
                        behaviors['co_purchase'].append((source, target))
        
        # Ensure we have enough similarity pairs
        num_similarity_pairs = len(set(behaviors['co_view']) & 
                                 set(behaviors['purchase_after_view']) - 
                                 set(behaviors['co_purchase']))
        
        print(f"Generated behavior graph with:")
        print(f"- {len(behaviors['co_view'])} co-view pairs")
        print(f"- {len(behaviors['purchase_after_view'])} purchase-after-view pairs")
        print(f"- {len(behaviors['co_purchase'])} co-purchase pairs")
        print(f"- {num_similarity_pairs} similarity pairs")
        
        return behaviors
    
    def generate_bpg(self) -> 'BehaviorProductGraph':
        """Generate BehaviorProductGraph from synthetic data"""
        from src.data.bpg import BehaviorProductGraph
        
        bpg = BehaviorProductGraph()
        
        # Add nodes
        for product_id, product_data in self.products.items():
            bpg.add_node(product_id, product_data)
        
        # Add edges
        for behavior_type, edges in self.behavior_graph.items():
            for source_id, target_id in edges:
                bpg.add_edge(source_id, target_id, behavior_type)
        
        return bpg