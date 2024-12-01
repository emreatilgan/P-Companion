import numpy as np
import torch
from typing import Dict, List, Tuple
import random
from collections import defaultdict

from src.data.bpg import BehaviorProductGraph

class SyntheticDataGenerator:
    def __init__(self, config):
        self.config = config
        
        # Synthetic data parameters
        self.num_products = 1000
        self.num_types = 50
        self.num_categories = 10
        self.vocab_size = 1000
        self.title_max_len = 10
        
        # Generate basic product information
        self.products = self._generate_products()
        self.behavior_graph = self._generate_behavior_graph()
        
    def _generate_product_title(self) -> str:
        """Generate a random product title"""
        title_len = random.randint(3, self.title_max_len)
        return ' '.join([f'word_{random.randint(0, self.vocab_size-1)}' 
                        for _ in range(title_len)])
    
    def _generate_products(self) -> Dict[str, Dict]:
        """Generate synthetic product catalog"""
        products = {}
        
        # Create product types with semantic meanings
        product_types = [
            f"type_{i}_{category}"
            for category in ['electronics', 'clothing', 'sports', 'home', 'office']
            for i in range(self.num_types // 5)
        ]
        
        for pid in range(self.num_products):
            # Generate product ID
            product_id = f"P{str(pid).zfill(6)}"
            
            # Assign product type
            product_type = random.choice(product_types)
            category = product_type.split('_')[2]
            
            # Generate product features
            features = torch.randn(self.config.PRODUCT_EMB_DIM)
            
            products[product_id] = {
                'title': self._generate_product_title(),
                'type': product_type,
                'category': category,
                'features': features,
            }
        
        return products
    
    def _generate_behavior_graph(self) -> Dict[str, List[Tuple[str, str]]]:
        """Generate synthetic behavioral data"""
        behaviors = {
            'co_purchase': [],
            'co_view': [],
            'purchase_after_view': []
        }
        
        # Define complementary type patterns
        complementary_patterns = {
            'electronics': ['accessories', 'electronics'],
            'clothing': ['shoes', 'clothing'],
            'sports': ['equipment', 'sports'],
            'home': ['decor', 'home'],
            'office': ['supplies', 'office']
        }
        
        # Generate co-purchase relationships
        for pid, product in self.products.items():
            # Number of complementary products for this item
            num_complements = random.randint(2, 5)
            
            # Get candidate complementary products
            category = product['type'].split('_')[2]
            complementary_categories = complementary_patterns.get(category, [category])
            
            candidate_products = [
                p_id for p_id, p in self.products.items()
                if p['type'].split('_')[2] in complementary_categories
                and p_id != pid
            ]
            
            if candidate_products:
                # Generate co-purchase relationships
                complement_ids = random.sample(
                    candidate_products,
                    min(num_complements, len(candidate_products))
                )
                for comp_id in complement_ids:
                    behaviors['co_purchase'].append((pid, comp_id))
                    
                    # Some co-purchased items are also co-viewed
                    if random.random() < 0.3:
                        behaviors['co_view'].append((pid, comp_id))
                    
                    # Some co-purchased items were viewed before purchase
                    if random.random() < 0.4:
                        behaviors['purchase_after_view'].append((pid, comp_id))
        
        return behaviors
    
    def generate_bpg(self) -> BehaviorProductGraph:
        """Generate BehaviorProductGraph from synthetic data"""
        bpg = BehaviorProductGraph()
        
        # Add nodes (products)
        for product_id, product_data in self.products.items():
            bpg.add_node(product_id, product_data)
        
        # Add edges (behaviors)
        for behavior_type, edges in self.behavior_graph.items():
            for source_id, target_id in edges:
                bpg.add_edge(source_id, target_id, behavior_type)
        
        return bpg
    
    def get_product_info(self) -> Dict[str, Dict]:
        """Return product catalog"""
        return self.products
    
    def get_behavior_data(self) -> Dict[str, List[Tuple[str, str]]]:
        """Return behavior data"""
        return self.behavior_graph