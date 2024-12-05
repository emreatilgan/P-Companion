import torch
import random
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict
from src.data.data_loader import BehaviorProductGraph
import logging
import itertools

class SyntheticDataGenerator:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Synthetic data parameters - reduced for faster experimentation
        self.num_products = 1000  # Reduced from 10000
        self.num_types = 20       # Reduced from 50
        self.vocab_size = 100     # Reduced from 1000
        self.title_max_len = 5    # Reduced from 10
        
        # Behavior generation parameters
        self.sparsity_factor = 0.1  # Only generate edges for 10% of possible pairs
        self.co_view_prob = 0.3     # Reduced from 0.4
        self.pav_given_cv_prob = 0.2  # Reduced from 0.3
        self.cp_given_pav_prob = 0.15  # Reduced from 0.2
        
        # Generate product catalog
        self.products = self._generate_products()
        
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
            
            # Generate product features with category bias
            features = torch.randn(self.config.PRODUCT_EMB_DIM)
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
        
        words = [random.choice(category_words[category])]
        words.extend([f'word_{random.randint(0, self.vocab_size-1)}'
                     for _ in range(random.randint(2, self.title_max_len-1))])
        return ' '.join(words)
    
    def generate_unified_bpg(self) -> 'BehaviorProductGraph':
        """Generate a single BPG with all relationships"""
        from src.data.bpg import BehaviorProductGraph
        bpg = BehaviorProductGraph()
        
        # Add all products
        for product_id, product_data in self.products.items():
            bpg.add_node(product_id, product_data)
        
        # Track statistics
        stats = defaultdict(int)
        similarity_pairs = set()  # (Bcv ∩ Bpv) - Bcp
        complementary_pairs = set()  # Bcp - (Bpv ∪ Bcv)
        
        # Generate all possible pairs
        all_products = list(self.products.keys())
        all_possible_pairs = list(itertools.combinations(all_products, 2))
        
        # Randomly sample pairs based on sparsity factor
        num_pairs_to_generate = int(len(all_possible_pairs) * self.sparsity_factor)
        selected_pairs = random.sample(all_possible_pairs, num_pairs_to_generate)
        
        # Generate behavioral relationships for selected pairs
        for p1, p2 in selected_pairs:
            # First, decide if products are in same category
            same_category = (self.products[p1]['category'] == 
                           self.products[p2]['category'])
            
            # Adjust probabilities based on category relationship
            cv_prob = self.co_view_prob * 1.5 if same_category else self.co_view_prob
            cp_prob = self.cp_given_pav_prob * 0.5 if same_category else self.cp_given_pav_prob
            
            if random.random() < cv_prob:
                bpg.add_edge(p1, p2, 'co_view')
                stats['co_view'] += 1
                
                if random.random() < self.pav_given_cv_prob:
                    bpg.add_edge(p1, p2, 'purchase_after_view')
                    stats['purchase_after_view'] += 1
                    
                    if random.random() >= cp_prob:
                        similarity_pairs.add((p1, p2))
                    else:
                        bpg.add_edge(p1, p2, 'co_purchase')
                        stats['co_purchase'] += 1
            else:
                # Some co-purchases are not co-viewed (complementary products)
                if random.random() < cp_prob:
                    bpg.add_edge(p1, p2, 'co_purchase')
                    stats['co_purchase'] += 1
                    complementary_pairs.add((p1, p2))
        
        # Ensure we have enough pairs of each type
        min_required_pairs = 100  # Minimum number of pairs needed for training
        
        if len(similarity_pairs) < min_required_pairs or len(complementary_pairs) < min_required_pairs:
            self.logger.warning("Not enough pairs generated. Adjusting probabilities and regenerating...")
            # Increase probabilities and try again
            self.sparsity_factor *= 2
            self.co_view_prob *= 1.5
            self.cp_given_pav_prob *= 1.5
            return self.generate_unified_bpg()
        
        # Log statistics
        self.logger.info("Generated behavior graph with:")
        self.logger.info(f"- {stats['co_view']} co-view pairs")
        self.logger.info(f"- {stats['purchase_after_view']} purchase-after-view pairs")
        self.logger.info(f"- {stats['co_purchase']} co-purchase pairs")
        self.logger.info(f"- {len(similarity_pairs)} similarity pairs")
        self.logger.info(f"- {len(complementary_pairs)} complementary pairs")
        
        # Store edge subsets in BPG for easy access
        bpg.similarity_pairs = list(similarity_pairs)
        bpg.complementary_pairs = list(complementary_pairs)
        
        return bpg