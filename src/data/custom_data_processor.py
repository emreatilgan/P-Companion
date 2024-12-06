import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict
from src.data.bpg import BehaviorProductGraph
from src.data.feature_extraction import ProductFeatureExtractor

class CustomDataProcessor:
    """Process custom dataset into BehaviorProductGraph format"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.feature_extractor = ProductFeatureExtractor(config)
        
    def process_data(
        self,
        product_data_path: str,
        co_view_data_path: Optional[str] = None,
        co_purchase_data_path: Optional[str] = None,
        purchase_after_view_data_path: Optional[str] = None,
        product_features_path: Optional[str] = None
    ) -> 'BehaviorProductGraph':
        """
        Process custom data files into BehaviorProductGraph
        
        Args:
            product_data_path: Path to product catalog CSV
                Required columns: product_id, title, type, category
            co_view_data_path: Path to co-view pairs CSV
                Required columns: product_id_1, product_id_2
            co_purchase_data_path: Path to co-purchase pairs CSV
                Required columns: product_id_1, product_id_2
            purchase_after_view_data_path: Path to purchase-after-view pairs CSV
                Required columns: product_id_1, product_id_2
            product_features_path: Path to precomputed product features
                If not provided, features will be generated from text
                
        Returns:
            BehaviorProductGraph object
        """
        from src.data.bpg import BehaviorProductGraph
        
        self.logger.info("Starting custom data processing...")
        
        # Load product catalog
        products_df = pd.read_csv(product_data_path)
        required_cols = ['product_id', 'title', 'type', 'category']
        missing_cols = [col for col in required_cols if col not in products_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in product data: {missing_cols}")
            
        # Create BPG
        bpg = BehaviorProductGraph()
        
        # Process products
        for _, row in products_df.iterrows():
            # Get or generate features
            if product_features_path:
                features = self._load_product_features(
                    row['product_id'], 
                    product_features_path
                )
            else:
                features = self._generate_product_features(row)
                
            # Add to BPG
            bpg.add_node(
                str(row['product_id']),
                {
                    'title': row['title'],
                    'type': row['type'],
                    'category': row['category'],
                    'features': features
                }
            )
            
        # Process behavioral data
        self._process_behavior_data(bpg, co_view_data_path, 'co_view')
        self._process_behavior_data(bpg, co_purchase_data_path, 'co_purchase')
        self._process_behavior_data(bpg, purchase_after_view_data_path, 'purchase_after_view')
        
        # Log statistics
        self._log_bpg_stats(bpg)
        
        return bpg
    
    def _load_product_features(
        self,
        product_id: str,
        features_path: str
    ) -> torch.Tensor:
        """Load precomputed product features"""
        features_df = pd.read_csv(features_path)
        features = features_df[features_df['product_id'] == product_id].iloc[0]
        feature_cols = [col for col in features_df.columns if col != 'product_id']
        return torch.tensor(features[feature_cols].values, dtype=torch.float)
    
    def _generate_product_features(self, product_row: pd.Series) -> torch.Tensor:
        """Generate product features from text data"""
        return self.feature_extractor.extract_features(product_row)
    
    def _process_behavior_data(
        self,
        bpg: 'BehaviorProductGraph',
        data_path: Optional[str],
        behavior_type: str
    ) -> None:
        """Process behavioral relationship data"""
        if data_path is None:
            self.logger.warning(f"No {behavior_type} data provided")
            return
            
        pairs_df = pd.read_csv(data_path)
        required_cols = ['product_id_1', 'product_id_2']
        missing_cols = [col for col in required_cols if col not in pairs_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {behavior_type} data: {missing_cols}")
            
        for _, row in pairs_df.iterrows():
            pid1, pid2 = str(row['product_id_1']), str(row['product_id_2'])
            if pid1 in bpg.nodes and pid2 in bpg.nodes:
                bpg.add_edge(pid1, pid2, behavior_type)
            
    def _log_bpg_stats(self, bpg: 'BehaviorProductGraph') -> None:
        """Log statistics about the processed graph"""
        self.logger.info("BPG Statistics:")
        self.logger.info(f"Number of products: {len(bpg.nodes)}")
        for edge_type in bpg.edges:
            self.logger.info(f"Number of {edge_type} pairs: {len(bpg.edges[edge_type])}")
            
    def validate_data_files(
        self,
        product_data_path: str,
        co_view_data_path: Optional[str] = None,
        co_purchase_data_path: Optional[str] = None,
        purchase_after_view_data_path: Optional[str] = None,
        product_features_path: Optional[str] = None
    ) -> None:
        """Validate data files before processing"""
        # Check product data
        if not Path(product_data_path).exists():
            raise FileNotFoundError(f"Product data file not found: {product_data_path}")
            
        products_df = pd.read_csv(product_data_path)
        required_cols = ['product_id', 'title', 'type', 'category']
        missing_cols = [col for col in required_cols if col not in products_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in product data: {missing_cols}")
            
        # Check features file
        if product_features_path:
            if not Path(product_features_path).exists():
                raise FileNotFoundError(f"Features file not found: {product_features_path}")
                
            features_df = pd.read_csv(product_features_path)
            if 'product_id' not in features_df.columns:
                raise ValueError("Features file must have 'product_id' column")
                
        # Check behavior data files
        for path, name in [
            (co_view_data_path, 'co_view'),
            (co_purchase_data_path, 'co_purchase'),
            (purchase_after_view_data_path, 'purchase_after_view')
        ]:
            if path:
                if not Path(path).exists():
                    raise FileNotFoundError(f"{name} data file not found: {path}")
                    
                pairs_df = pd.read_csv(path)
                required_cols = ['product_id_1', 'product_id_2']
                missing_cols = [col for col in required_cols if col not in pairs_df.columns]
                if missing_cols:
                    raise ValueError(f"Missing required columns in {name} data: {missing_cols}")