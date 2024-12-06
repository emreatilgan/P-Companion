import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator
import logging
from collections import defaultdict
import dask.dataframe as dd
import h5py
from tqdm import tqdm
from src.data.bpg import BehaviorProductGraph

class LargeScaleDataProcessor:
    """Process large-scale datasets into BehaviorProductGraph format"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.chunk_size = config.CHUNK_SIZE  # e.g., 10000
        
    def process_data(
        self,
        product_data_path: str,
        co_view_data_path: Optional[str] = None,
        co_purchase_data_path: Optional[str] = None,
        purchase_after_view_data_path: Optional[str] = None,
        product_features_path: Optional[str] = None,
        temp_dir: Optional[str] = None
    ) -> 'BehaviorProductGraph':
        """
        Process large-scale data files into BehaviorProductGraph
        
        Args:
            product_data_path: Path to product catalog
            co_view_data_path: Path to co-view pairs
            co_purchase_data_path: Path to co-purchase pairs
            purchase_after_view_data_path: Path to purchase-after-view pairs
            product_features_path: Path to precomputed features
            temp_dir: Directory for temporary files
            
        Uses:
            - Dask for out-of-memory dataframe processing
            - HDF5 for storing intermediate results
            - Chunked processing for behavioral data
        """
        from src.data.bpg import BehaviorProductGraph
        
        self.logger.info("Starting large-scale data processing...")
        
        # Create temporary directory if needed
        temp_dir = temp_dir or Path("temp")
        Path(temp_dir).mkdir(exist_ok=True)
        
        # Process products in chunks
        bpg = BehaviorProductGraph()
        self._process_products_chunked(
            product_data_path,
            product_features_path,
            bpg,
            temp_dir
        )
        
        # Process behavioral data in chunks
        for path, behavior_type in [
            (co_view_data_path, 'co_view'),
            (co_purchase_data_path, 'co_purchase'),
            (purchase_after_view_data_path, 'purchase_after_view')
        ]:
            if path:
                self._process_behavior_chunked(path, behavior_type, bpg, temp_dir)
        
        return bpg
    
    def _process_products_chunked(
        self,
        product_data_path: str,
        features_path: Optional[str],
        bpg: 'BehaviorProductGraph',
        temp_dir: str
    ) -> None:
        """Process product data in chunks"""
        # Use Dask to read CSV in chunks
        products_ddf = dd.read_csv(product_data_path)
        
        # Process chunks
        for chunk_idx, chunk_df in enumerate(products_ddf.partitions):
            chunk_df = chunk_df.compute()
            
            # Process features if available
            if features_path:
                features_chunk = self._get_features_chunk(
                    features_path,
                    chunk_df['product_id'].tolist(),
                    chunk_idx,
                    temp_dir
                )
            
            # Add products to BPG
            for idx, row in chunk_df.iterrows():
                features = (features_chunk[idx] 
                          if features_path else 
                          self._generate_product_features(row))
                
                bpg.add_node(
                    str(row['product_id']),
                    {
                        'title': row['title'],
                        'type': row['type'],
                        'category': row['category'],
                        'features': features
                    }
                )
            
            # Log progress
            self.logger.info(f"Processed product chunk {chunk_idx + 1}")
    
    def _process_behavior_chunked(
        self,
        data_path: str,
        behavior_type: str,
        bpg: 'BehaviorProductGraph',
        temp_dir: str
    ) -> None:
        """Process behavioral data in chunks"""
        # Use Dask to read CSV in chunks
        pairs_ddf = dd.read_csv(data_path)
        
        # Process chunks
        for chunk_idx, chunk_df in enumerate(pairs_ddf.partitions):
            chunk_df = chunk_df.compute()
            
            # Add edges to BPG
            for _, row in chunk_df.iterrows():
                pid1, pid2 = str(row['product_id_1']), str(row['product_id_2'])
                if pid1 in bpg.nodes and pid2 in bpg.nodes:
                    bpg.add_edge(pid1, pid2, behavior_type)
            
            # Log progress
            self.logger.info(f"Processed {behavior_type} chunk {chunk_idx + 1}")
    
    def _get_features_chunk(
        self,
        features_path: str,
        product_ids: List[str],
        chunk_idx: int,
        temp_dir: str
    ) -> torch.Tensor:
        """Get features for a chunk of products"""
        # Use HDF5 for efficient feature storage and retrieval
        features_file = Path(temp_dir) / f"features_chunk_{chunk_idx}.h5"
        
        if features_file.exists():
            with h5py.File(features_file, 'r') as f:
                return torch.tensor(f['features'][:])
        
        # Read features for products in chunk
        features_df = dd.read_csv(features_path)
        chunk_features = features_df[
            features_df['product_id'].isin(product_ids)
        ].compute()
        
        # Save to HDF5
        feature_cols = [col for col in chunk_features.columns 
                       if col != 'product_id']
        features = chunk_features[feature_cols].values
        
        with h5py.File(features_file, 'w') as f:
            f.create_dataset('features', data=features)
        
        return torch.tensor(features)

    def estimate_memory_requirements(
        self,
        product_data_path: str,
        co_view_data_path: Optional[str] = None,
        co_purchase_data_path: Optional[str] = None,
        purchase_after_view_data_path: Optional[str] = None
    ) -> Dict[str, float]:
        """Estimate memory requirements for processing"""
        memory_stats = {}
        
        # Count products
        products_ddf = dd.read_csv(product_data_path)
        num_products = len(products_ddf)
        memory_stats['products'] = (
            num_products * self.config.PRODUCT_EMB_DIM * 4  # 4 bytes per float
        ) / (1024 ** 2)  # Convert to MB
        
        # Count behavioral pairs
        for path, name in [
            (co_view_data_path, 'co_view'),
            (co_purchase_data_path, 'co_purchase'),
            (purchase_after_view_data_path, 'purchase_after_view')
        ]:
            if path:
                pairs_ddf = dd.read_csv(path)
                num_pairs = len(pairs_ddf)
                memory_stats[name] = (num_pairs * 16) / (1024 ** 2)  # 2 ids * 8 bytes
        
        return memory_stats

    def validate_data_files_chunked(
        self,
        product_data_path: str,
        co_view_data_path: Optional[str] = None,
        co_purchase_data_path: Optional[str] = None,
        purchase_after_view_data_path: Optional[str] = None,
        product_features_path: Optional[str] = None
    ) -> None:
        """Validate data files in chunks without loading entire files"""
        self.logger.info("Validating data files...")
        
        # Check product data schema
        products_ddf = dd.read_csv(product_data_path)
        required_cols = ['product_id', 'title', 'type', 'category']
        missing_cols = [col for col in required_cols 
                       if col not in products_ddf.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in product data: {missing_cols}")
        
        # Check behavioral data schemas
        for path, name in [
            (co_view_data_path, 'co_view'),
            (co_purchase_data_path, 'co_purchase'),
            (purchase_after_view_data_path, 'purchase_after_view')
        ]:
            if path:
                pairs_ddf = dd.read_csv(path)
                required_cols = ['product_id_1', 'product_id_2']
                missing_cols = [col for col in required_cols 
                              if col not in pairs_ddf.columns]
                if missing_cols:
                    raise ValueError(f"Missing columns in {name}: {missing_cols}")
        
        # Check features if provided
        if product_features_path:
            features_ddf = dd.read_csv(product_features_path)
            if 'product_id' not in features_ddf.columns:
                raise ValueError("Features must have 'product_id' column")
                
        self.logger.info("Data validation completed")