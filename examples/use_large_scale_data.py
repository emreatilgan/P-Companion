import logging
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from config import Config
from src.data.large_scale_processor import LargeScaleDataProcessor
from src.data.data_loader import SimilarityDataset, ComplementaryDataset
from scripts.pretrain_product2vec import pretrain_product2vec
from train import train

class LargeScaleConfig(Config):
    def __init__(self):
        super().__init__()
        
        # Data processing parameters
        self.CHUNK_SIZE = 10000  # Number of products per chunk
        self.MAX_MEMORY = "32GB"  # Maximum memory usage
        self.NUM_WORKERS = 4      # Number of parallel workers
        self.TEMP_DIR = "temp"    # Directory for temporary files
        
        # Training parameters for large datasets
        self.BATCH_SIZE = 512     # Increased batch size
        self.ACCUMULATION_STEPS = 4  # Gradient accumulation steps

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize config
    config = LargeScaleConfig()
    
    # Initialize processor
    processor = LargeScaleDataProcessor(config)
    
    # Define data paths
    data_dir = Path("your_data_directory")
    paths = {
        "product_data": data_dir / "products.csv",
        "co_view_data": data_dir / "co_views.csv",
        "co_purchase_data": data_dir / "co_purchases.csv",
        "purchase_after_view_data": data_dir / "purchase_after_views.csv",
        "product_features": data_dir / "features.csv"
    }
    
    # Estimate memory requirements
    logger.info("Estimating memory requirements...")
    memory_stats = processor.estimate_memory_requirements(**paths)
    for name, mem_mb in memory_stats.items():
        logger.info(f"{name}: {mem_mb:.2f} MB")
    
    # Validate data files
    logger.info("Validating data files...")
    processor.validate_data_files_chunked(**paths)
    
    # Process data into BPG with chunking
    logger.info("Processing data into BehaviorProductGraph...")
    bpg = processor.process_data(
        **paths,
        temp_dir=config.TEMP_DIR
    )
    
    # Create datasets with lazy loading
    similarity_dataset = SimilarityDataset(bpg, config)
    
    # Use DataLoader with appropriate settings for large data
    train_loader = DataLoader(
        similarity_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2
    )
    
    # Pretrain Product2Vec with gradient accumulation
    logger.info("Starting Product2Vec pretraining...")
    pretrained_embeddings = pretrain_product2vec(
        config, 
        similarity_dataset,
        accumulation_steps=config.ACCUMULATION_STEPS
    )
    
    # Create P-Companion datasets
    train_dataset = ComplementaryDataset(bpg, config, mode='train')
    val_dataset = ComplementaryDataset(bpg, config, mode='val')
    
    # Create dataloaders with appropriate settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2
    )
    
    # Train model with gradient accumulation
    logger.info("Starting P-Companion training...")
    train(
        config,
        train_loader,
        val_loader,
        pretrained_embeddings,
        accumulation_steps=config.ACCUMULATION_STEPS
    )
    
    # Clean up temporary files
    logger.info("Cleaning up temporary files...")
    for file in Path(config.TEMP_DIR).glob("*"):
        file.unlink()
    Path(config.TEMP_DIR).rmdir()

if __name__ == "__main__":
    main()