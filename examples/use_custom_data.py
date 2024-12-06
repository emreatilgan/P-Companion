# examples/use_custom_data.py
import logging
import os
from pathlib import Path

from config import Config
from src.data.custom_data_processor import CustomDataProcessor
from src.data.data_loader import SimilarityDataset, ComplementaryDataset
from scripts.pretrain_product2vec import pretrain_product2vec
from train import train
from torch.utils.data import DataLoader


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize config
    config = Config()
    
    # Initialize data processor
    processor = CustomDataProcessor(config)
    
    # Define data paths
    data_dir = Path("your_data_directory")  # Change this
    paths = {
        "product_data": data_dir / "products.csv",
        "co_view_data": data_dir / "co_views.csv",
        "co_purchase_data": data_dir / "co_purchases.csv",
        "purchase_after_view_data": data_dir / "purchase_after_views.csv",
        "product_features": data_dir / "features.csv"  # Optional
    }
    
    # Validate data files
    logger.info("Validating data files...")
    processor.validate_data_files(**paths)
    
    # Process data into BPG
    logger.info("Processing data into BehaviorProductGraph...")
    bpg = processor.process_data(**paths)
    
    # Create datasets
    similarity_dataset = SimilarityDataset(bpg, config)
    
    # Pretrain Product2Vec
    logger.info("Starting Product2Vec pretraining...")
    pretrained_embeddings = pretrain_product2vec(config, similarity_dataset)
    
    # Create P-Companion datasets
    train_dataset = ComplementaryDataset(bpg, config, mode='train')
    val_dataset = ComplementaryDataset(bpg, config, mode='val')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    # Train model
    logger.info("Starting P-Companion training...")
    train(config, train_loader, val_loader, pretrained_embeddings)

if __name__ == "__main__":
    main()