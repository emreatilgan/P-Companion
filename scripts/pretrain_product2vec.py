import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import logging
import os
from typing import Dict

from config import Config
from src.models.product2vec import Product2Vec, SimilarityDataset
from src.data.synthetic_data import SyntheticDataGenerator

def pretrain_product2vec(config: Config) -> Dict[str, torch.Tensor]:
    """Pretrain Product2Vec model and save embeddings"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Generate synthetic data
    generator = SyntheticDataGenerator(config)
    bpg = generator.generate_bpg()
    
    # Create similarity dataset
    dataset = SimilarityDataset(bpg, config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    
    # Initialize model and optimizer
    model = Product2Vec(config)
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Train model
    embeddings_dict = model.train_model(
        train_loader=dataloader,
        optimizer=optimizer,
        num_epochs=config.PRODUCT2VEC_EPOCHS
    )
    
    # Save model and embeddings
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'embeddings': embeddings_dict
    }, os.path.join(config.MODEL_DIR, 'product2vec.pth'))
    
    logger.info(f"Saved pretrained Product2Vec model and embeddings")
    
    return embeddings_dict

if __name__ == "__main__":
    config = Config()
    pretrain_product2vec(config)