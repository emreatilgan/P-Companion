import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import logging
import os
from typing import Dict

from src.models.product2vec import Product2Vec
from src.data.data_loader import collate_fn  # Import collate_fn from data_loader

def pretrain_product2vec(config, similarity_dataset) -> Dict[str, torch.Tensor]:
    """Pretrain Product2Vec model and save embeddings
    
    Args:
        config: Configuration object
        similarity_dataset: Dataset containing similarity pairs
        
    Returns:
        Dictionary mapping product IDs to their embeddings
    """
    logger = logging.getLogger(__name__)
    
    # Create dataloader with imported collate_fn
    dataloader = DataLoader(
        similarity_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    # Initialize model and optimizer
    model = Product2Vec(config).to(config.DEVICE)
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Train model
    embeddings_dict = model.train_model(
        train_loader=dataloader,
        optimizer=optimizer,
        num_epochs=config.PRODUCT2VEC_EPOCHS
    )
    
    # Save model and embeddings
    save_path = os.path.join(config.MODEL_DIR, 'product2vec.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'embeddings': embeddings_dict,
        'type_to_idx': similarity_dataset.bpg.type_to_idx if hasattr(similarity_dataset.bpg, 'type_to_idx') else None
    }, save_path)
    
    logger.info(f"Saved pretrained Product2Vec model and embeddings to {save_path}")
    
    return embeddings_dict