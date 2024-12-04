import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import logging
import os
from typing import Dict
from collections import defaultdict

def collate_fn(batch):
    """Custom collate function for Product2Vec training"""
    batch_dict = defaultdict(list)
    
    for sample in batch:
        for key, value in sample.items():
            if key == 'anchor_neighbors':
                if value is not None:
                    batch_dict[key].append(value)
            else:
                batch_dict[key].append(value)
    
    # Stack tensors
    result = {}
    for key, values in batch_dict.items():
        if key == 'anchor_ids' or key == 'positive_id' or key == 'negative_ids':
            result[key] = values
        elif key == 'anchor_neighbors' and values:
            # Pad neighbor sequences to max length in batch
            max_neighbors = max(v.size(0) for v in values)
            padded_values = []
            for v in values:
                if v.size(0) < max_neighbors:
                    padding = torch.zeros(max_neighbors - v.size(0), v.size(1))
                    v = torch.cat([v, padding], dim=0)
                padded_values.append(v)
            result[key] = torch.stack(padded_values)
        else:
            result[key] = torch.stack(values)
    
    return result

def pretrain_product2vec(config):
    """Pretrain Product2Vec model and save embeddings"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Generate synthetic data
    from src.data.synthetic_data import SyntheticDataGenerator
    generator = SyntheticDataGenerator(config)
    bpg = generator.generate_bpg()
    
    # Create similarity dataset
    from src.data.data_loader import SimilarityDataset
    dataset = SimilarityDataset(bpg, config)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    # Initialize model and optimizer
    from src.models.product2vec import Product2Vec
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
        'embeddings': embeddings_dict,
        'product_to_idx': dataset.bpg.product_to_idx if hasattr(dataset.bpg, 'product_to_idx') else None
    }, os.path.join(config.MODEL_DIR, 'product2vec.pth'))
    
    logger.info(f"Saved pretrained Product2Vec model and embeddings")
    
    return embeddings_dict