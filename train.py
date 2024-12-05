import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import logging
from tqdm import tqdm
import os

from config import Config
from src.models.p_companion import PCompanion
from src.data.data_loader import SyntheticBPGDataset, collate_fn
from src.utils.metrics import Metrics
from scripts.pretrain_product2vec import pretrain_product2vec

def train(config, train_loader, val_loader, pretrained_embeddings):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize model with pretrained embeddings
    model = PCompanion(config, pretrained_embeddings).to(config.DEVICE)
    
    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Training loop
    best_hit10 = 0.0
    for epoch in range(config.NUM_EPOCHS):
        # Training phase
        model.train()
        total_loss = 0.0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move tensors to device
                batch = {
                    k: v.to(config.DEVICE) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                # Forward pass
                outputs = model(batch)
                loss = model.compute_loss(batch, outputs)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update progress bar
                total_loss += loss.item()
                pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})
        
        # Validation phase
        metrics = Metrics.evaluate_model(model, val_loader, config.DEVICE)
        
        # Log metrics
        logger.info(f"Epoch {epoch+1} Validation Metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
        
        # Save best model
        if metrics['hit@10'] > best_hit10:
            best_hit10 = metrics['hit@10']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
            }, os.path.join(config.MODEL_DIR, 'best_model.pth'))
            
        logger.info(f"Best Hit@10: {best_hit10:.4f}")

def main():
    # Initialize config
    config = Config()
    
    # Create model directory if it doesn't exist
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # First, pretrain Product2Vec
    logger = logging.getLogger(__name__)
    logger.info("Starting Product2Vec pretraining...")
    pretrained_embeddings = pretrain_product2vec(config)
    logger.info("Product2Vec pretraining completed")
    
    # Create synthetic datasets
    train_dataset = SyntheticBPGDataset(config, mode='train')
    val_dataset = SyntheticBPGDataset(config, mode='val')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Start P-Companion training
    logger.info("Starting P-Companion training...")
    train(config, train_loader, val_loader, pretrained_embeddings)
    logger.info("Training completed")

if __name__ == "__main__":
    main()