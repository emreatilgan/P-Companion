import torch
import os
from datetime import datetime

class Config:
    def __init__(self):
        # Model parameters
        self.PRODUCT_EMB_DIM = 128    # Product embedding dimension (d)
        self.TYPE_EMB_DIM = 64        # Type embedding dimension (L)
        self.HIDDEN_SIZE = 256
        self.NUM_ATTENTION_HEADS = 4
        self.DROPOUT = 0.1
        
        # Training parameters
        self.BATCH_SIZE = 256
        self.LEARNING_RATE = 0.001
        self.NUM_EPOCHS = 20
        self.MARGIN = 1.0             # Margin parameters (λ, ϵ)
        self.ALPHA = 0.8              # Trade-off parameter between type and item prediction
        self.NUM_COMP_TYPES = 3       # Number of complementary types to predict
        
        # Data parameters
        self.NUM_TYPES = 34800        # Number of product types (from paper)
        self.NEGATIVE_SAMPLE_RATIO = 1.0
        
        # Paths
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.MODEL_DIR = os.path.join(self.BASE_DIR, 'models', f'run_{timestamp}')
        
        # Device configuration
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model saving/loading
        self.SAVE_FREQ = 5  # Save model every N epochs
        self.CHECKPOINT_DIR = os.path.join(self.MODEL_DIR, 'checkpoints')
        self.BEST_MODEL_PATH = os.path.join(self.MODEL_DIR, 'best_model.pth')
        
        # Logging
        self.LOG_DIR = os.path.join(self.MODEL_DIR, 'logs')
        self.LOG_FREQ = 100  # Log every N batches
        
        # Create necessary directories
        for directory in [self.DATA_DIR, self.MODEL_DIR, self.CHECKPOINT_DIR, self.LOG_DIR]:
            os.makedirs(directory, exist_ok=True)