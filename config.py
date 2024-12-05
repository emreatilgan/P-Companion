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
        
        # Product2Vec pretraining parameters
        self.PRODUCT2VEC_EPOCHS = 10  # Number of epochs for pretraining
        self.MARGIN = 1.0             # Margin for triplet loss
        self.NEG_SAMPLES = 5          # Number of negative samples per positive pair
        
        # Training parameters
        self.BATCH_SIZE = 256
        self.LEARNING_RATE = 0.001
        self.NUM_EPOCHS = 20
        self.ALPHA = 0.8              # Trade-off parameter between type and item prediction
        self.NUM_COMP_TYPES = 3       # Number of complementary types to predict
        
        # Data parameters
        self.NUM_TYPES = 34800        # Number of product types
        self.SYNTHETIC_NUM_PRODUCTS = 10000  # Number of synthetic products to generate
        self.NEGATIVE_SAMPLE_RATIO = 1.0
        
        # Synthetic data generation parameters
        self.MIN_NEIGHBORS = 3        # Minimum number of neighbors per product
        self.MAX_NEIGHBORS = 10       # Maximum number of neighbors per product
        self.CO_VIEW_RATIO = 0.4      # Ratio of product pairs that are co-viewed
        self.PAV_RATIO = 0.3          # Ratio of co-viewed products that are purchased after viewing
        self.CO_PURCHASE_RATIO = 0.2  # Ratio of product pairs that are co-purchased
        
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
        
        # Create necessary directories
        for directory in [self.DATA_DIR, self.MODEL_DIR, self.CHECKPOINT_DIR, self.LOG_DIR]:
            os.makedirs(directory, exist_ok=True)
            
    def __str__(self):
        """String representation of config for logging"""
        attrs = vars(self)
        return '\n'.join(f'{key}: {value}' for key, value in attrs.items())