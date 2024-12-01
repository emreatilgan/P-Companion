import torch

class Config:
    # Model parameters
    PRODUCT_EMB_DIM = 128  # Product embedding dimension (d)
    TYPE_EMB_DIM = 64     # Type embedding dimension (L)
    
    # Training parameters
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    MARGIN = 1.0          # Margin parameters (λ, ϵ)
    ALPHA = 0.8           # Trade-off parameter between type and item prediction
    
    # Model architecture
    HIDDEN_SIZE = 256
    NUM_ATTENTION_HEADS = 4
    DROPOUT = 0.1
    
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')