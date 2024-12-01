class Constants:
    # Edge types in BPG
    CO_PURCHASE = 'co_purchase'
    CO_VIEW = 'co_view'
    PURCHASE_AFTER_VIEW = 'purchase_after_view'
    
    # Special tokens
    PAD_TOKEN = '[PAD]'
    UNK_TOKEN = '[UNK]'
    
    # Minimum frequency thresholds
    MIN_PRODUCT_FREQ = 2
    MIN_TYPE_FREQ = 5
    
    # Model constants
    EPSILON = 1e-8
    MAX_SEQUENCE_LENGTH = 512
    
    # Evaluation constants
    TOP_K_VALUES = [1, 3, 10, 60]
    
    # Data processing
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1