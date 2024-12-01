import pytest
import torch
from src.data.synthetic_data import SyntheticDataGenerator
from src.models.p_companion import PCompanion
from config import Config

def test_synthetic_data_generation():
    config = Config()
    generator = SyntheticDataGenerator(config)
    
    # Test product generation
    products = generator.get_product_info()
    assert len(products) > 0
    
    # Check product structure
    sample_product = next(iter(products.values()))
    assert 'title' in sample_product
    assert 'type' in sample_product
    assert 'category' in sample_product
    assert 'features' in sample_product
    assert isinstance(sample_product['features'], torch.Tensor)
    
    # Test behavior graph generation
    behaviors = generator.get_behavior_data()
    assert 'co_purchase' in behaviors
    assert 'co_view' in behaviors
    assert 'purchase_after_view' in behaviors
    
    # Test BPG generation
    bpg = generator.generate_bpg()
    assert len(bpg.nodes) > 0
    assert len(bpg.edges['co_purchase']) > 0

def test_model_with_synthetic_data():
    config = Config()
    generator = SyntheticDataGenerator(config)
    bpg = generator.generate_bpg()
    
    # Create model
    model = PCompanion(config)
    
    # Create sample batch
    sample_batch = {
        'query_features': torch.randn(2, config.PRODUCT_EMB_DIM),
        'query_types': torch.tensor([0, 1]),
        'target_features': torch.randn(2, config.PRODUCT_EMB_DIM),
        'positive_types': torch.tensor([[1], [2]]),
        'negative_types': torch.tensor([[3], [4]]),
        'positive_items': torch.randn(2, config.PRODUCT_EMB_DIM),
        'negative_items': torch.randn(2, config.PRODUCT_EMB_DIM)
    }
    
    # Test forward pass
    outputs = model(sample_batch)
    assert 'projected_embeddings' in outputs
    assert 'complementary_types' in outputs
    assert 'type_similarities' in outputs
    
    # Test loss computation
    loss = model.compute_loss(sample_batch, outputs)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad

if __name__ == "__main__":
    pytest.main([__file__])