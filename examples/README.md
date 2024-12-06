# Using Custom Datasets

This guide explains how to use your own dataset with the P-Companion implementation.

## Required Data Files

### 1. Product Catalog (Required)
File: `products.csv`
```csv
product_id,title,type,category
1,"Product A","type_1_electronics","electronics"
2,"Product B","type_2_clothing","clothing"
...
```
Required columns:
- product_id: Unique identifier for each product
- title: Product title/name
- type: Product type (format: type_X_category)
- category: Product category

### 2. Behavioral Data (At least one required)

#### Co-View Relationships
File: `co_views.csv`
```csv
product_id_1,product_id_2
1,2
1,3
...
```

#### Co-Purchase Relationships
File: `co_purchases.csv`
```csv
product_id_1,product_id_2
1,5
2,6
...
```

#### Purchase-After-View Relationships
File: `purchase_after_views.csv`
```csv
product_id_1,product_id_2
1,4
2,5
...
```

### 3. Product Features (Optional)
File: `features.csv`
```csv
product_id,feature1,feature2,...,featureN
1,0.1,0.2,...,0.5
2,0.3,0.4,...,0.6
...
```
- If not provided, features will be randomly generated
- Feature dimensionality must match config.PRODUCT_EMB_DIM

## Usage Example

```python
from src.data.custom_data_processor import CustomDataProcessor
from config import Config

# Initialize
config = Config()
processor = CustomDataProcessor(config)

# Process data
bpg = processor.process_data(
    product_data_path="path/to/products.csv",
    co_view_data_path="path/to/co_views.csv",
    co_purchase_data_path="path/to/co_purchases.csv",
    purchase_after_view_data_path="path/to/purchase_after_views.csv",
    product_features_path="path/to/features.csv"  # Optional
)
```

See `examples/use_custom_data.py` for a complete example.

## Data Requirements

1. Product Types:
   - Should follow format: type_X_category
   - X can be any identifier
   - category should match product's category

2. Behavioral Data:
   - Product IDs must exist in product catalog
   - Relationships should be undirected (A->B implies B->A)
   - At least one type of behavioral data is required

3. Product Features:
   - Must match PRODUCT_EMB_DIM in config
   - Should be normalized/standardized
   - If not provided, random features will be used

## Best Practices

1. Data Preparation:
   - Clean and normalize text data
   - Remove duplicates from behavioral data
   - Ensure consistent product IDs across files

2. Data Validation:
   - Use `validate_data_files()` before processing
   - Check for missing product IDs
   - Verify feature dimensionality

3. Memory Management:
   - For large datasets, consider batch processing
   - Monitor memory usage during feature generation
   - Use appropriate batch sizes in config

## Troubleshooting

Common issues and solutions:
1. Missing product IDs in behavioral data
2. Inconsistent feature dimensions
3. Memory errors with large datasets
4. Invalid product type formats

For any issues, check the logs and ensure all data requirements are met.

# Large-Scale Dataset Processing Guide

This guide explains how to use P-Companion with large-scale datasets.

## System Requirements

### Memory Requirements
For a dataset with:
- 1M products
- 10M co-view pairs
- 5M co-purchase pairs
- 3M purchase-after-view pairs

Estimated memory:
- Product embeddings: ~500MB (128-dim float32)
- Behavioral data: ~300MB
- Working memory: ~2-4GB
- Total with buffer: ~8GB recommended

### Storage Requirements
- Temporary files: ~1GB per 100K products
- Model checkpoints: ~100MB per save
- Feature cache: ~500MB per 100K products

## Data Format Requirements

Same as regular format, but with considerations for scale:

### Product Catalog
```csv
product_id,title,type,category
# Handles millions of rows efficiently through chunking
```

### Behavioral Data
```csv
product_id_1,product_id_2
# Split large files if needed:
# co_views_part1.csv, co_views_part2.csv, etc.
```

## Optimizations

1. Chunked Processing:
   ```python
   processor = LargeScaleDataProcessor(config)
   processor.CHUNK_SIZE = 10000  # Adjust based on memory
   ```

2. Memory Management:
   - Uses Dask for out-of-memory computations
   - HDF5 for feature storage
   - Gradient accumulation during training

3. Parallel Processing:
   - Multi-worker data loading
   - Distributed feature processing
   - Configurable number of workers

## Usage Example

```python
# Initialize with large-scale config
config = LargeScaleConfig()
config.CHUNK_SIZE = 10000
config.MAX_MEMORY = "32GB"

# Process data in chunks
processor = LargeScaleDataProcessor(config)
bpg = processor.process_data(
    product_data_path="products.csv",
    co_view_data_path="co_views.csv",
    temp_dir="temp"
)

# Train with gradient accumulation
train(
    config,
    train_loader,
    val_loader,
    accumulation_steps=4
)
```

## Best Practices

1. Memory Management:
   - Monitor memory usage with `estimate_memory_requirements()`
   - Use appropriate chunk sizes
   - Clean up temporary files

2. Data Preparation:
   - Split large files into manageable chunks
   - Pre-process features if possible
   - Use efficient file formats (parquet, HDF5)

3. Training Configuration:
   - Use gradient accumulation
   - Adjust batch sizes based on memory
   - Enable mixed precision training

## Performance Tuning

### Memory vs Speed Trade-offs
```python
# More memory, faster processing
config.CHUNK_SIZE = 20000
config.NUM_WORKERS = 8

# Less memory, slower processing
config.CHUNK_SIZE = 5000
config.NUM_WORKERS = 2
```

### Disk I/O Optimization
- Use SSD for temporary storage
- Enable memory mapping for large files
- Compress intermediate results

## Error Handling

Common issues and solutions:

1. Out of Memory:
   ```python
   # Reduce chunk size
   config.CHUNK_SIZE = config.CHUNK_SIZE // 2
   
   # Enable garbage collection
   import gc
   gc.collect()
   ```

2. Slow Processing:
   ```python
   # Increase parallelism
   config.NUM_WORKERS = os.cpu_count()
   
   # Use memory mapping
   config.USE_MMAP = True
   ```

3. Data Corruption:
   ```python
   # Enable checksums
   config.VERIFY_CHECKSUMS = True
   
   # Regular validation
   processor.validate_chunk(chunk_data)
   ```

## Monitoring and Logging

1. Memory Usage:
   ```python
   memory_stats = processor.estimate_memory_requirements(
       product_data_path=path
   )
   logger.info(f"Memory required: {memory_stats}")
   ```

2. Processing Progress:
   ```python
   # Progress bars for each phase
   with tqdm(total=total_chunks) as pbar:
       for chunk in process_chunks():
           pbar.update(1)
   ```

3. Performance Metrics:
   ```python
   # Time per chunk
   processing_time = time.time() - start_time
   logger.info(f"Chunk processed in {processing_time}s")
   ```

## Future Improvements

Planned optimizations:
1. Distributed processing support
2. Streaming data processing
3. Online feature updates
4. Automatic memory optimization

## Additional Notes

1. Use cases for large-scale processing:
   - E-commerce catalogs (millions of products)
   - User interaction data (billions of pairs)
   - Real-time recommendation systems

2. Alternative approaches:
   - Database-backed processing
   - Streaming processing
   - Distributed computing

3. Scaling considerations:
   - Linear memory scaling with products
   - Quadratic scaling with behavioral pairs
   - CPU vs GPU trade-offs