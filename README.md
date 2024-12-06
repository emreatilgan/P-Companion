# P-Companion Implementation

This repository contains an unofficial implementation of the paper ["P-Companion: A Principled Framework for Diversified Complementary Product Recommendation"](https://assets.amazon.science/d5/16/3f7809974a899a11bacdadefdf24/p-companion-a-principled-framework-for-diversified-complementary-product-recommendation.pdf).

## Overview

P-Companion is a framework for product recommendations that considers both relevance and diversity. The key innovation is its two-phase approach:

1. Product2Vec Phase:
   - Pre-trains product embeddings using a graph attention network
   - Uses co-view and purchase-after-view relationships
   - Creates foundation embeddings for all products

2. Complementary Learning Phase:
   - Type transition learning for diversity
   - Item-level complementary prediction
   - Multi-task learning approach

## Project Structure

```
src/
├── models/
│   ├── product2vec.py      # Product2Vec embeddings with GAT
│   ├── type_transition.py  # Type prediction module
│   ├── item_prediction.py  # Item prediction module
│   └── p_companion.py      # Main model integration
├── data/
│   ├── data_loader.py      # Data loading utilities
│   ├── bpg.py              # Behavior Product Graph
│   └── synthetic_data.py   # Synthetic data generation
└── utils/
    ├── metrics.py          # Evaluation metrics
    └── constants.py        # Model constants
├── train.py
├── config.py
└── requirements.txt
```

## Key Components

### Behavior Product Graph (BPG)
- Integrates product features and behavioral data
- Handles three types of edges:
  * Co-view relationships
  * Purchase-after-view relationships
  * Co-purchase relationships

### Product2Vec
- Graph attention-based embedding learning
- Uses (Bcv ∩ Bpv) - Bcp for similarity learning
- Generates pretrained embeddings for all products

### P-Companion
- Type Transition Module for diversity
- Item Prediction Module for relevance
- Joint learning framework

## Current Status

The implementation includes:

1. Complete end-to-end training pipeline
2. Synthetic data generation for testing
3. Evaluation metrics:
   - Hit@K (K=1,3,10)
   - Type diversity
   - Mean relevance

Current performance on synthetic data:
- Product2Vec Loss: 0.23 (after 10 epochs)
- P-Companion Best Hit@10: 1.68%
- Perfect type diversity (1.0)
- Mean relevance needs improvement

## TODOs and Future Work

1. Model Improvements:
   - Fine-tune model parameters
   - Investigate negative mean relevance
   - Improve embedding quality

2. Data Generation:
   - Adjust synthetic data distributions
   - Add more realistic patterns
   - Better complementary relationship modeling

3. Analysis:
   - Add embedding visualization
   - Analyze type transition patterns
   - Study failure cases

## Usage

1. Training the model:
```bash
python train.py
```

## Dependencies

See `requirements.txt` for detailed dependencies

## References

[1] Junheng Hao, Tong Zhao, Jin Li, Xin Luna Dong, Christos Faloutsos, Yizhou Sun, and Wei Wang. 2020. P-Companion: A Principled Framework for Diversified Complementary Product Recommendation. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management (CIKM ‘20). Association for Computing Machinery, New York, NY, USA, 2517–2524. https://doi.org/10.1145/3340531.3412732

[2] Claude AI (Version 3.5 Sonnet, October 2024). Anthropic. https://www.anthropic.com/claude

[3] Anthropic. (2024). Claude Documentation. https://docs.anthropic.com