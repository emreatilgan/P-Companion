# P-Companion: Diversified Complementary Product Recommendation

This is an unofficial implementation of the paper "P-Companion: A Principled Framework for Diversified Complementary Product Recommendation".
https://assets.amazon.science/d5/16/3f7809974a899a11bacdadefdf24/p-companion-a-principled-framework-for-diversified-complementary-product-recommendation.pdf

## Features

- Product2Vec embeddings using Graph Attention Networks
- Complementary type transition modeling
- Type-guided complementary item prediction
- Support for cold-start products
- Comprehensive evaluation metrics

## Project Structure

```
.
├── src/
│   ├── models/
│   │   ├── product2vec.py
│   │   ├── type_transition.py
│   │   ├── item_prediction.py
│   │   └── p_companion.py
│   ├── data/
│   │   ├── data_loader.py
│   │   └── bpg.py
│   │   └── synthetic_data.py
│   └── utils/
│       ├── metrics.py
│       ├── constants.py
│       ├── preprocessing.py
│       ├── logger.py
│       └── evaluation.py
├── train.py
├── inference.py
├── config.py
└── requirements.txt
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model:

```bash
python train.py
```

## Requirements

See `requirements.txt` for detailed dependencies:

## Evaluation Metrics

- Hit@K (K=1,3,10,60)
- Type diversity
- Mean relevance score

## Configuration

Key configuration parameters in `config.py`:

- PRODUCT_EMB_DIM: Product embedding dimension (128)
- TYPE_EMB_DIM: Type embedding dimension (64)
- NUM_COMP_TYPES: Number of complementary types to predict (3)
- ALPHA: Trade-off parameter between type and item prediction (0.8)

## References

[1] Junheng Hao, Tong Zhao, Jin Li, Xin Luna Dong, Christos Faloutsos, Yizhou Sun, and Wei Wang. 2020. P-Companion: A Principled Framework for Diversified Complementary Product Recommendation. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management (CIKM ‘20). Association for Computing Machinery, New York, NY, USA, 2517–2524. https://doi.org/10.1145/3340531.3412732

[2] Claude AI (Version 3.5 Sonnet, October 2024). Anthropic. https://www.anthropic.com/claude

[3] Anthropic. (2024). Claude Documentation. https://docs.anthropic.com
