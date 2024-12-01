# P-Companion: Diversified Complementary Product Recommendation

This is an implementation of the paper "P-Companion: A Principled Framework for Diversified Complementary Product Recommendation".

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

### Inference

To run inference:

```bash
python inference.py --query_id <product_id>
```

## Requirements

See `requirements.txt` for detailed dependencies:

```
torch>=1.9.0
numpy>=1.19.2
tqdm>=4.50.0
tensorboard>=2.4.0
```

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