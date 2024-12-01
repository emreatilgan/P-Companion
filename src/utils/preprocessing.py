import torch
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import Counter
from tqdm import tqdm
from constants import Constants

class TextPreprocessor:
    def __init__(self, config):
        self.config = config
        self.vocab = {}
        self.type_vocab = {}
        self.max_length = Constants.MAX_SEQUENCE_LENGTH
        
    def fit(self, texts: List[str], types: List[str]) -> None:
        """Build vocabulary from texts and types"""
        # Build text vocabulary
        word_counter = Counter()
        for text in tqdm(texts, desc="Processing texts"):
            words = text.lower().split()[:self.max_length]
            word_counter.update(words)
        
        # Filter by frequency and create vocab
        self.vocab = {
            Constants.PAD_TOKEN: 0,
            Constants.UNK_TOKEN: 1,
            **{
                word: idx + 2
                for idx, (word, count) in enumerate(
                    word_counter.most_common()
                )
                if count >= Constants.MIN_PRODUCT_FREQ
            }
        }
        
        # Build type vocabulary
        type_counter = Counter(types)
        self.type_vocab = {
            type_name: idx
            for idx, (type_name, count) in enumerate(
                type_counter.most_common()
            )
            if count >= Constants.MIN_TYPE_FREQ
        }
    
    def text_to_features(self, text: str) -> torch.Tensor:
        """Convert text to feature vector with padding"""
        words = text.lower().split()[:self.max_length]
        indices = [
            self.vocab.get(word, self.vocab[Constants.UNK_TOKEN])
            for word in words
        ]
        
        # Pad sequence
        if len(indices) < self.max_length:
            indices += [self.vocab[Constants.PAD_TOKEN]] * (self.max_length - len(indices))
        
        return torch.tensor(indices, dtype=torch.long)
    
    def type_to_index(self, type_name: str) -> int:
        """Convert type name to index"""
        return self.type_vocab.get(type_name, -1)
    
    def save(self, path: str) -> None:
        """Save preprocessor state"""
        torch.save({
            'vocab': self.vocab,
            'type_vocab': self.type_vocab,
            'max_length': self.max_length
        }, path)
    
    def load(self, path: str) -> None:
        """Load preprocessor state"""
        state = torch.load(path)
        self.vocab = state['vocab']
        self.type_vocab = state['type_vocab']
        self.max_length = state['max_length']