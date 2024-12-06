import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import logging
from functools import lru_cache

class ProductFeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize text embedding model
        self.text_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # Initialize TF-IDF vectorizer for traditional features
        self.tfidf = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.category_embeddings = {}
        self.type_embeddings = {}
        
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove numbers but keep important ones (e.g., 2-in-1)
        text = re.sub(r'\b\d+\b', '', text)
        
        return text
    
    @lru_cache(maxsize=1000)
    def get_text_embeddings(self, text: str) -> torch.Tensor:
        """Get neural embeddings for text using SentenceTransformer"""
        return torch.tensor(self.text_model.encode(text))
    
    def get_category_embedding(self, category: str) -> torch.Tensor:
        """Get or create category embedding"""
        if category not in self.category_embeddings:
            self.category_embeddings[category] = self.get_text_embeddings(category)
        return self.category_embeddings[category]
    
    def get_type_embedding(self, product_type: str) -> torch.Tensor:
        """Get or create type embedding"""
        if product_type not in self.type_embeddings:
            # Split type (e.g., type_1_electronics -> electronics)
            type_category = product_type.split('_')[-1]
            self.type_embeddings[product_type] = self.get_text_embeddings(type_category)
        return self.type_embeddings[product_type]
    
    def extract_features(self, product_row: pd.Series) -> torch.Tensor:
        """Extract features from product data
        
        Combines multiple features:
        1. Title embeddings from neural model
        2. Category embeddings
        3. Type embeddings
        4. TF-IDF features (optional)
        """
        # Preprocess text
        clean_title = self.preprocess_text(product_row['title'])
        
        # Get neural embeddings for title
        title_embedding = self.get_text_embeddings(clean_title)
        
        # Get category and type embeddings
        category_embedding = self.get_category_embedding(product_row['category'])
        type_embedding = self.get_type_embedding(product_row['type'])
        
        # Combine embeddings
        combined_embedding = torch.cat([
            title_embedding,
            category_embedding,
            type_embedding
        ])
        
        # Project to desired dimension
        # TODO: get rid of this part
        if combined_embedding.shape[0] != self.config.PRODUCT_EMB_DIM:
            projection = torch.nn.Linear(
                combined_embedding.shape[0],
                self.config.PRODUCT_EMB_DIM
            )
            combined_embedding = projection(combined_embedding)
        
        # Normalize the final embedding
        combined_embedding = torch.nn.functional.normalize(
            combined_embedding,
            p=2,
            dim=0
        )
        
        return combined_embedding