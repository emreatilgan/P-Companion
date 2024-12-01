from typing import Dict, List, Tuple
import torch
import numpy as np
from tqdm import tqdm
from constants import Constants

class Evaluator:
    def __init__(self, config):
        self.config = config
    
    def evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        Returns:
            Dictionary containing evaluation metrics
        """
        model.eval()
        metrics = {
            f'hit@{k}': 0.0 for k in Constants.TOP_K_VALUES
        }
        metrics.update({
            'type_diversity': 0.0,
            'mean_relevance': 0.0
        })
        
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                outputs = model(batch)
                
                # Calculate metrics
                for k in Constants.TOP_K_VALUES:
                    metrics[f'hit@{k}'] += self.calculate_hit_at_k(
                        outputs['projected_embeddings'],
                        batch['target_features'],
                        k
                    )
                
                metrics['type_diversity'] += self.calculate_type_diversity(
                    outputs['complementary_types']
                )
                
                metrics['mean_relevance'] += self.calculate_mean_relevance(
                    outputs['projected_embeddings'],
                    batch['target_features']
                )
                
                num_batches += 1
        
        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches
        
        return metrics
    
    @staticmethod
    def calculate_hit_at_k(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        k: int
    ) -> float:
        """Calculate Hit@K metric"""
        top_k = torch.topk(predictions, k, dim=1)[1]
        hits = torch.any(top_k == targets.unsqueeze(1), dim=1)
        return hits.float().mean().item()
    
    @staticmethod
    def calculate_type_diversity(predicted_types: torch.Tensor) -> float:
        """Calculate diversity of predicted types"""
        unique_types = torch.unique(predicted_types, dim=1)
        return unique_types.size(1) / predicted_types.size(1)
    
    @staticmethod
    def calculate_mean_relevance(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate mean relevance score"""
        similarities = torch.cosine_similarity(predictions, targets.unsqueeze(1), dim=-1)
        return similarities.mean().item()