import torch
from typing import Dict, List
import numpy as np

class Metrics:
    @staticmethod
    def hit_at_k(predictions: torch.Tensor, ground_truth: torch.Tensor, k: int) -> float:
        """
        Calculate Hit@K metric
        Args:
            predictions: Tensor of shape [batch_size, num_predictions]
            ground_truth: Tensor of shape [batch_size]
            k: Number of top items to consider
        Returns:
            hit_rate: Float
        """
        # Ensure k is not larger than the number of predictions
        k = min(k, predictions.size(1))
        
        # Get top k predictions
        top_k = torch.topk(predictions, k, dim=1)[1]
        
        # Check if ground truth is in top k
        hits = torch.any(top_k == ground_truth.unsqueeze(1), dim=1)
        
        return hits.float().mean().item()
    
    @staticmethod
    def type_diversity(predicted_types: torch.Tensor) -> float:
        """
        Calculate diversity of predicted complementary types
        Args:
            predicted_types: Tensor of shape [batch_size, num_types]
        Returns:
            diversity: Float between 0 and 1
        """
        if predicted_types.numel() == 0:
            return 0.0
            
        unique_types = torch.unique(predicted_types, dim=1)
        return unique_types.size(1) / predicted_types.size(1)
    
    @staticmethod
    def mean_relevance(predictions: torch.Tensor, ground_truth: torch.Tensor) -> float:
        """
        Calculate mean relevance score using cosine similarity
        Args:
            predictions: Tensor of shape [batch_size, num_predictions, embedding_dim]
            ground_truth: Tensor of shape [batch_size, embedding_dim]
        Returns:
            relevance: Float between -1 and 1
        """
        # Calculate cosine similarity
        similarities = torch.cosine_similarity(
            predictions, 
            ground_truth.unsqueeze(1), 
            dim=-1
        )
        return similarities.mean().item()
    
    @staticmethod
    def evaluate_model(model: torch.nn.Module, 
                      data_loader: torch.utils.data.DataLoader,
                      device: torch.device) -> Dict[str, float]:
        """
        Evaluate model performance
        """
        model.eval()
        metrics = {
            'hit@1': 0.0,
            'hit@3': 0.0,
            'hit@10': 0.0,
            'type_diversity': 0.0,
            'mean_relevance': 0.0
        }
        
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = {k: v.to(device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Get model predictions
                outputs = model(batch)
                
                # Get similarity scores between projected embeddings and all possible items
                similarities = torch.matmul(
                    outputs['projected_embeddings'].view(-1, outputs['projected_embeddings'].size(-1)),
                    batch['target_features'].T
                )
                
                # Calculate metrics
                for k in [1, 3, min(10, similarities.size(1))]:
                    metrics[f'hit@{k}'] += Metrics.hit_at_k(
                        similarities,
                        torch.arange(similarities.size(0), device=device),
                        k=k
                    )
                
                metrics['type_diversity'] += Metrics.type_diversity(
                    outputs['complementary_types']
                )
                
                metrics['mean_relevance'] += Metrics.mean_relevance(
                    outputs['projected_embeddings'],
                    batch['positive_items']
                )
                
                num_batches += 1
        
        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches
        
        return metrics