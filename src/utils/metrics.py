import torch

class Metrics:
    @staticmethod
    def hit_at_k(predictions, ground_truth, k=10):
        """
        Calculate Hit@K metric
        Args:
            predictions: Tensor of shape [batch_size, num_items]
            ground_truth: Tensor of shape [batch_size]
            k: Number of top items to consider
        Returns:
            hit_rate: Float
        """
        # Get top k predictions
        top_k = torch.topk(predictions, k, dim=1)[1]
        
        # Check if ground truth is in top k
        hits = torch.any(top_k == ground_truth.unsqueeze(1), dim=1)
        
        return hits.float().mean().item()
    
    @staticmethod
    def type_diversity(predicted_types):
        """
        Calculate diversity of predicted complementary types
        Args:
            predicted_types: Tensor of shape [batch_size, num_types]
        Returns:
            diversity: Float
        """
        unique_types = torch.unique(predicted_types, dim=1)
        return unique_types.size(1) / predicted_types.size(1)
    
    @staticmethod
    def evaluate_model(model, data_loader, device):
        """
        Evaluate model performance
        """
        model.eval()
        metrics = {
            'hit@1': 0.0,
            'hit@3': 0.0,
            'hit@10': 0.0,
            'type_diversity': 0.0
        }
        
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = {k: v.to(device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Get model predictions
                outputs = model(batch)
                
                # Calculate metrics
                for k in [1, 3, 10]:
                    metrics[f'hit@{k}'] += Metrics.hit_at_k(
                        outputs['projected_embeddings'],
                        batch['target_features'],
                        k=k
                    )
                
                metrics['type_diversity'] += Metrics.type_diversity(
                    outputs['complementary_types']
                )
                
                num_batches += 1
        
        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches
        
        return metrics