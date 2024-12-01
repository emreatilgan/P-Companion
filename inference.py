import torch
import logging
import os
from typing import List, Dict, Any

from config import Config
from models.p_companion import PCompanion
from data.bpg import BehaviorProductGraph

class PCompanionInference:
    def __init__(self, model_path: str, config: Config):
        self.config = config
        self.device = config.DEVICE
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = PCompanion(config).to(self.device)
        
        # Load model weights
        self._load_model(model_path)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize BPG
        self.bpg = BehaviorProductGraph()
        # Load your data into BPG here
        
    def _load_model(self, model_path: str):
        """Load trained model weights"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Loaded model from {model_path}")
    
    def _prepare_input(self, query_id: str) -> Dict[str, torch.Tensor]:
        """Prepare model input from query product ID"""
        if query_id not in self.bpg.nodes:
            raise ValueError(f"Product ID {query_id} not found in BPG")
            
        # Get product features and type
        query_features = self.bpg.nodes[query_id]['features']
        query_type = self.bpg.nodes[query_id]['type']
        
        # Get neighbor features
        neighbor_features = []
        for neighbor_id in self.bpg.get_neighbors(query_id):
            if neighbor_id in self.bpg.nodes:
                neighbor_features.append(self.bpg.nodes[neighbor_id]['features'])
        
        # Prepare input batch
        batch = {
            'query_features': query_features.unsqueeze(0).to(self.device),
            'query_type': torch.tensor([query_type]).to(self.device)
        }
        
        if neighbor_features:
            batch['query_neighbor_features'] = torch.stack(neighbor_features).to(self.device)
            
        return batch
    
    def recommend(self, query_id: str, num_recommendations: int = 10) -> Dict[str, Any]:
        """
        Generate complementary product recommendations
        
        Args:
            query_id: ID of the query product
            num_recommendations: Number of recommendations to generate
            
        Returns:
            Dictionary containing:
                - complementary_types: List of predicted complementary product types
                - recommendations: List of recommended product IDs for each type
                - scores: Relevance scores for each recommendation
        """
        with torch.no_grad():
            # Prepare input
            batch = self._prepare_input(query_id)
            
            # Get model predictions
            outputs = self.model(batch)
            
            # Process predictions
            complementary_types = outputs['complementary_types'][0].cpu().numpy()
            projected_embeddings = outputs['projected_embeddings'][0]
            
            # Find nearest neighbors for each complementary type
            recommendations = []
            scores = []
            
            for type_idx, proj_emb in zip(complementary_types, projected_embeddings):
                # Get all products of this type
                type_products = self.bpg.get_products_by_type(type_idx)
                
                if not type_products:
                    continue
                    
                # Calculate similarities
                type_embeddings = torch.stack([
                    self.bpg.nodes[p_id]['features'] 
                    for p_id in type_products
                ]).to(self.device)
                
                similarities = torch.matmul(
                    proj_emb.unsqueeze(0),
                    type_embeddings.T
                )[0]
                
                # Get top recommendations
                top_k = min(num_recommendations, len(type_products))
                top_scores, top_indices = torch.topk(similarities, k=top_k)
                
                recommendations.append([
                    type_products[idx] for idx in top_indices.cpu().numpy()
                ])
                scores.append(top_scores.cpu().numpy())
            
            return {
                'complementary_types': complementary_types.tolist(),
                'recommendations': recommendations,
                'scores': scores
            }

def main():
    # Initialize config
    config = Config()
    
    # Initialize inference
    inferencer = PCompanionInference(
        model_path=os.path.join(config.MODEL_DIR, 'best_model.pth'),
        config=config
    )
    
    # Example usage
    query_id = "example_product_id"
    recommendations = inferencer.recommend(query_id)
    
    print(f"Recommendations for product {query_id}:")
    for type_idx, (type_recs, type_scores) in enumerate(zip(
        recommendations['recommendations'],
        recommendations['scores']
    )):
        print(f"\nComplementary type {recommendations['complementary_types'][type_idx]}:")
        for rec_id, score in zip(type_recs, type_scores):
            print(f"  Product {rec_id}: {score:.4f}")

if __name__ == "__main__":
    main()