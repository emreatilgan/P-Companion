from typing import Dict, Set, List, Any, Optional
import torch

class BehaviorProductGraph:
    """Behavior Product Graph Implementation"""
    
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}  # Product ID to features mapping
        self.edges: Dict[str, Set[tuple]] = {
            'co_purchase': set(),
            'co_view': set(),
            'purchase_after_view': set()
        }
        
    def add_node(self, product_id: str, features: Dict[str, Any]) -> None:
        """Add a product node to the graph"""
        self.nodes[product_id] = features
        
    def add_edge(self, source_id: str, target_id: str, edge_type: str) -> None:
        """Add an edge between two products"""
        if edge_type in self.edges:
            self.edges[edge_type].add((source_id, target_id))
            
    def get_neighbors(self, product_id: str, edge_type: Optional[str] = None) -> Set[str]:
        """Get neighbor products for a given product"""
        neighbors = set()
        if edge_type and edge_type in self.edges:
            neighbors.update(
                [target for source, target in self.edges[edge_type] 
                 if source == product_id]
            )
        else:
            for edge_set in self.edges.values():
                neighbors.update(
                    [target for source, target in edge_set 
                     if source == product_id]
                )
        return neighbors
    
    def get_all_types(self) -> Set[str]:
        """Get all unique product types in the graph"""
        return {node['type'] for node in self.nodes.values()}
    
    def get_products_by_type(self, product_type: str) -> List[str]:
        """Get all products of a given type"""
        return [
            pid for pid, data in self.nodes.items()
            if data['type'] == product_type
        ]
    
    def get_exclusive_co_purchase_pairs(self) -> List[tuple]:
        """Get co-purchase pairs that are not in co-view"""
        co_view_pairs = set(self.edges['co_view'])
        exclusive_pairs = [(src, tgt, 1) for src, tgt in self.edges['co_purchase']
                         if (src, tgt) not in co_view_pairs]
        return exclusive_pairs
    
    def get_co_view_intersection_pairs(self) -> List[tuple]:
        """Get co-view pairs that intersect with purchase-after-view"""
        pav_pairs = set(self.edges['purchase_after_view'])
        intersection = [(src, tgt, -1) for src, tgt in self.edges['co_view']
                      if (src, tgt) in pav_pairs]
        return intersection