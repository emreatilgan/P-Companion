class BehaviorProductGraph:
    def __init__(self):
        self.nodes = {}  # Product ID to features mapping
        self.edges = {
            'co_purchase': set(),
            'co_view': set(),
            'purchase_after_view': set()
        }
        
    def add_node(self, product_id, features):
        self.nodes[product_id] = features
        
    def add_edge(self, source_id, target_id, edge_type):
        if edge_type in self.edges:
            self.edges[edge_type].add((source_id, target_id))
            
    def get_neighbors(self, product_id, edge_type=None):
        neighbors = set()
        if edge_type and edge_type in self.edges:
            neighbors.update(
                [target for source, target in self.edges[edge_type] if source == product_id]
            )
        else:
            for edge_set in self.edges.values():
                neighbors.update(
                    [target for source, target in edge_set if source == product_id]
                )
        return neighbors