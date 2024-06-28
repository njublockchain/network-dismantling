import networkx as nx
from typing import List
from network_dismantling.dismanlter.dismantler import DismantlingStrategy


class DegreeDismantling(DismantlingStrategy):
    def dismantle(self, G: nx.Graph, num_nodes: int) -> List[int]:
        nodes_to_remove = sorted(G.degree, key=lambda x: x[1], reverse=True)[:num_nodes]
        return [node for node, _ in nodes_to_remove]
