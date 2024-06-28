import networkx as nx
import itertools
from typing import List

from network_dismantling.dismanlter.dismantler import DismantlingStrategy


class BruteForceDismantling(DismantlingStrategy):
    def __init__(self, target_size_ratio=0.1, max_depth=None):
        self.target_size_ratio = target_size_ratio
        self.max_depth = max_depth

    def dismantle(self, G: nx.Graph, num_nodes: int) -> List[int]:
        original_size = G.number_of_nodes()
        target_size = int(original_size * self.target_size_ratio)

        # If max_depth is not set, use num_nodes
        max_depth = self.max_depth if self.max_depth is not None else num_nodes

        for k in range(1, min(max_depth, num_nodes) + 1):
            for combination in itertools.combinations(G.nodes(), k):
                H = G.copy()
                H.remove_nodes_from(combination)
                if self.is_dismantled(H, target_size):
                    return list(combination)

        # If no solution found within max_depth, return the best found so far
        return self.greedy_fallback(G, num_nodes)

    def is_dismantled(self, G: nx.Graph, target_size: int) -> bool:
        largest_cc = max(nx.connected_components(G), key=len)
        return len(largest_cc) <= target_size

    def greedy_fallback(self, G: nx.Graph, num_nodes: int) -> List[int]:
        nodes_to_remove = []
        G_copy = G.copy()
        for _ in range(num_nodes):
            node_to_remove = max(G_copy.nodes(), key=G_copy.degree)
            nodes_to_remove.append(node_to_remove)
            G_copy.remove_node(node_to_remove)
        return nodes_to_remove
