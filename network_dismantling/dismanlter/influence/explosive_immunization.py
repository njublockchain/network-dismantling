from collections import defaultdict
from typing import List
import networkx as nx

from network_dismantling.dismanlter.dismantler import DismantlingStrategy


class ExplosiveImmunizationDismantling(DismantlingStrategy):
    def __init__(self, q=0.1, num_iterations=10):
        self.q = q  # Fraction of nodes to remove in each iteration
        self.num_iterations = num_iterations

    def dismantle(self, G: nx.Graph, num_nodes: int) -> List[int]:
        nodes_to_remove = []
        G_copy = G.copy()

        while len(nodes_to_remove) < num_nodes:
            scores = self.compute_ei_scores(G_copy)
            if not scores:
                break
            candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            num_to_remove = min(
                int(self.q * len(G_copy)), num_nodes - len(nodes_to_remove)
            )
            for node, _ in candidates[:num_to_remove]:
                nodes_to_remove.append(node)
                G_copy.remove_node(node)

        return nodes_to_remove

    def compute_ei_scores(self, G):
        scores = defaultdict(float)
        for _ in range(self.num_iterations):
            components = list(nx.connected_components(G))
            for component in components:
                if len(component) > 1:
                    subgraph = G.subgraph(component)
                    for node in component:
                        s = self.compute_s(subgraph, node)
                        scores[node] += s / self.num_iterations
        return scores

    def compute_s(self, G, node):
        G_copy = G.copy()
        G_copy.remove_node(node)
        new_components = list(nx.connected_components(G_copy))
        return sum(len(c) * (len(c) - 1) for c in new_components)
