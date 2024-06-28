import networkx as nx
from typing import List
from network_dismantling.dismanlter.dismantler import DismantlingStrategy


class BetweennessDismantling(DismantlingStrategy):
    def dismantle(self, G: nx.Graph, num_nodes: int) -> List[int]:
        betweenness = nx.betweenness_centrality(G)
        nodes_to_remove = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[
            :num_nodes
        ]
        return [node for node, _ in nodes_to_remove]
