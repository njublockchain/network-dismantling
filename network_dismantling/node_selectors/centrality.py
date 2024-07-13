from typing import Any, List
import networkx as nx

from .node_selector import NodeSelector


class Betweenness(NodeSelector):
    def select(self, G: nx.Graph, num_nodes: int) -> List[Any]:
        betweenness = nx.betweenness_centrality(G)
        return sorted(betweenness, key=betweenness.get, reverse=True)[:num_nodes]


class Degree(NodeSelector):
    def select(self, G: nx.Graph, num_nodes: int) -> List[Any]:
        return sorted(G.degree, key=lambda x: x[1], reverse=True)[:num_nodes]
