import graph_tool.all as gt
from typing import Any, List


from .node_selector import NodeSelector


class Betweenness(NodeSelector):
    def select(self, G: gt.Graph, num_nodes: int) -> List[Any]:
        betweenness = gt.betweenness(G)[0]
        return sorted(enumerate(betweenness), key=lambda x: x[1], reverse=True)[:num_nodes]


class Degree(NodeSelector):
    def select(self, G: gt.Graph, num_nodes: int) -> List[Any]:
        return sorted(G.degree, key=lambda x: x[1], reverse=True)[:num_nodes]
