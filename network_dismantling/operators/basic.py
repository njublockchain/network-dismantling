import networkx as nx

from typing import Any, List
from network_dismantling.operators.operator import DismantlingOperator


class NodeRemovalOperator(DismantlingOperator):
    """
    Dismantling operator that removes nodes from the graph.

    This dismantling operator removes nodes from the graph.
    """

    def operate(self, G: nx.Graph, elements: List[Any]) -> nx.Graph:
        """
        Remove nodes from the graph.

        :param G: The input graph
        :param elements: Nodes to remove
        :return: The modified graph
        """
        H = G.copy()
        H.remove_nodes_from(elements)
        return H


class EdgeRemovalOperator(DismantlingOperator):
    """
    Dismantling operator that removes edges from the graph.

    This dismantling operator removes edges from the graph.
    """

    def operate(self, G: nx.Graph, elements: List[Any]) -> nx.Graph:
        """
        Remove edges from the graph.

        :param G: The input graph
        :param elements: Edges to remove
        :return: The modified graph
        """
        H = G.copy()
        H.remove_edges_from(elements)
        return H
