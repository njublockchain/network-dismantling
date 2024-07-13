import networkx as nx
from abc import ABC, abstractmethod
from typing import Any, List, Tuple
import torch.nn as nn

from .operators.operator import DismantlingOperator
from .selector import ElementSelector


class DismantlingStrategy(nn.Module):
    """
    Abstract class for dismantling strategies.

    A dismantling strategy is a method to remove nodes from a graph in order to dismantle it.
    """

    @abstractmethod
    def dismantle(self, G: nx.Graph, num_nodes: int) -> List[int]:
        """
        Dismantle the graph by removing nodes.

        :param G: The input graph
        :param num_nodes: Number of nodes to remove
        :return: List of node IDs to remove
        """
        pass


class NetworkDismantler:
    """
    Class to dismantle a network using a given selector and operator.

    The network dismantler dismantles a network by selecting elements (nodes or edges) using a selector and then removing
    them using an operator.
    """

    def __init__(self, selector: ElementSelector, operator: DismantlingOperator):
        """
        Initialize the network dismantler.

        :param selector: Element selector
        :param operator: Dismantling operator
        """
        self.selector = selector
        self.operator = operator

    def dismantle(self, G: nx.Graph, num_elements: int) -> Tuple[nx.Graph, List[Any]]:
        """
        Dismantle the graph by removing elements.

        :param G: The input graph
        :param num_elements: Number of elements to remove
        :return: The dismantled graph and the selected elements
        """
        selected_elements = self.selector.select(G, num_elements)
        dismantled_G = self.operator.operate(G, selected_elements)
        return dismantled_G, selected_elements
