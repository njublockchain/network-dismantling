import graph_tool.all as gt

from abc import abstractmethod
from typing import Any, List
from network_dismantling.selector import ElementSelector


class NodeSelector(ElementSelector):
    """
    Abstract class for node selectors.

    A node selector is a method to select nodes from a graph.
    """

    @abstractmethod
    def select(self, G: gt.Graph, num_nodes: int) -> List[Any]:
        """
        Select nodes from the graph.

        :param G: The input graph
        :param num_nodes: Number of nodes to select
        :return: List of selected nodes
        """
        pass
