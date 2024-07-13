import networkx as nx

from abc import abstractmethod
from typing import Any, List, Tuple
from network_dismantling.selector import ElementSelector


class EdgeSelector(ElementSelector):
    """
    Abstract class for edge selectors.

    An edge selector is a method to select edges from a graph.
    """

    @abstractmethod
    def select(self, G: nx.Graph, num_edges: int) -> List[Tuple[Any, Any]]:
        """
        Select edges from the graph.

        :param G: The input graph
        :param num_edges: Number of edges to select
        :return: List of selected edges
        """
        pass
