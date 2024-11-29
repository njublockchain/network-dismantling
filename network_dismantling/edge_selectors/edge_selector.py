import graph_tool.all as gt
from abc import abstractmethod
from typing import Any, List, Tuple
from network_dismantling.selector import ElementSelector


class EdgeSelector(ElementSelector):
    """
    Abstract class for edge selectors.

    An edge selector is a method to select edges from a graph.
    """

    @abstractmethod
    def select(self, G: gt.Graph, num_elements: int) -> List[Tuple[Any, Any]]:
        """
        Select edges from the graph.

        :param G: The input graph
        :param num_elements: Number of edges to select
        :return: List of selected edges
        """
