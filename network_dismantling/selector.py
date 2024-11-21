import graph_tool.all as gt

from abc import ABC, abstractmethod
from typing import Any, List


class ElementSelector(ABC):
    @abstractmethod
    def select(self, G: gt.Graph, num_elements: int) -> List[Any]:
        """
        Select elements (nodes or edges) from the graph.

        :param G: The input graph
        :param num_elements: Number of elements to select
        :return: List of selected elements
        """
        pass
