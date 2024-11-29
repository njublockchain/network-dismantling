import graph_tool.all as gt

from abc import ABC, abstractmethod
from typing import Any, List


class DismantlingOperator(ABC):
    @abstractmethod
    def operate(self, G: gt.Graph, elements: List[Any]) -> gt.Graph:
        """
        Perform the dismantling operation on the graph.

        :param G: The input graph
        :param elements: Elements to operate on
        :return: The modified graph
        """
