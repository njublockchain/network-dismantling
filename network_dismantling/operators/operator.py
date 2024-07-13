from abc import ABC, abstractmethod
from typing import Any, List
import networkx as nx


class DismantlingOperator(ABC):
    @abstractmethod
    def operate(self, G: nx.Graph, elements: List[Any]) -> nx.Graph:
        """
        Perform the dismantling operation on the graph.

        :param G: The input graph
        :param elements: Elements to operate on
        :return: The modified graph
        """
        pass
