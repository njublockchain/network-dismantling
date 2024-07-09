import networkx as nx
import numpy as np
from typing import List, Dict, Callable
from abc import ABC, abstractmethod

from network_dismantling.evaluator.evaluator import EvaluationMetric


class NumComponentsMetric(EvaluationMetric):
    """
    Evaluation metric that computes the number of connected components of a graph.

    The number of connected components is a measure of the number of connected subgraphs in the graph.
    """

    def compute(self, G: nx.Graph) -> float:
        """
        Compute the number of connected components of the graph.

        :param G: The graph.
        :return: The number of connected components of the graph.
        """
        return nx.number_connected_components(G)

    @property
    def name(self) -> str:
        """
        Return the name of the evaluation metric.

        :return: The name of the evaluation metric.
        """
        return "Number of Components"
