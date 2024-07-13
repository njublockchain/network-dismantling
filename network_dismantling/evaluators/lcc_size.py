import networkx as nx
import numpy as np
from typing import List, Dict, Callable
from abc import ABC, abstractmethod

from network_dismantling.evaluators.evaluator import EvaluationMetric


class LCCSizeMetric(EvaluationMetric):
    """
    Evaluation metric that computes the size of the largest connected component of a graph.

    The size of the largest connected component is a measure of the size of the largest connected subgraph of the graph.
    """

    def compute(self, G: nx.Graph) -> float:
        """
        Compute the size of the largest connected component of the graph.

        :param G: The graph.
        :return: The size of the largest connected component of the graph.
        """
        return len(max(nx.connected_components(G), key=len))

    @property
    def name(self) -> str:
        """
        Return the name of the evaluation metric.

        :return: The name of the evaluation metric.
        """
        return "LCC Size"
