import networkx as nx
import numpy as np
from typing import List, Dict, Callable
from abc import ABC, abstractmethod

from network_dismantling.evaluator.evaluator import EvaluationMetric


class GlobalEfficiencyMetric(EvaluationMetric):
    """
    Evaluation metric that computes the global efficiency of a graph.

    The global efficiency is a measure of the efficiency of information transfer in a graph. The global efficiency is the
    average of the inverse shortest path lengths between all pairs of nodes in the graph.
    """

    def compute(self, G: nx.Graph) -> float:
        """
        Compute the global efficiency of the graph.

        :param G: The graph.
        :return: The global efficiency of the graph.
        """
        return nx.global_efficiency(G)

    @property
    def name(self) -> str:
        """
        Return the name of the evaluation metric.

        :return: The name of the evaluation metric.
        """
        return "Global Efficiency"
