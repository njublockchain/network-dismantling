import graph_tool.all as gt
import numpy as np
from typing import List, Dict

from network_dismantling.evaluators.evaluator import EvaluationMetric


class GlobalEfficiencyMetric(EvaluationMetric):
    """
    Evaluation metric that computes the global efficiency of a graph.

    The global efficiency is a measure of the efficiency of information transfer in a graph. The global efficiency is the
    average of the inverse shortest path lengths between all pairs of nodes in the graph.
    """

    def compute(self, G: gt.Graph) -> float:
        """
        Compute the global efficiency of the graph.

        :param G: The graph.
        :return: The global efficiency of the graph.
        """
        return gt.global_efficiency(G)

    @property
    def name(self) -> str:
        """
        Return the name of the evaluation metric.

        :return: The name of the evaluation metric.
        """
        return "Global Efficiency"
