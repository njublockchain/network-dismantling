import graph_tool.all as gt

from typing import List, Dict
from network_dismantling.evaluators.evaluator import EvaluationMetric


class NumComponentsMetric(EvaluationMetric):
    """
    Evaluation metric that computes the number of connected components of a graph.

    The number of connected components is a measure of the number of connected subgraphs in the graph.
    """

    def compute(self, G: gt.Graph) -> float:
        """
        Compute the number of connected components of the graph.

        :param G: The graph.
        :return: The number of connected components of the graph.
        """
        return gt.number_connected_components(G)

    @property
    def name(self) -> str:
        """
        Return the name of the evaluation metric.

        :return: The name of the evaluation metric.
        """
        return "Number of Components"
