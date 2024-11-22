import graph_tool.all as gt

from typing import List, Dict
from abc import ABC, abstractmethod

from network_dismantling.dismantler import NetworkDismantler


class EvaluationMetric(ABC):
    """
    Abstract base class for evaluation metrics.

    Evaluation metrics are used to evaluate the performance of a dismantling strategy by computing a metric on the
    dismantled graph.
    """

    @abstractmethod
    def compute(self, G: gt.Graph) -> float:
        """
        Compute the evaluation metric on the graph.

        :param G: The graph.
        :return: The value of the evaluation metric.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of the evaluation metric.

        :return: The name of the evaluation metric.
        """


class EvaluationStrategy(ABC):
    """
    Abstract base class for evaluation strategies.

    Evaluation strategies are used to evaluate the performance of a dismantling strategy by comparing the values of
    evaluation metrics on the original and dismantled graphs.
    """

    @abstractmethod
    def evaluate(
        self,
        original_graph: gt.Graph,
        dismantled_graph: gt.Graph,
        metrics: List[EvaluationMetric],
    ) -> Dict[str, float]:
        """
        Evaluate the performance of a dismantling strategy.

        :param original_graph: The original graph.
        :param dismantled_graph: The dismantled graph.
        :param metrics: The evaluation metrics to compute.
        :return: A dictionary mapping evaluation metric names to their values.
        """


class RelativeChangeStrategy(EvaluationStrategy):
    """
    Evaluation strategy that computes the relative change in evaluation metrics between the original and dismantled graphs.
    """

    def evaluate(
        self,
        original_graph: gt.Graph,
        dismantled_graph: gt.Graph,
        metrics: List[EvaluationMetric],
    ) -> Dict[str, float]:
        """
        Evaluate the performance of a dismantling strategy by computing the relative change in evaluation metrics between
        the original and dismantled graphs.

        :param original_graph: The original graph.
        :param dismantled_graph: The dismantled graph.
        :param metrics: The evaluation metrics to compute.
        :return: A dictionary mapping evaluation metric names to their relative changes.
        """
        results = {}
        for metric in metrics:
            original_value = metric.compute(original_graph)
            dismantled_value = metric.compute(dismantled_graph)
            relative_change = (dismantled_value - original_value) / original_value
            results[metric.name] = relative_change

        results["Removed Nodes"] = (
            original_graph.num_vertices() - dismantled_graph.num_vertices()
        ) / original_graph.num_vertices()

        return results


class AbsoluteValueStrategy(EvaluationStrategy):
    """
    Evaluation strategy that computes the absolute values of evaluation metrics on the dismantled graph.
    """

    def evaluate(
        self,
        original_graph: gt.Graph,
        dismantled_graph: gt.Graph,
        metrics: List[EvaluationMetric],
    ) -> Dict[str, float]:
        """
        Evaluate the performance of a dismantling strategy by computing the absolute values of evaluation metrics on the
        dismantled graph.

        :param original_graph: The original graph.
        :param dismantled_graph: The dismantled graph.
        :param metrics: The evaluation metrics to compute.
        :return: A dictionary mapping evaluation metric names to their values.
        """
        results = {}
        for metric in metrics:
            results[metric.name] = metric.compute(dismantled_graph)
        results["Removed Nodes"] = (
            original_graph.number_of_nodes() - dismantled_graph.number_of_nodes()
        )
        return results


class DismantlingEvaluator:
    """
    Class for evaluating dismantling strategies.

    The DismantlingEvaluator class is used to evaluate the performance of dismantling strategies by comparing the values
    of evaluation metrics on the original and dismantled graphs.
    """

    def __init__(self, metrics: List[EvaluationMetric], strategy: EvaluationStrategy):
        """
        Initialize the dismantling evaluator.

        :param metrics: The evaluation metrics to compute.
        :param strategy: The evaluation strategy to use.
        """
        self.metrics = metrics
        self.strategy = strategy

    def evaluate(
        self, original_graph: gt.Graph, dismantled_graph: gt.Graph
    ) -> Dict[str, float]:
        """
        Evaluate the performance of a dismantling strategy.

        :param original_graph: The original graph.
        :param dismantled_graph: The dismantled graph.
        :return: A dictionary mapping evaluation metric names to their values.
        """
        return self.strategy.evaluate(original_graph, dismantled_graph, self.metrics)

    @staticmethod
    def compare_strategies(
        original_graph: gt.Graph,
        dismantlers: List[NetworkDismantler],
        num_nodes_to_remove: int,
        evaluator: "DismantlingEvaluator",
    ) -> Dict[str, Dict[str, float]]:
        results = {}
        for dismantler in dismantlers:
            dismantled_graph, _ = dismantler.dismantle(
                original_graph, num_nodes_to_remove
            )
            evaluation = evaluator.evaluate(original_graph, dismantled_graph)
            strategy_name = dismantler.selector.__class__.__name__ + " with " + dismantler.operator.__class__.__name__
            results[strategy_name] = evaluation

        return results
