import networkx as nx
import numpy as np
from typing import List, Dict, Callable
from abc import ABC, abstractmethod

from network_dismantling.dismantler.dismantler import (
    DismantlingStrategy,
    NetworkDismantler,
)


class EvaluationMetric(ABC):
    @abstractmethod
    def compute(self, G: nx.Graph) -> float:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class EvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(
        self,
        original_graph: nx.Graph,
        dismantled_graph: nx.Graph,
        metrics: List[EvaluationMetric],
    ) -> Dict[str, float]:
        pass


class RelativeChangeStrategy(EvaluationStrategy):
    def evaluate(
        self,
        original_graph: nx.Graph,
        dismantled_graph: nx.Graph,
        metrics: List[EvaluationMetric],
    ) -> Dict[str, float]:
        results = {}
        for metric in metrics:
            original_value = metric.compute(original_graph)
            dismantled_value = metric.compute(dismantled_graph)
            relative_change = (dismantled_value - original_value) / original_value
            results[metric.name] = relative_change
        results["Removed Nodes"] = (
            1 - dismantled_graph.number_of_nodes() / original_graph.number_of_nodes()
        )
        return results


class AbsoluteValueStrategy(EvaluationStrategy):
    def evaluate(
        self,
        original_graph: nx.Graph,
        dismantled_graph: nx.Graph,
        metrics: List[EvaluationMetric],
    ) -> Dict[str, float]:
        results = {}
        for metric in metrics:
            results[metric.name] = metric.compute(dismantled_graph)
        results["Removed Nodes"] = (
            original_graph.number_of_nodes() - dismantled_graph.number_of_nodes()
        )
        return results


class DismantlingEvaluator:
    def __init__(self, metrics: List[EvaluationMetric], strategy: EvaluationStrategy):
        self.metrics = metrics
        self.strategy = strategy

    def evaluate(
        self, original_graph: nx.Graph, dismantled_graph: nx.Graph
    ) -> Dict[str, float]:
        return self.strategy.evaluate(original_graph, dismantled_graph, self.metrics)

    @staticmethod
    def compare_strategies(
        original_graph: nx.Graph,
        dismantling_strategies: List[DismantlingStrategy],
        num_nodes_to_remove: int,
        evaluator: "DismantlingEvaluator",
    ) -> Dict[str, Dict[str, float]]:
        results = {}
        for strategy in dismantling_strategies:
            dismantler = NetworkDismantler(strategy)
            dismantled_graph, _ = dismantler.dismantle(
                original_graph, num_nodes_to_remove
            )
            evaluation = evaluator.evaluate(original_graph, dismantled_graph)
            results[strategy.__class__.__name__] = evaluation

        return results
