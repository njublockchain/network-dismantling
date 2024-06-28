import networkx as nx
import numpy as np
from typing import List, Dict, Callable
from abc import ABC, abstractmethod

from network_dismantling.evaluator.evaluator import EvaluationMetric


class GlobalEfficiencyMetric(EvaluationMetric):
    def compute(self, G: nx.Graph) -> float:
        return nx.global_efficiency(G)

    @property
    def name(self) -> str:
        return "Global Efficiency"
