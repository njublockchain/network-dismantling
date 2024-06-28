import networkx as nx
import numpy as np
from typing import List, Dict, Callable
from abc import ABC, abstractmethod

from network_dismantling.evaluator.evaluator import EvaluationMetric


class LCCSizeMetric(EvaluationMetric):
    def compute(self, G: nx.Graph) -> float:
        return len(max(nx.connected_components(G), key=len))

    @property
    def name(self) -> str:
        return "LCC Size"
