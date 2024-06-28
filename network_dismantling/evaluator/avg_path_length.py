import networkx as nx
import numpy as np
from typing import List, Dict, Callable
from abc import ABC, abstractmethod

from network_dismantling.evaluator.evaluator import EvaluationMetric


class AvgPathLengthMetric(EvaluationMetric):
    def compute(self, G: nx.Graph) -> float:
        lcc = max(nx.connected_components(G), key=len)
        return nx.average_shortest_path_length(G.subgraph(lcc))

    @property
    def name(self) -> str:
        return "Average Path Length"
