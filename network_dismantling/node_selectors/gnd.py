import networkx as nx
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
from scipy.sparse import csr_matrix  # type: ignore
from scipy.sparse.csgraph import connected_components  # type: ignore
from tqdm import tqdm  # type: ignore

from .node_selector import NodeSelector


class GND(NodeSelector):
    """
    Dismantling strategy that removes nodes based on their GND score.

    This dismantling strategy removes nodes based on their GND score, which is a measure of the importance of a node in
    the graph. The nodes with the highest GND score are removed first.
    """

    def __init__(self, c=1, epsilon=0.01, max_iterations=100):
        """
        Initialize the dismantling strategy.

        :param c: The damping factor.
        :param epsilon: The convergence threshold.
        :param max_iterations: The maximum number of iterations.
        """
        self.c = c
        self.epsilon = epsilon
        self.max_iterations = max_iterations

    def dismantle(self, G: nx.Graph, num_nodes: int) -> List[int]:
        """
        Dismantle the graph by removing nodes based on their GND score.

        :param G: The graph to dismantle.
        :param num_nodes: The number of nodes to remove.
        :return: A list of node indices to remove.
        """
        A = nx.adjacency_matrix(G)
        N = A.shape[0]
        D = np.array(A.sum(axis=1)).flatten()
        L = csr_matrix(np.diag(D) - A)

        x = np.random.rand(N)
        x = x / np.linalg.norm(x)

        for _ in tqdm(range(self.max_iterations), desc="GND Dismantling"):
            x_old = x
            x = L.dot(x) + self.c * x
            x = x / np.linalg.norm(x)
            if np.linalg.norm(x - x_old) < self.epsilon:
                break

        scores = np.abs(x)
        nodes_to_remove = np.argsort(scores)[::-1][:num_nodes]
        return nodes_to_remove.tolist()


class CoreHD(NodeSelector):
    """
    Dismantling strategy that removes nodes based on their core number.

    This dismantling strategy removes nodes based on their core number, which is the largest k-core that a node belongs
    to. The nodes with the highest core number are removed first.
    """

    def dismantle(self, G: nx.Graph, num_nodes: int) -> List[int]:
        """
        Dismantle the graph by removing nodes based on their core number.

        :param G: The graph to dismantle.
        :param num_nodes: The number of nodes to remove.
        :return: A list of node indices to remove.
        """
        core_numbers = nx.core_number(G)
        degrees = dict(G.degree())

        nodes_to_remove = []
        for _ in tqdm(range(num_nodes), desc="CoreHD Dismantling"):
            max_core = max(core_numbers.values())
            candidates = [
                node for node, core in core_numbers.items() if core == max_core
            ]
            node_to_remove = max(candidates, key=lambda n: degrees[n])

            nodes_to_remove.append(node_to_remove)
            G.remove_node(node_to_remove)

            # Update core numbers and degrees
            core_numbers = nx.core_number(G)
            degrees = dict(G.degree())

        return nodes_to_remove
