import networkx as nx
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from network_dismantling.dismanlter.dismantler import DismantlingStrategy


class GNDDismantling(DismantlingStrategy):
    def __init__(self, c=1, epsilon=0.01, max_iterations=100):
        self.c = c
        self.epsilon = epsilon
        self.max_iterations = max_iterations

    def dismantle(self, G: nx.Graph, num_nodes: int) -> List[int]:
        A = nx.adjacency_matrix(G)
        N = A.shape[0]
        D = np.array(A.sum(axis=1)).flatten()
        L = csr_matrix(np.diag(D) - A)

        x = np.random.rand(N)
        x = x / np.linalg.norm(x)

        for _ in range(self.max_iterations):
            x_old = x
            x = L.dot(x) + self.c * x
            x = x / np.linalg.norm(x)
            if np.linalg.norm(x - x_old) < self.epsilon:
                break

        scores = np.abs(x)
        nodes_to_remove = np.argsort(scores)[::-1][:num_nodes]
        return nodes_to_remove.tolist()
