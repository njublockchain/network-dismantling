import networkx as nx
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from network_dismantling.dismantler.dismantler import DismantlingStrategy
from tqdm import tqdm

class CoreHDDismantling(DismantlingStrategy):
    def dismantle(self, G: nx.Graph, num_nodes: int) -> List[int]:
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
