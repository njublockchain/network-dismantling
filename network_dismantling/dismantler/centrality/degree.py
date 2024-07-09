import networkx as nx
from typing import List
from network_dismantling.dismantler.dismantler import DismantlingStrategy


class DegreeDismantling(DismantlingStrategy):
    """
    Dismantling strategy that removes nodes based on their degree centrality.

    This dismantling strategy removes nodes based on their degree centrality, i.e., the number of edges connected to a
    node. The nodes with the highest degree centrality are removed first.
    """

    def dismantle(self, G: nx.Graph, num_nodes: int) -> List[int]:
        nodes_to_remove = sorted(G.degree, key=lambda x: x[1], reverse=True)[:num_nodes]
        return [node for node, _ in nodes_to_remove]
