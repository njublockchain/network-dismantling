import networkx as nx
from typing import List
from network_dismantling.dismantler.dismantler import DismantlingStrategy


class BetweennessDismantling(DismantlingStrategy):
    """
    Dismantling strategy that removes nodes based on betweenness centrality.

    This dismantling strategy removes nodes based on their betweenness centrality, i.e., the number of shortest paths
    that pass through a node. The nodes with the highest betweenness centrality are removed first.
    """

    def dismantle(self, G: nx.Graph, num_nodes: int) -> List[int]:
        """
        Dismantle the graph by removing nodes based on betweenness centrality.

        :param G: The graph to dismantle.
        :param num_nodes: The number of nodes to remove.
        :return: A list of node indices to remove.
        """
        betweenness = nx.betweenness_centrality(G)
        nodes_to_remove = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[
            :num_nodes
        ]
        return [node for node, _ in nodes_to_remove]
