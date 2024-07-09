from typing import List
import networkx as nx

from network_dismantling.dismantler.dismantler import DismantlingStrategy


class CollectiveInfluenceDismantling(DismantlingStrategy):
    """
    Dismantling strategy that removes nodes based on their collective influence.

    This dismantling strategy removes nodes based on their collective influence, which is a measure of the influence of a
    node on its neighbors. The nodes with the highest collective influence are removed first.
    """

    def __init__(self, l=2):
        """
        Initialize the dismantling strategy.

        :param l: The radius of the ball to consider when computing the collective influence.
        """
        self.l = l  # Ball radius

    def dismantle(self, G: nx.Graph, num_nodes: int) -> List[int]:
        """
        Dismantle the graph by removing nodes based on their collective influence.

        :param G: The graph to dismantle.
        :param num_nodes: The number of nodes to remove.
        :return: A list of node indices to remove.
        """
        nodes_to_remove = []
        G_copy = G.copy()

        while len(nodes_to_remove) < num_nodes:
            ci_scores = self.compute_collective_influence(G_copy)
            if not ci_scores:
                break
            node_to_remove = max(ci_scores, key=ci_scores.get)
            nodes_to_remove.append(node_to_remove)
            G_copy.remove_node(node_to_remove)

        return nodes_to_remove

    def compute_collective_influence(self, G):
        """
        Compute the collective influence of each node in the graph.

        :param G: The graph.
        :return: A dictionary with the collective influence scores of each node.
        """
        ci_scores = {}
        for node in G.nodes():
            ki = G.degree(node)
            ball = nx.single_source_shortest_path_length(G, node, cutoff=self.l)
            ball_boundary = [n for n in ball if ball[n] == self.l]
            ci = (ki - 1) * sum(G.degree(v) - 1 for v in ball_boundary)
            ci_scores[node] = ci
        return ci_scores
