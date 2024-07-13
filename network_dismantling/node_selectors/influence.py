from collections import defaultdict
from typing import List
import networkx as nx

from .node_selector import NodeSelector


class CollectiveInfluence(NodeSelector):
    """
    Dismantling strategy that removes nodes based on their collective influence.

    This dismantling strategy removes nodes based on their collective influence, which is a measure of the influence of a node on its neighbors.
    The nodes with the highest collective influence are removed first.
    """

    def __init__(self, l=2):  # noqa: E741
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
        nodes_to_remove: List[int] = []
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


class ExplosiveImmunization(NodeSelector):
    """
    Dismantling strategy that removes nodes based on their explosive immunization score.

    This dismantling strategy removes nodes based on their explosive immunization score, which is a measure of the
    influence of a node on the connectivity of the graph. The nodes with the highest explosive immunization score are
    removed first.
    """

    def __init__(self, q=0.1, num_iterations=10):
        """
        Initialize the dismantling strategy.

        :param q: Fraction of nodes to remove in each iteration.
        :param num_iterations: Number of iterations to compute the explosive immunization score.
        """
        self.q = q  # Fraction of nodes to remove in each iteration
        self.num_iterations = num_iterations

    def dismantle(self, G: nx.Graph, num_nodes: int) -> List[int]:
        """
        Dismantle the graph by removing nodes based on their explosive immunization score.

        :param G: The graph to dismantle.
        :param num_nodes: The number of nodes to remove.
        :return: A list of node indices to remove.
        """
        nodes_to_remove: List[int] = []
        G_copy = G.copy()

        while len(nodes_to_remove) < num_nodes:
            scores = self.compute_ei_scores(G_copy)
            if not scores:
                break
            candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            num_to_remove = min(
                int(self.q * len(G_copy)), num_nodes - len(nodes_to_remove)
            )
            for node, _ in candidates[:num_to_remove]:
                nodes_to_remove.append(node)
                G_copy.remove_node(node)

        return nodes_to_remove

    def compute_ei_scores(self, G):
        """
        Compute the explosive immunization score of each node in the graph.

        :param G: The graph.
        :return: A dictionary with the explosive immunization scores of each node.
        """
        scores = defaultdict(float)
        for _ in range(self.num_iterations):
            components = list(nx.connected_components(G))
            for component in components:
                if len(component) > 1:
                    subgraph = G.subgraph(component)
                    for node in component:
                        s = self.compute_s(subgraph, node)
                        scores[node] += s / self.num_iterations
        return scores

    def compute_s(self, G, node):
        """
        Compute the explosive immunization score of a node.

        :param G: The graph.
        :param node: The node.
        :return: The explosive immunization score of the node.
        """
        G_copy = G.copy()
        G_copy.remove_node(node)
        new_components = list(nx.connected_components(G_copy))
        return sum(len(c) * (len(c) - 1) for c in new_components)
