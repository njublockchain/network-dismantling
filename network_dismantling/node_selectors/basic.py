import itertools
import graph_tool.all as gt
from typing import List

from .node_selector import NodeSelector


class BruteForce(NodeSelector):
    """
    Dismantling strategy that removes nodes in a brute-force manner.

    This dismantling strategy removes nodes in a brute-force manner, by trying all possible combinations of nodes to
    remove and checking if the dismantling condition is met. The dismantling condition is met when the largest connected
    component of the graph is smaller than a target size ratio.
    """

    def __init__(self, target_size_ratio=0.1, max_depth=None):
        """
        Initialize the dismantling strategy.

        :param target_size_ratio: The target size ratio of the largest connected component after dismantling.
        :param max_depth: The maximum depth of the brute-force search. If not set, the number of nodes to remove is used.
        """
        self.target_size_ratio = target_size_ratio
        self.max_depth = max_depth

    def select(self, G: gt.Graph, num_nodes: int) -> List[int]:
        """
        Select nodes to remove from the graph.

        :param G: The graph to dismantle.
        :param num_nodes: The number of nodes to remove.
        :return: A list of node indices to remove.
        """
        original_size = G.number_of_nodes()
        target_size = int(original_size * self.target_size_ratio)

        # If max_depth is not set, use num_nodes
        max_depth = self.max_depth if self.max_depth is not None else num_nodes

        for k in range(1, min(max_depth, num_nodes) + 1):
            for combination in itertools.combinations(G.nodes(), k):
                H = G.copy()
                H.remove_nodes_from(combination)
                if self.is_dismantled(H, target_size):
                    return list(combination)

        # If no solution found within max_depth, return the best found so far
        return self.greedy_fallback(G, num_nodes)

    def is_dismantled(self, G: gt.Graph, target_size: int) -> bool:
        """
        Check if the graph is dismantled.

        :param G: The graph to check.
        :param target_size: The target size of the largest connected component.
        :return: True if the largest connected component is smaller than the target size, False otherwise.
        """
        lcc = gt.extract_largest_component(G)
        return lcc.num_vertices() <= target_size

    def greedy_fallback(self, G: gt.Graph, num_nodes: int) -> List[int]:
        """
        Greedy fallback dismantling strategy.

        This dismantling strategy removes nodes greedily, by removing the node with the highest degree until the target
        number of nodes is removed.

        :param G: The graph to dismantle.
        :param num_nodes: The number of nodes to remove.
        :return: A list of node indices to remove.
        """
        nodes_to_remove = []
        G_copy = G.copy()
        for _ in range(num_nodes):
            node_to_remove = max(G_copy.nodes(), key=G_copy.degree)
            nodes_to_remove.append(node_to_remove)
            G_copy.remove_node(node_to_remove)
        return nodes_to_remove
