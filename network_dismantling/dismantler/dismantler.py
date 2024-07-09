import networkx as nx
from abc import ABC, abstractmethod
from typing import List, Tuple


class DismantlingStrategy(ABC):
    """
    Abstract class for dismantling strategies.

    A dismantling strategy is a method to remove nodes from a graph in order to dismantle it.
    """

    @abstractmethod
    def dismantle(self, G: nx.Graph, num_nodes: int) -> List[int]:
        """
        Dismantle the graph by removing nodes.

        :param G: The input graph
        :param num_nodes: Number of nodes to remove
        :return: List of node IDs to remove
        """
        pass


class NetworkDismantler:
    """
    Class to dismantle a network using a specified strategy.
    """

    def __init__(self, strategy: DismantlingStrategy):
        """
        Initialize the dismantler with the specified strategy.

        :param strategy: The dismantling strategy to use
        """
        self.strategy = strategy

    def dismantle(self, G: nx.Graph, num_nodes: int) -> Tuple[nx.Graph, List[int]]:
        """
        Dismantle the graph using the specified strategy.

        :param G: The input graph
        :param num_nodes: Number of nodes to remove
        :return: Tuple of (dismantled graph, list of removed nodes)
        """
        nodes_to_remove = self.strategy.dismantle(G, num_nodes)
        H = G.copy()
        H.remove_nodes_from(nodes_to_remove)
        return H, nodes_to_remove


# # Example usage
# if __name__ == "__main__":
#     # Create a sample graph
#     G = nx.karate_club_graph()

#     # Create dismantler with degree strategy
#     degree_dismantler = NetworkDismantler(DegreeDismantling())

#     # Dismantle the graph
#     dismantled_G, removed_nodes = degree_dismantler.dismantle(G, 5)

#     print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
#     print(f"Dismantled graph: {dismantled_G.number_of_nodes()} nodes, {dismantled_G.number_of_edges()} edges")
#     print(f"Removed nodes: {removed_nodes}")
