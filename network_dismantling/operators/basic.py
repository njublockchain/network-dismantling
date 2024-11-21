from collections import deque
import networkx as nx

from typing import Any, List
from network_dismantling.operators.operator import DismantlingOperator


class NodeRemovalOperator(DismantlingOperator):
    """
    Dismantling operator that removes nodes from the graph.

    This dismantling operator removes nodes from the graph.
    """

    def operate(self, G: nx.Graph, elements: List[Any]) -> nx.Graph:
        """
        Remove nodes from the graph.

        :param G: The input graph
        :param elements: Nodes to remove
        :return: The modified graph
        """
        H = G.copy()
        H.remove_nodes_from(elements)
        return H

class TemporalNodeNeighborEgdesRemovalOperator(DismantlingOperator):
    def operate(self, G: nx.Graph, elements: List[Any]) -> nx.Graph:
        """
        Remove nodes from the graph.

        :param G: The input graph
        :param elements: Nodes to remove
        :return: The modified graph
        """
        H = G.copy()
        H = self.dismantling_from_nodes(H, elements)
        return H

    # cannot use the original data, because the original data does not have the message, time, and message_type attributes
    def dismantling_from_nodes(
        G,
        nodes,
        message_type_label="message_type",
        time_label="time",
        message_label="message",
    ):
        G_dismantled = G.copy()

        queue = deque(nodes)
        processed_nodes = set()

        for node in nodes:
            # add neighbors to the queue
            for neighbor in G.neighbors(node):
                if neighbor not in processed_nodes:
                    queue.append(neighbor)
                    processed_nodes.add(neighbor)

        peripheral_nodes = [
            node
            for node in G_dismantled.nodes()
            if G_dismantled.in_degree(node) == 0 or G_dismantled.out_degree(node) == 0
        ]

        # remove the edges of the role nodes
        edges_to_remove = list(G_dismantled.edges(nodes))
        # check whether the first in edge is removed, if removed, remove the all edges of the node
        G_dismantled.remove_edges_from(edges_to_remove)

        while queue:
            node = queue.popleft()
            if node in processed_nodes:
                continue

            if node in peripheral_nodes:
                # if the node is a peripheral node, means it may have external edges, so skip
                continue

            # get all out edges and in edges of the node
            out_edges = list(G_dismantled.out_edges(node, data=True, keys=True))
            in_edges = list(G_dismantled.in_edges(node, data=True, keys=True))

            # check whether the first in edge is removed, if removed, remove the all edges of the node
            for u, v, k, out_data in out_edges:
                message_type = out_data["message_type"]
                out_time = out_data["time"]
                out_val = out_data["message"]
                # check whether prev in_edges sum message is greater than the out_val
                is_valid = (
                    sum(
                        [
                            in_data[message_label]
                            for _, _, _, in_data in in_edges
                            if in_data[message_type_label] == message_type
                            and in_data[time_label] < out_time
                        ]
                    )
                    >= out_val
                )

                # if there is no enough balance, remove the out edge
                if not is_valid:
                    # additional_edges_to_remove.append((u, v, k))
                    G_dismantled.remove_edge(u, v, k)

                    # add the destination node to the queue
                    if v not in processed_nodes:
                        queue.append(v)
                        processed_nodes.add(v)

        return G_dismantled


class EdgeRemovalOperator(DismantlingOperator):
    """
    Dismantling operator that removes edges from the graph.

    This dismantling operator removes edges from the graph.
    """

    def operate(self, G: nx.Graph, elements: List[Any]) -> nx.Graph:
        """
        Remove edges from the graph.

        :param G: The input graph
        :param elements: Edges to remove
        :return: The modified graph
        """
        H = G.copy()
        H.remove_edges_from(elements)
        return H

class TemporalEdgeRemovalOperator(DismantlingOperator):
    """
    Dismantling operator that removes edges from the graph.

    This dismantling operator removes edges from the graph.
    """

    def operate(self, G: nx.Graph, elements: List[Any]) -> nx.Graph:
        """
        Remove edges from the graph.

        :param G: The input graph
        :param elements: Edges to remove
        :return: The modified graph
        """
        H = G.copy()
        H = self.dismantling_from_edges(H, elements)
        return H

    def dismantling_from_edges(
        G,
        edges,
        message_type_label="message_type",
        time_label="time",
        message_label="message",
    ):
        G_dismantled = G.copy()

        queue = deque(edges)
        processed_edges = set()

        for edge in edges:
            # add neighbors to the queue
            for neighbor in G.neighbors(edge[0]):
                if neighbor not in processed_edges:
                    queue.append(neighbor)
                    processed_edges.add(neighbor)

        peripheral_nodes = [
            node
            for node in G_dismantled.nodes()
            if G_dismantled.in_degree(node) == 0 or G_dismantled.out_degree(node) == 0
        ]

        while queue:
            edge = queue.popleft()
            if edge in processed_edges:
                continue

            if edge in peripheral_nodes:
                # if the node is a peripheral node, means it may have external edges, so skip
                continue

            # get all out edges and in edges of the node
            out_edges = list(G_dismantled.out_edges(edge[0], data=True, keys=True))
            in_edges = list(G_dismantled.in_edges(edge[0], data=True, keys=True))

            # check whether the first in edge is removed, if removed, remove the all edges of the node
            for u, v, k, out_data in out_edges:
                message_type = out_data["message_type"]
                out_time = out_data["time"]
                out_val = out_data["message"]
                # check whether prev in_edges sum message is greater than the out_val
                is_valid = (
                    sum(
                        [
                            in_data[message_label]
                            for _, _, _, in_data in in_edges
                            if in_data[message_type_label] == message_type
                            and in_data[time_label] < out_time
                        ]
                    )
                    >= out_val
                )

                # if there is no enough balance, remove the out edge
                if not is_valid:
                    # additional_edges_to_remove.append((u, v, k))
                    G_dismantled.remove_edge(u, v, k)

                    # add the destination node to the queue
                    if v not in processed_edges:
                        queue.append(v)
                        processed_edges.add(v)

        return G_dismantled