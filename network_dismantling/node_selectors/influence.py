import graph_tool.all as gt

from collections import defaultdict
from typing import List

from network_dismantling.node_selectors.node_selector import NodeSelector

class CollectiveInfluence:
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

    def select(self, G: gt.Graph, num_elements: int) -> List[int]:
        """
        Dismantle the graph by removing nodes based on their collective influence.

        :param G: The graph to dismantle.
        :param num_elements: The number of nodes to remove.
        :return: A list of node indices to remove.
        """
        num_nodes = num_elements
        nodes_to_remove: List[int] = []
        G_copy = G.copy()

        while len(nodes_to_remove) < num_nodes:
            ci_scores = self.compute_collective_influence(G_copy)
            if not ci_scores:
                break
            node_to_remove = max(ci_scores, key=ci_scores.get)
            nodes_to_remove.append(int(node_to_remove))
            G_copy.remove_vertex(node_to_remove, fast=True)

        return nodes_to_remove

    def compute_collective_influence(self, G: gt.Graph):
        """
        Compute the collective influence of each node in the graph.

        :param G: The graph.
        :return: A dictionary with the collective influence scores of each node.
        """
        ci_scores = {}
        degrees = G.degree_property_map("total")
        for node in G.vertices():
            ki = degrees[node]
            if ki <= 1:
                ci_scores[node] = 0
                continue
            # Compute distances from node to all other nodes
            dist_map = gt.shortest_distance(G, source=node, max_dist=self.l).a
            # Get vertices at distance exactly l
            ball_boundary = [v for v in G.vertices() if dist_map[int(v)] == self.l]
            # Compute the sum over the degrees of the boundary nodes
            sum_kj_minus1 = sum(degrees[v] - 1 for v in ball_boundary)
            ci = (ki - 1) * sum_kj_minus1
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

    def select(self, G: gt.Graph, num_elements: int) -> List[int]:
        """
        Dismantle the graph by removing nodes based on their explosive immunization score.

        :param G: The graph to dismantle.
        :param num_elements: The number of nodes to remove.
        :return: A list of node indices to remove.
        """
        num_nodes = num_elements
        nodes_to_remove: List[int] = []
        G_copy = G.copy()
        mask = G_copy.new_vertex_property("bool", val=True)

        while len(nodes_to_remove) < num_nodes:
            G_view = gt.GraphView(G_copy, vfilt=mask)
            scores = self.compute_ei_scores(G_view)
            if not scores:
                break
            candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            num_to_remove = min(
                max(1, int(self.q * G_view.num_vertices())), num_nodes - len(nodes_to_remove)
            )
            for node_index, _ in candidates[:num_to_remove]:
                nodes_to_remove.append(node_index)
                mask[node_index] = False  # Mark node as removed

        return nodes_to_remove

    def compute_ei_scores(self, G: gt.GraphView):
        """
        Compute the explosive immunization score of each node in the graph.

        :param G: The graph view.
        :return: A dictionary with the explosive immunization scores of each node.
        """
        scores = defaultdict(float)
        for _ in range(self.num_iterations):
            comp, _ = gt.label_components(G)
            # 使用 defaultdict 来存储组件
            components = defaultdict(list)
            for v in G.vertices():
                comp_label = comp[v]
                components[comp_label].append(int(v))
            for component in components.values():
                if len(component) > 1:
                    subgraph_mask = G.new_vertex_property("bool", val=False)
                    for idx in component:
                        subgraph_mask[idx] = True
                    subgraph = gt.GraphView(G, vfilt=subgraph_mask)
                    for node_index in component:
                        s = self.compute_s(subgraph, node_index)
                        scores[node_index] += s / self.num_iterations
        return scores

    def compute_s(self, G: gt.GraphView, node_index):
        """
        Compute the explosive immunization score of a node.

        :param G: The graph view.
        :param node_index: The index of the node.
        :return: The explosive immunization score of the node.
        """
        mask = G.new_vertex_property("bool")
        mask.a = G.get_vertex_filter()[0].a.copy()
        mask[node_index] = False  # Simulate removal of the node
        G_copy = gt.GraphView(G, vfilt=mask)
        
        if G_copy.num_vertices() == 0:
            return 0  # return 0 when graph is empty

        comp, _ = gt.label_components(G_copy)
        component_sizes = defaultdict(int)
        for v in G_copy.vertices():
            comp_label = comp[v]
            component_sizes[comp_label] += 1
        return sum(size * (size - 1) for size in component_sizes.values())
    