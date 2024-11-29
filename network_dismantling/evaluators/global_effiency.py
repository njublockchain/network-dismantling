import graph_tool.all as gt
import numpy as np

from network_dismantling.evaluators.evaluator import EvaluationMetric

class GlobalEfficiencyMetric(EvaluationMetric):
    """
    Evaluation metric that computes the global efficiency of a graph.

    The global efficiency is a measure of the efficiency of information transfer in a graph.
    It is defined as the average of the inverse shortest path lengths between all pairs of nodes
    in the graph.
    """

    def compute(self, G: gt.Graph) -> float:
        """
        Compute the global efficiency of the graph.

        :param G: The graph.
        :return: The global efficiency of the graph.
        """
        N = G.num_vertices()
        if N <= 1:
            return 0.0  # Efficiency is zero for graphs with one or no nodes.

        # Compute the shortest distances between all pairs of nodes
        dist_matrix = gt.shortest_distance(G, weights=None, directed=False)

        # Initialize efficiency sum
        efficiency_sum = 0.0

        # Iterate over all pairs of vertices
        for v in G.vertices():
            # Extract distances as a NumPy array and convert to float type
            distances = dist_matrix[v].get_array().astype(float)
            # Exclude self-distance by setting it to infinity
            distances[int(v)] = np.inf
            # Compute inverse distances, handling infinities
            with np.errstate(divide='ignore', invalid='ignore'):
                inv_distances = 1.0 / distances
                inv_distances[~np.isfinite(inv_distances)] = 0.0  # Set infinities and NaNs to zero
            # Sum up the efficiencies for this vertex
            efficiency_sum += np.sum(inv_distances)

        # Calculate global efficiency
        global_efficiency = efficiency_sum / (N * (N - 1))

        return global_efficiency

    @property
    def name(self) -> str:
        """
        Return the name of the evaluation metric.

        :return: The name of the evaluation metric.
        """
        return "Global Efficiency"
    