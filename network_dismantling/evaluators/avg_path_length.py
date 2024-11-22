import graph_tool.all as gt


from network_dismantling.evaluators.evaluator import EvaluationMetric


class AvgPathLengthMetric(EvaluationMetric):
    """
    Evaluation metric that computes the average path length of a graph.

    The average path length is a measure of the average distance between pairs of nodes in the graph. The average path
    length is the average of the shortest path lengths between all pairs of nodes in the graph.
    """

    def compute(self, G: gt.Graph) -> float:
        """
        Compute the average path length of the graph.

        :param G: The graph.
        :return: The average path length of the graph.
        """
        lcc = gt.extract_largest_component(G)
        distances = gt.shortest_distance(lcc)
        return distances[distances != float("inf")].a.mean()

    @property
    def name(self) -> str:
        """
        Return the name of the evaluation metric.

        :return: The name of the evaluation metric.
        """
        return "Average Path Length"
