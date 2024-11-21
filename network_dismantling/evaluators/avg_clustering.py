import graph_tool.all as gt

from network_dismantling.evaluators.evaluator import EvaluationMetric


class AvgClusteringMetric(EvaluationMetric):
    """
    Evaluation metric that computes the average clustering coefficient of a graph.

    The average clustering coefficient is a measure of the degree to which nodes in a graph tend to cluster together.
    The average clustering coefficient is the average of the clustering coefficients of all nodes in the graph.
    """

    def compute(self, G: gt.Graph) -> float:
        """
        Compute the average clustering coefficient of the graph.

        :param G: The graph.
        :return: The average clustering coefficient of the graph.
        """
        return gt.average_clustering(G)

    @property
    def name(self) -> str:
        """
        Return the name of the evaluation metric.

        :return: The name of the evaluation metric.
        """
        return "Average Clustering"
