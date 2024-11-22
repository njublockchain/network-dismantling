import graph_tool.all as gt

from network_dismantling.evaluators.evaluator import EvaluationMetric


class LCCSizeMetric(EvaluationMetric):
    """
    Evaluation metric that computes the size of the largest connected component of a graph.

    The size of the largest connected component is a measure of the size of the largest connected subgraph of the graph.
    """

    def compute(self, G: gt.Graph) -> float:
        """
        Compute the size of the largest connected component of the graph.

        :param G: The graph.
        :return: The size of the largest connected component of the graph.
        """
        lcc = gt.extract_largest_component(G)
        return lcc.num_vertices()

    @property
    def name(self) -> str:
        """
        Return the name of the evaluation metric.

        :return: The name of the evaluation metric.
        """
        return "LCC Size"
