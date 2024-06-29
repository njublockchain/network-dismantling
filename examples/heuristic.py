import networkx as nx
from network_dismantling.dismanlter.influence import (
    CollectiveInfluenceDismantling,
    ExplosiveImmunizationDismantling,
)
from network_dismantling.evaluator import (
    DismantlingEvaluator,
    RelativeChangeStrategy,
    LCCSizeMetric,
    NumComponentsMetric,
    GlobalEfficiencyMetric,
    AvgPathLengthMetric,
    AvgClusteringMetric,
)


# Example usage
if __name__ == "__main__":
    # Create a test graph
    G = nx.karate_club_graph()

    # Initialize different dismantling strategies
    dismantling_strategies = [
        CollectiveInfluenceDismantling(l=2),
        ExplosiveImmunizationDismantling(q=0.1, num_iterations=10),
    ]

    # Initialize evaluation metrics
    metrics = [
        LCCSizeMetric(),
        NumComponentsMetric(),
        AvgPathLengthMetric(),
        GlobalEfficiencyMetric(),
        AvgClusteringMetric(),
    ]

    # Choose evaluation strategy
    evaluation_strategy = RelativeChangeStrategy()

    # Create evaluator
    evaluator = DismantlingEvaluator(metrics, evaluation_strategy)

    # Compare strategies
    num_nodes_to_remove = int(G.number_of_nodes() * 0.1)  # Remove 10% of nodes
    comparison_results = DismantlingEvaluator.compare_strategies(
        G, dismantling_strategies, num_nodes_to_remove, evaluator
    )

    # Print detailed results
    for strategy_name, metrics in comparison_results.items():
        print(f"\n{strategy_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
