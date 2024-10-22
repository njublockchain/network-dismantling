import networkx as nx
from network_dismantling import NetworkDismantler
from network_dismantling.operators.basic import NodeRemovalOperator
from network_dismantling.node_selectors import (
    CoreHD,
    GND,
    GDM,
    CoreGDM,
)
from network_dismantling.evaluators import (
    DismantlingEvaluator,
    RelativeChangeStrategy,
    LCCSizeMetric,
    NumComponentsMetric,
    GlobalEfficiencyMetric,
    AvgPathLengthMetric,
    AvgClusteringMetric,
)
from network_dismantling.utils.early_stopper import SimpleEarlyStopper


# Example usage
if __name__ == "__main__":
    # Create a test graph
    G = nx.karate_club_graph()

    # Initialize different dismantling strategies
    dismantlers = [
        NetworkDismantler(selector=GND(), operator=NodeRemovalOperator()),
        NetworkDismantler(
            selector=CoreHD(),
            operator=NodeRemovalOperator(),
        ),
        NetworkDismantler(
            selector=GDM(early_stopper=SimpleEarlyStopper()),
            operator=NodeRemovalOperator(),
        ),
        NetworkDismantler(
            selector=CoreGDM(early_stopper=SimpleEarlyStopper()),
            operator=NodeRemovalOperator(),
        ),
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
        G, dismantlers, num_nodes_to_remove, evaluator
    )

    # Print detailed results
    for strategy_name, metrics in comparison_results.items():
        print(f"\n{strategy_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
