# network-dismantling

`network-dismantling` is a library for network dismantling, which refers to the process of strategically removing nodes from a network to disrupt its functionality. This repository provides implementations of various network dismantling algorithms, leveraging the power of PyTorch for efficient computation and flexibility.

## Installation

To install the dependencies, easily install from PyPI:

```bash
pip install network-dismantling
```

Make sure you have PyTorch installed if you want to run a neural-network model.
You can install PyTorch from [here](https://pytorch.org/get-started/locally/).

Or directly using conda:

```bash
conda install network-dismantling
```

Alternatively, you can clone the repository and install it locally:

```bash
git clone https://github.com/njublockchain/network-dismantling.git
cd network-dismantling
pip install -e .
```

## Usage

```python
# Create a test graph
G = nx.karate_club_graph()

# Initialize CoreHD strategies
corehd_strategy = CoreHDDismantling()

# Create NetworkDismantlers with different strategies
corehd_dismantler = NetworkDismantler(corehd_strategy)

# Dismantle the graph using both strategies
num_nodes_to_remove = int(G.number_of_nodes() * 0.1)  # Remove 10% of nodes

dismantled_G_corehd, removed_nodes_corehd = corehd_dismantler.dismantle(G, num_nodes_to_remove)

print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"CoreHD dismantled graph: {dismantled_G_corehd.number_of_nodes()} nodes, {dismantled_G_corehd.number_of_edges()} edges")
print(f"CoreHD removed nodes: {removed_nodes_corehd}")
```

```python
G = nx.karate_club_graph()

# Initialize CoreGDM strategy
coregdm_strategy = CoreGDMDismantling()

# Create NetworkDismantler with CoreGDM strategy
coregdm_dismantler = NetworkDismantler(coregdm_strategy)

# Dismantle the graph using CoreGDM
num_nodes_to_remove = int(G.number_of_nodes() * 0.1)  # Remove 10% of nodes

dismantled_G_coregdm, removed_nodes_coregdm = coregdm_dismantler.dismantle(G, num_nodes_to_remove)

print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"CoreGDM dismantled graph: {dismantled_G_coregdm.number_of_nodes()} nodes, {dismantled_G_coregdm.number_of_edges()} edges")
print(f"CoreGDM removed nodes: {removed_nodes_coregdm}")
```

```python
    # Create a test graph
    G = nx.karate_club_graph()
    
    # Initialize different dismantling strategies
    dismantling_strategies = [
        CollectiveInfluenceDismantling(l=2),
        ExplosiveImmunizationDismantling(q=0.1, num_iterations=10),
        CoreGDMDismantling(),
        GNDDismantling(),
        CoreHDDismantling()
    ]
    
    # Initialize evaluation metrics
    metrics = [
        LCCSizeMetric(),
        NumComponentsMetric(),
        AvgPathLengthMetric(),
        GlobalEfficiencyMetric(),
        AvgClusteringMetric()
    ]
    
    # Choose evaluation strategy
    evaluation_strategy = RelativeChangeStrategy()
    
    # Create evaluator
    evaluator = DismantlingEvaluator(metrics, evaluation_strategy)
    
    # Compare strategies
    num_nodes_to_remove = int(G.number_of_nodes() * 0.1)  # Remove 10% of nodes
    comparison_results = DismantlingEvaluator.compare_strategies(G, dismantling_strategies, num_nodes_to_remove, evaluator)
    
    # Print detailed results
    for strategy_name, metrics in comparison_results.items():
        print(f"\n{strategy_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
```

## Documentation

The full documentation is available at [network-dismantling.readthedocs.io](https://network-dismantling.readthedocs.io).

## Contributing

We welcome contributions to the network-dismantling library! If you have an idea for an improvement or a new feature, feel free to open an issue or submit a pull request. Please make sure to follow our contributing guidelines.
