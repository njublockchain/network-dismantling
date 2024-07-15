import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
from tqdm import tqdm

from .node_selector import NodeSelector


class GATLayer:
    """
    A simple GAT model that predicts the score of each node in the graph.

    This model is a simple GAT model that predicts the score of each node in the graph. The model consists of multiple
    GAT layers with ELU activation functions, followed by a final GAT layer that outputs the score of each node.
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=3, heads=8):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(input_dim, hidden_dim, heads=heads))
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
        self.layers.append(GATConv(hidden_dim * heads, 1, heads=1))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.layers[:-1]:
            x = F.elu(layer(x, edge_index))
        x = self.layers[-1](x, edge_index)
        return torch.sigmoid(x).view(-1)


class GDM(NodeSelector):
    """
    Dismantling strategy that removes nodes based on their GDM score.

    This dismantling strategy removes nodes based on their GDM score, which is a measure of the importance of a node in
    the graph. The nodes with the highest GDM score are removed first.
    """

    def __init__(self, hidden_dim=64, num_layers=3, heads=8):
        """
        Initialize the dismantling strategy.

        :param hidden_dim: The dimension of the hidden layers.
        :param num_layers: The number of hidden layers.
        :param heads: The number of attention heads in the GAT layers.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.model = None
        self.train_model()

    def train_model(self):
        """
        Train the GDM model.

        This method trains the GDM model on synthetic networks to predict the importance of each node in the graph.
        """
        # Generate synthetic networks for training
        train_networks = self.generate_synthetic_networks(1000, 25)

        # Prepare training data
        train_data = []
        for G in tqdm(train_networks, desc="Preparing GDM training data"):
            data = Data(
                x=self.extract_features(G), edge_index=from_networkx(G).edge_index
            )
            optimal_set = set(
                sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[
                    : int(0.18 * G.number_of_nodes())
                ]
            )
            data.y = torch.tensor([1.0 if n in optimal_set else 0.0 for n in G.nodes()])
            if torch.cuda.is_available():
                data = data.to("cuda")
            train_data.append(data)

        # Initialize and train the model
        self.model = GATLayer(
            input_dim=4,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            heads=self.heads,
        )
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        for epoch in tqdm(range(100), desc="Training GDM model"):
            self.model.train()
            total_loss = 0
            for data in train_data:
                optimizer.zero_grad()
                out = self.model(data)
                loss = F.mse_loss(out, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(train_data)
            if (epoch + 1) % 10 == 0:
                tqdm.write(f"Training GDM Epoch {epoch+1}/100, Loss: {loss:.4f}")

    def select(self, G: nx.Graph, num_nodes: int) -> List[int]:
        """
        Dismantle the graph by removing nodes based on their GDM score.

        :param G: The graph to dismantle.
        :param num_nodes: The number of nodes to remove.
        :return: A list of node indices to remove.
        """
        self.model.eval()
        data = Data(x=self.extract_features(G), edge_index=from_networkx(G).edge_index)
        if torch.cuda.is_available():
            data = data.to("cuda")

        with torch.no_grad():
            importance = self.model(data)
        nodes_to_remove = sorted(
            enumerate(importance), key=lambda x: x[1], reverse=True
        )[:num_nodes]
        return [node for node, _ in nodes_to_remove]

    def compute_omega(self, G: nx.Graph, removed_nodes: List[int]) -> float:
        """
        Compute the GDM dismantling efficiency of the removed nodes.

        :param G: The original graph.
        :param removed_nodes: The nodes that have been removed.
        :return: The GDM dismantling efficiency of the removed nodes.
        """
        data = Data(x=self.extract_features(G), edge_index=from_networkx(G).edge_index)
        with torch.no_grad():
            importance = self.model(data)
        omega_s = sum(importance[node] for node in removed_nodes)
        omega_m = sum(sorted(importance, reverse=True)[: len(removed_nodes)])
        return min(omega_s / omega_m, 1.0)

    @staticmethod
    def extract_features(G):
        """
        Extract node features from the graph.

        :param G: The graph.
        :return: A tensor of node features.
        """
        features = []
        for node in G.nodes():
            degree = G.degree(node)
            clustering = nx.clustering(G, node)
            k_core = nx.core_number(G)[node]
            chi_square = (
                sum((G.degree(n) - degree) ** 2 for n in G.neighbors(node)) / degree
                if degree > 0
                else 0
            )
            features.append([degree, clustering, k_core, chi_square])
        return torch.tensor(features, dtype=torch.float)

    @staticmethod
    def generate_synthetic_networks(num_networks, num_nodes):
        """
        Generate synthetic networks for training the GDM model.

        :param num_networks: The number of synthetic networks to generate.
        :param num_nodes: The number of nodes in each synthetic network.
        :return: A list of synthetic networks.
        """
        networks = []
        for _ in range(num_networks):
            if np.random.random() < 0.33:
                G = nx.barabasi_albert_graph(num_nodes, 3)
            elif np.random.random() < 0.66:
                G = nx.erdos_renyi_graph(num_nodes, 0.1)
            else:
                G = nx.watts_strogatz_graph(num_nodes, 4, 0.1)
            networks.append(G)
        return networks


class GCNScoreModel(nn.Module):
    """
    A simple GCN model that predicts the score of each node in the graph.

    This model is a simple GCN model that predicts the score of each node in the graph. The model consists of multiple
    GCN layers with ReLU activation functions, followed by a final GCN layer that outputs the score of each node.
    """

    def __init__(self, input_dim, hidden_dim, num_layers):
        """
        Initialize the GCN model.

        :param input_dim: The dimension of the input features.
        :param hidden_dim: The dimension of the hidden layers.
        :param num_layers: The number of hidden layers.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.layers.append(GCNConv(hidden_dim, 1))

    def forward(self, data):
        """
        Forward pass of the GCN model.

        :param data: The input data.
        :return: The predicted scores of each node.
        """
        x, edge_index = data.x, data.edge_index
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, edge_index))
        x = self.layers[-1](x, edge_index)
        return x


class CoreGDM(NodeSelector):
    def __init__(self, hidden_dim=64, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model = GCNScoreModel(
            input_dim=4, hidden_dim=hidden_dim, num_layers=num_layers
        )
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.train_model()

    def select(self, G: nx.Graph, num_nodes: int) -> List[int]:
        nodes_to_remove = []
        G_copy = G.copy()

        while len(nodes_to_remove) < num_nodes:
            # Get 2-core of the network
            two_core = nx.k_core(G_copy, k=2)
            if len(two_core) == 0:
                # If 2-core is empty, remove nodes from the remaining graph
                remaining_nodes = list(G_copy.nodes())
                nodes_to_remove.extend(
                    remaining_nodes[: num_nodes - len(nodes_to_remove)]
                )
                break

            # Prepare data for the model
            data = self.prepare_data(two_core)
            if torch.cuda.is_available():
                data = data.to("cuda")

            # Get scores from the model
            self.model.eval()
            with torch.no_grad():
                scores = self.model(data).squeeze()

            # Select node with highest score
            node_to_remove = list(two_core.nodes)[scores.argmax().item()]
            nodes_to_remove.append(node_to_remove)
            G_copy.remove_node(node_to_remove)

        return nodes_to_remove

    def train_model(self):
        # Generate synthetic networks for training
        train_networks = self.generate_synthetic_networks(1000, 100)

        for epoch in tqdm(range(100), desc="Training CoreGDM model"):
            total_loss = 0
            for G in train_networks:
                two_core = nx.k_core(G, k=2)
                if len(two_core) == 0:
                    continue

                data = self.prepare_data(two_core)
                if torch.cuda.is_available():
                    data = data.to("cuda")

                self.optimizer.zero_grad()
                scores = self.model(data).squeeze()

                # Use CoreHD as the target
                target_scores = torch.tensor(
                    [two_core.degree(n) for n in two_core.nodes()], dtype=torch.float32
                )
                if torch.cuda.is_available():
                    target_scores = target_scores.to("cuda")
                loss = F.mse_loss(scores, target_scores)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                tqdm.write(
                    f"Training CoreGDM Epoch {epoch+1}/100, Loss: {total_loss/len(train_networks):.4f}"
                )

    def prepare_data(self, G):
        features = []
        clustering_mapping = nx.clustering(G)
        eigenvector_centrality_mapping = nx.eigenvector_centrality_numpy(G)
        core_number_mapping = nx.core_number(G)
        for node in G.nodes():
            degree = G.degree(node)
            clustering = clustering_mapping[node]
            core_number = core_number_mapping[node]
            eigenvector_centrality = eigenvector_centrality_mapping[node]
            features.append([degree, clustering, core_number, eigenvector_centrality])

        return Data(
            x=torch.tensor(features, dtype=torch.float32),
            edge_index=from_networkx(G).edge_index,
        )

    @staticmethod
    def generate_synthetic_networks(num_networks, num_nodes):
        networks = []
        for _ in range(num_networks):
            if np.random.random() < 0.33:
                G = nx.barabasi_albert_graph(num_nodes, 3)
            elif np.random.random() < 0.66:
                G = nx.erdos_renyi_graph(num_nodes, 0.1)
            else:
                G = nx.watts_strogatz_graph(num_nodes, 4, 0.1)
            networks.append(G)
        return networks
