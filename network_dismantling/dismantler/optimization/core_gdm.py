import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import numpy as np
from typing import List
from tqdm import tqdm

from network_dismantling.dismantler.dismantler import DismantlingStrategy


class GCNScoreModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.layers.append(GCNConv(hidden_dim, 1))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, edge_index))
        x = self.layers[-1](x, edge_index)
        return x


class CoreGDMDismantling(DismantlingStrategy):
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

    def dismantle(self, G: nx.Graph, num_nodes: int) -> List[int]:
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
                tqdm.write(f"Training CoreGDM Epoch {epoch+1}/100, Loss: {total_loss/len(train_networks):.4f}")

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
