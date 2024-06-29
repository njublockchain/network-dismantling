import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
from tqdm import tqdm

from network_dismantling.dismanlter.dismantler import DismantlingStrategy


class GATDismantlingLayer(nn.Module):
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


class GDMDismantling(DismantlingStrategy):
    def __init__(self, hidden_dim=64, num_layers=3, heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.model = None
        self.train_model()

    def train_model(self):
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
        self.model = GATDismantlingLayer(
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

    def dismantle(self, G: nx.Graph, num_nodes: int) -> List[int]:
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
        data = Data(x=self.extract_features(G), edge_index=from_networkx(G).edge_index)
        with torch.no_grad():
            importance = self.model(data)
        omega_s = sum(importance[node] for node in removed_nodes)
        omega_m = sum(sorted(importance, reverse=True)[: len(removed_nodes)])
        return min(omega_s / omega_m, 1.0)

    @staticmethod
    def extract_features(G):
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
