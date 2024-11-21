from typing import List
import graph_tool.all as gt
import os
import numpy as np
import requests
from tqdm import tqdm


def _download_with_tqdm(url: str, fname: str):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))

    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


class RealWorldDatasetFetcher:
    def __init__(
        self,
        dataset_name: str,
        dataset_path_prefix: str = "./network_dismantling_data",
    ):
        self.dataset_name = dataset_name + ".gt.xz"
        self.dataset_path = os.path.abspath(
            os.path.join(dataset_path_prefix, "realworld", dataset_name)
        )

    def fetch(self):
        url = (
            "https://github.com/NetworkDismantling/review/raw/refs/heads/main/dataset/test_review/"
            + self.dataset_name
            + ".gt.xz"
        )
        # download with progress bar
        if not os.path.exists(self.dataset_path):
            print(
                f"Downloading {self.dataset_name} dataset and saving it to {self.dataset_path}"
            )
            _download_with_tqdm(url, self.dataset_path)
            print("Download completed")

    def load(self):
        return gt.load_graph(self.dataset_path)


class SyntheticDatasetGenerator:
    def __init__(
        self,
        dataset_name: str,
        dataset_path_prefix: str = "./network_dismantling_data",
    ):
        self.dataset_name = dataset_name + ".gt.xz"
        self.dataset_path = os.path.abspath(
            os.path.join(dataset_path_prefix, "synthetic", dataset_name)
        )

    @staticmethod
    def generate_synthetic_networks(num_networks: int, num_nodes: int) -> List[gt.Graph]:
        """
        Generate synthetic networks for training the GDM model using graph-tool.

        Args:
            num_networks: The number of synthetic networks to generate
            num_nodes: The number of nodes in each synthetic network
        
        Returns:
            List of synthetic networks as graph-tool Graph objects
        """
        networks = []
        for _ in range(num_networks):
            rand = np.random.random()
            if rand < 0.33:
                # Barabási-Albert model using Price network
                # m=3 new edges per vertex
                G = gt.price_network(num_nodes, m=3, directed=False)
                
            elif rand < 0.66:
                # Erdős-Rényi model 
                # p=0.1 edge probability
                G = gt.random_graph(num_nodes, 
                                lambda: np.random.binomial(1, 0.1), 
                                directed=False)
                
            else:
                # Watts-Strogatz model
                # First create regular ring lattice
                G = gt.lattice([num_nodes], dim=1, k=4, periodic=True)
                # Then rewire with probability 0.1
                gt.random_rewire(G, model="erdos", 
                            n_iter=G.num_edges()*0.1,
                            edge_sweep=True)
                
            networks.append(G)
        return networks

    def load(self):
        return gt.load_graph(self.dataset_path)