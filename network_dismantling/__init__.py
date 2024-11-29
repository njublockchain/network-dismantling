from network_dismantling.dismantler import DismantlingStrategy, NetworkDismantler
from network_dismantling.operators.operator import DismantlingOperator
from network_dismantling.selector import ElementSelector


def require_gt():
    """
    Check graph-tool is installed
    """
    try:
        import graph_tool.all as gt

        _ = gt
    except ImportError as exc:
        raise ImportError(
            "Please install graph-tool to use network_dismantling. You can install it as the instructions in https://graph-tool.skewed.de/installation.html"
        ) from exc


def require_pytorch():
    """
    Check PyTorch is installed
    """
    try:
        import torch

        _ = torch
    except ImportError as exc:
        raise ImportError(
            "Please install PyTorch to use network_dismantling. You can install it as the instructions in https://pytorch.org/get-started/locally/"
        ) from exc


def require_pyg():
    """
    Check PyTorch Geometric is installed
    """
    try:
        import torch_geometric

        _ = torch_geometric
    except ImportError as exc:
        raise ImportError(
            "Please install PyTorch Geometric to use network_dismantling. You can install it as the instructions in https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html"
        ) from exc


require_gt()
require_pytorch()
require_pyg()

__all__ = [
    "DismantlingStrategy",
    "NetworkDismantler",
    "DismantlingOperator",
    "ElementSelector",
]
