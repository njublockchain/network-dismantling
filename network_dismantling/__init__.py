def require_gt():
    '''
    Check graph-tool is installed
    '''
    try:
        import graph_tool.all as gt
    except ImportError:
        raise ImportError("Please install graph-tool to use network_dismantling. You can install it as the instructions in https://graph-tool.skewed.de/installation.html")

def require_pytorch():
    '''
    Check PyTorch is installed
    '''
    try:
        import torch
    except ImportError:
        raise ImportError("Please install PyTorch to use network_dismantling. You can install it as the instructions in https://pytorch.org/get-started/locally/")

def require_pyg():
    '''
    Check PyTorch Geometric is installed
    '''
    try:
        import torch_geometric
    except ImportError:
        raise ImportError("Please install PyTorch Geometric to use network_dismantling. You can install it as the instructions in https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html")

require_gt()
require_pytorch()
require_pyg()

from network_dismantling.dismantler import DismantlingStrategy, NetworkDismantler
from network_dismantling.operators.operator import DismantlingOperator
from network_dismantling.selector import ElementSelector

__all__ = [
    "DismantlingStrategy",
    "NetworkDismantler",
    "EvaluationMetric",
    "EvaluationStrategy",
    "DismantlingOperator",
    "ElementSelector",
]
