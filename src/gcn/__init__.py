"""Graph convolutional networks (Kipf and Welling, 2017) in plain PyTorch."""

from .data import Graph, load, normalized_adjacency
from .models import GCN, MLP
from .train import Config, label_propagation, train

__all__ = [
    "GCN",
    "MLP",
    "Config",
    "Graph",
    "label_propagation",
    "load",
    "normalized_adjacency",
    "train",
]
