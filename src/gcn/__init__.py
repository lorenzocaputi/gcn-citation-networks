"""Graph convolutional networks (Kipf and Welling, 2017) in plain PyTorch."""

from .data import Graph, load, normalized_adjacency

__all__ = ["Graph", "load", "normalized_adjacency"]
