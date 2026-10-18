"""The two-layer GCN of Kipf and Welling (2017, eq. 9) and an MLP with the same shape."""

from __future__ import annotations

import torch
from torch import nn


def sparse_dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Dropout on the stored entries of a sparse tensor (zeros stay zero)."""
    if not training or p == 0:
        return x
    x = x.coalesce()
    keep = torch.rand(x.values().shape) >= p
    values = x.values()[keep] / (1 - p)
    return torch.sparse_coo_tensor(x.indices()[:, keep], values, x.shape, check_invariants=False)


class GraphConvolution(nn.Module):
    """H' = A_hat H W, with W initialised as in Glorot and Bengio (2010) and no bias."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None) -> torch.Tensor:
        support = torch.sparse.mm(x, self.weight) if x.is_sparse else x @ self.weight
        return support if adj is None else torch.sparse.mm(adj, support)


class GCN(nn.Module):
    """Z = A_hat ReLU(A_hat X W0) W1, with dropout before each layer.

    X may be sparse; dropout then acts on its non-zero entries, as in the
    reference implementation.
    """

    def __init__(self, in_features: int, hidden: int, classes: int, dropout: float = 0.5):
        super().__init__()
        self.layer1 = GraphConvolution(in_features, hidden)
        self.layer2 = GraphConvolution(hidden, classes)
        self.p = dropout
        self.dropout = nn.Dropout(dropout)

    def hidden(self, x: torch.Tensor, adj: torch.Tensor | None) -> torch.Tensor:
        x = sparse_dropout(x, self.p, self.training) if x.is_sparse else self.dropout(x)
        return torch.relu(self.layer1(x, adj))

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None) -> torch.Tensor:
        return self.layer2(self.dropout(self.hidden(x, adj)), adj)

    def regularized(self) -> list[nn.Parameter]:
        """Parameters that get L2 regularisation (only the first layer, as in the paper)."""
        return [self.layer1.weight]


class MLP(GCN):
    """The same network with the adjacency matrix replaced by the identity."""

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        return super().forward(x, None)

    def hidden(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        return super().hidden(x, None)
