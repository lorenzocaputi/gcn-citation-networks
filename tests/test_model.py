import torch

from gcn import GCN, normalized_adjacency
from gcn.data import Graph
from gcn.models import sparse_dropout
from gcn.train import label_propagation


def toy_graph() -> Graph:
    # Two triangles joined by one edge; nodes 0-2 are class 0, nodes 3-5 class 1.
    pairs = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
    src = [a for a, b in pairs] + [b for a, b in pairs]
    dst = [b for a, b in pairs] + [a for a, b in pairs]
    return Graph(
        name="toy",
        features=torch.eye(6),
        labels=torch.tensor([0, 0, 0, 1, 1, 1]),
        edges=torch.tensor([src, dst]),
        train=torch.tensor([0, 5]),
        val=torch.tensor([1, 4]),
        test=torch.tensor([2, 3]),
    )


def test_normalized_adjacency_matches_dense_formula():
    g = toy_graph()
    a = torch.zeros(6, 6)
    a[g.edges[0], g.edges[1]] = 1
    a_tilde = a + torch.eye(6)
    d = a_tilde.sum(dim=1).pow(-0.5)
    expected = d[:, None] * a_tilde * d[None, :]
    assert torch.allclose(normalized_adjacency(g.edges, 6).to_dense(), expected)


def test_sparse_and_dense_inputs_agree():
    torch.manual_seed(0)
    g = toy_graph()
    adj = normalized_adjacency(g.edges, 6)
    model = GCN(6, 4, 2).eval()
    x = torch.rand(6, 6) * (torch.rand(6, 6) > 0.5)
    assert torch.allclose(model(x, adj), model(x.to_sparse(), adj), atol=1e-6)


def test_sparse_dropout_keeps_zeros_and_rescales():
    torch.manual_seed(0)
    x = torch.zeros(200, 50)
    x[::2, ::3] = 1.0
    out = sparse_dropout(x.to_sparse(), 0.5, training=True).to_dense()
    assert (out[x == 0] == 0).all()
    kept = out[x == 1]
    assert set(kept.unique().tolist()) <= {0.0, 2.0}
    assert 0.4 < (kept > 0).float().mean() < 0.6


def test_label_propagation_on_two_triangles():
    result = label_propagation(toy_graph())
    assert result.test_acc == 1.0
