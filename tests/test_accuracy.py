import statistics

from gcn import load, normalized_adjacency, train


def test_cora_matches_paper():
    """Kipf and Welling report 81.5 on Cora; ten seeds must average at least 80.5."""
    graph = load("cora")
    adj = normalized_adjacency(graph.edges, graph.num_nodes)
    accs = [100 * train(graph, "gcn", seed, adj=adj)[1].test_acc for seed in range(10)]
    assert statistics.mean(accs) >= 80.5, accs
