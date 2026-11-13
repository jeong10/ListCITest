"""Core-backed gu.getIncoming / getOutgoing and the undirected Graph.neighbors / gu.subgraph fast paths
versus the legacy definitions (scan of all edges / rebuild), see fusion_python test/primitives."""
import random

import pytest

from _rand_graph import rand_graph
from src.inference.utils.graph_utils import GraphUtils as gu
from src.inference.utils.set_utils import SetUtils as su


def scan_in(w, G):
    names = set(n['name'] for n in w)
    return [e for e in G.edges if (e['type_'] == 'bidirected' and e['from_'] in names) or e['to_'] in names]


def scan_out(w, G):
    names = set(n['name'] for n in w)
    return [e for e in G.edges if e['type_'] != 'bidirected' and e['from_'] in names]


def graphs(rng):
    G = rand_graph(rng, rng.randint(3, 10))
    yield G
    nodes = G.nodes
    yield gu.subgraph(G, rng.sample(nodes, rng.randint(1, len(nodes))))
    yield gu.transform(G, rng.sample(nodes, 1), rng.sample(nodes, 1))      # do-cuts: fallback path
    H = G.copy()
    H.deleteNodes(rng.sample(H.nodes, 1))                                  # mutated store
    yield H
    yield gu.subgraph(gu.transform(G, None, rng.sample(nodes, 1)), nodes[:-1])


@pytest.mark.parametrize('seed', range(8))
def test_incoming_outgoing(seed):
    rng = random.Random(seed)
    for _ in range(40):
        for G in graphs(rng):
            nodes = G.nodes
            for _ in range(6):
                w = rng.sample(nodes, rng.randint(0, min(3, len(nodes))))
                if rng.random() < 0.15:
                    w = w + [{'name': 'nope', 'label': 'nope', 'type_': 'basic'}]
                assert gu.getIncoming(w, G) == scan_in(w, G)
                assert gu.getOutgoing(w, G) == scan_out(w, G)


def legacy_neighbors(G, nodes, edgeType=None):
    names = dict((n['name'], True) for n in nodes)
    nb = {}
    for e in [e for e in G.edges if (e['type_'] == edgeType if edgeType else True)]:
        if e['to_'] in names:
            nb[e['from_']] = True
        if e['from_'] in names:
            nb[e['to_']] = True
    return [n for n in G.nodes if n['name'] in nb and n['name'] not in names]


def legacy_subgraph(G, subset):
    graph = G.copy()
    graph.deleteNodes(graph.nodes)
    graph.addNodes(su.intersection(G.nodes, subset, 'name'))
    sub = dict((n['name'], True) for n in subset)
    edges = [e for e in G.edges if e['from_'] in sub and e['to_'] in sub]
    graph.deleteEdges(graph.edges)
    graph.addEdges(edges)
    return graph


@pytest.mark.parametrize('seed', range(8))
def test_undirected(seed):
    rng = random.Random(seed)
    for _ in range(40):
        D = rand_graph(rng, rng.randint(3, 10))
        M = gu.moralize(D)
        assert M._ug is not None
        nodes = M.nodes
        for _ in range(6):
            w = rng.sample(nodes, rng.randint(1, min(3, len(nodes))))
            if rng.random() < 0.2:
                w = w + [{'name': 'nope', 'label': 'nope', 'type_': 'basic'}]
            for et in (None, 'directed', 'bidirected', 'undirected'):
                assert M.neighbors(w, et) == legacy_neighbors(M, w, et)
            subset = rng.sample(nodes, rng.randint(0, len(nodes)))
            a = gu.subgraph(M, subset)
            b = legacy_subgraph(M, subset)
            assert (a.nodes, a.edges) == (b.nodes, b.edges)
            if a.nodes:
                v = rng.sample(a.nodes, 1)
                assert a.neighbors(v) == b.neighbors(v) == legacy_neighbors(b, v)
                a.deleteNodes(v)
                b.deleteNodes(v)
                assert (a.nodes, a.edges) == (b.nodes, b.edges)
