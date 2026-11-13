"""Phase 5 fast paths of the shared primitives (fusion_python test/projection, test/primitives): projectOver / unproject
on the directed store, core-mask moralize, PathSearch caches, each against the legacy behaviour."""
import random

import pytest

from _rand_graph import rand_graph
from src.graph.classes.graph import Graph
from src.graph.classes.graph_defs import undirectedEdgeType
from src.inference.utils.graph_utils import GraphUtils as gu
from src.path_analysis.classes.path_search import PathSearch
from src.path_analysis.d_separation import DSeparation as DS
from src.path_analysis.utils.path_utils import PathUtils
from src.projection.projection_utils import ProjectionUtils as PU


def rand_latent_graph(rng, n):
    names = ['V%d' % i for i in range(n)]
    lat = set(rng.sample(names, rng.randint(0, max(0, n - 2))))
    nodes = []
    for v in names:
        t = 'latent' if v in lat else ('basic' if rng.random() > 0.08 else 'selection')
        nodes.append({'name': v, 'label': v, 'type_': t, 'metadata': {'k': v}})
    order = names[:]
    rng.shuffle(order)
    p = rng.choice([0.1, 0.25, 0.5])
    edges, seen = [], set()
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append({'from_': order[i], 'to_': order[j], 'type_': 'directed', 'metadata': {'e': i * n + j}})
                seen.add(frozenset((order[i], order[j])))
    for _ in range(rng.randint(0, max(1, n // 2))):
        a, b = rng.sample(names, 2)
        if frozenset((a, b)) not in seen:
            seen.add(frozenset((a, b)))
            edges.append({'from_': a, 'to_': b, 'type_': 'bidirected'})
    rng.shuffle(nodes)
    return Graph(nodes=nodes, edges=edges)


def dump(G):
    return None if G is None else (G.nodes, G.edges)


def call(fn):
    try:
        return fn()
    except Exception as e:  # noqa
        return ('exc', type(e).__name__, str(e))


def without(cls, attr, fn):
    saved = getattr(cls, attr)
    setattr(cls, attr, staticmethod(lambda G: None))
    try:
        return call(fn)
    finally:
        setattr(cls, attr, saved)


@pytest.mark.parametrize('seed', range(8))
def test_project_over_fast_equals_legacy(seed):
    rng = random.Random(seed)
    for _ in range(50):
        G = rand_latent_graph(rng, rng.randint(3, 14))
        before = dump(G)
        f = lambda: dump(PU.projectOverNonLatentNodes(G))
        assert call(f) == without(PU, '_projectOverFast', f)
        assert dump(G) == before


@pytest.mark.parametrize('seed', range(6))
def test_unproject_fast_equals_legacy(seed):
    rng = random.Random(900 + seed)
    for _ in range(50):
        G = rand_latent_graph(rng, rng.randint(2, 14))
        f = lambda: dump(PU.unproject(G))
        assert call(f) == without(PU, '_unprojectFast', f)


def legacy_moralize(G):
    graph = G.copy()
    graph.toUndirected()
    edgesToAdd = []
    for node in G.nodes:
        names = [n['name'] for n in gu.parents(node, G)]
        for (a, b) in [(a, b) for a in names for b in names if b > a]:
            if gu.hasEdge(a, b, graph) or gu.hasEdge(b, a, graph):
                continue
            edgesToAdd.append({'from_': a, 'to_': b, 'type_': undirectedEdgeType.id_})
    graph.addEdges(edgesToAdd)
    return graph


@pytest.mark.parametrize('seed', range(6))
def test_moralize_equals_legacy(seed):
    rng = random.Random(seed)
    for _ in range(40):
        G = rand_graph(rng, rng.randint(2, 14)) if rng.random() < 0.5 else rand_latent_graph(rng, rng.randint(2, 14))
        a, b = gu.moralize(G), legacy_moralize(G)
        assert (a.nodes, a.edges, [list(a.nx.adj[n]) for n in a.nx.nodes]) == \
               (b.nodes, b.edges, [list(b.nx.adj[n]) for n in b.nx.nodes])


def enc(paths):
    if not isinstance(paths, list):
        return paths
    return [[(e.edge['from_'], e.edge['to_'], e.edge['type_'], e.direction) for e in p.edges] for p in paths]


@pytest.mark.parametrize('seed', range(5))
def test_path_search_caches(seed):
    rng = random.Random(seed)
    for _ in range(20):
        G = rand_graph(rng, rng.randint(4, 9))
        pick = rng.sample(G.nodes, 3)
        X, Y, Z = pick[:1], pick[1:2], (pick[2:] if rng.random() < 0.6 else [])
        for fn in (lambda: enc(DS.findDConnectedPaths(G, X, Y, Z)),
                   lambda: enc(DS.findDSeparatedPaths(G, X, Y, Z)),
                   lambda: enc(PathUtils.findDirectedPaths(G, X, Y)),
                   lambda: enc(PathUtils.findConfoundingPaths(G, X, Y, Z))):
            PathSearch.useCaches = True
            a = call(fn)
            PathSearch.useCaches = False
            try:
                b = call(fn)
            finally:
                PathSearch.useCaches = True
            assert a == b
