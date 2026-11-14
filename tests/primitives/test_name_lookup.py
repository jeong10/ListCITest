"""gu.getNodeByName / gu.getEdgeByName (Phase 5.2 fast paths; fusion_python test/primitives/test_name_lookup.py) vs the definition by
scanning G.nodes / G.edges."""
import random

from _rand_graph import rand_graph
from src.graph.classes.graph import Graph
from src.graph.classes.graph_defs import directedEdgeType, bidirectedEdgeType
from src.inference.utils.graph_utils import GraphUtils as gu


def cn(s):
    return s.strip().upper() if s else ''


def ref_node(name, G):
    if not name or not G or not G.nodes:
        return None
    return next((n for n in G.nodes if cn(n['name']) == cn(name)), None)


def ref_edge(from_, to_, G, type_=None):
    if not G or not G.edges:
        return None
    f, t = ref_node(from_, G), ref_node(to_, G)
    if not f or not t:
        return None
    return next((e for e in G.edges if e['from_'] == f['name'] and e['to_'] == t['name']
                 and (not type_ or e['type_'] == type_.id_)), None)


def variants(name, rng):
    return rng.choice([name, name.lower(), ' ' + name + ' ', name.upper(), 'zz' + name, ''])


def check(G, rng, n=40):
    names = [x['name'] for x in G.nodes] + ['nope']
    for _ in range(n):
        a, b = variants(rng.choice(names), rng), variants(rng.choice(names), rng)
        t = rng.choice([None, directedEdgeType, bidirectedEdgeType])
        assert gu.getNodeByName(a, G) == ref_node(a, G)
        assert gu.getEdgeByName(a, b, G, t) == ref_edge(a, b, G, t)


def test_random_graphs():
    rng = random.Random(3)
    for _ in range(60):
        check(rand_graph(rng, rng.randint(3, 10)), rng)


def test_subgraph_views_and_name_collisions():
    rng = random.Random(4)
    names = ['x', 'Y', 'm', 'M', ' M']
    G = Graph(nodes=[{'name': n, 'label': n, 'type_': 'basic'} for n in names],
              edges=[{'from_': 'M', 'to_': 'Y', 'type_': 'directed'}, {'from_': ' M', 'to_': 'Y', 'type_': 'directed'},
                     {'from_': 'x', 'to_': 'm', 'type_': 'directed'}, {'from_': 'x', 'to_': 'Y', 'type_': 'bidirected'}])
    check(G, rng, 200)
    for _ in range(20):
        H = gu.subgraph(G, rng.sample(G.nodes, rng.randint(1, len(G.nodes))))
        check(H, rng, 40)
    check(Graph(), rng, 5)
