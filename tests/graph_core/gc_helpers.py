"""Shared helpers for the graph_core tests (written from the spec, not from the implementation)."""
import json
import os
import random

import networkx as nx

from src.graph.classes.graph import Graph

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
GRAPH_DIR = os.path.join(ROOT, 'test', 'graph')
BASELINE = '/Users/chris/Development/fusiontemp/fusion_python-baseline'

NODE_TYPES = ['basic', 'latent', 'sb', 'tr', 'int', 'cluster']


class Spec(object):
    """Plain description of a random graph (names, node types, ordered edge triples)."""

    def __init__(self, names, types, edges, labels=None):
        self.names = list(names)
        self.types = dict(types)
        self.edges = list(edges)          # (from, to, type) in insertion order
        self.labels = labels or {}

    def node_dicts(self):
        out = []
        for n in self.names:
            out.append({'name': n, 'label': self.labels.get(n, n), 'type_': self.types[n],
                        'metadata': {'m': n}})
        return out

    def edge_dicts(self):
        return [{'from_': u, 'to_': v, 'type_': t} for (u, v, t) in self.edges]

    def build(self):
        return Graph(nodes=self.node_dicts(), edges=self.edge_dicts())

    def nx_ref(self):
        D = nx.DiGraph()
        for n in self.names:
            D.add_node(n)
        for (u, v, t) in self.edges:
            D.add_edge(u, v, type_=t)
        return D

    def directed(self):
        return [(u, v) for (u, v, t) in self.edges if t == 'directed']

    def bidirected(self):
        return [(u, v) for (u, v, t) in self.edges if t == 'bidirected']


def random_spec(seed, max_n=40):
    rng = random.Random(seed)
    n = 1 + (seed * 7 + rng.randint(0, 5)) % max_n
    if seed % 40 == 0:
        n = 1
    names = ['V%d' % i for i in range(n)]
    rng.shuffle(names)                     # insertion order unrelated to naming
    topo = list(names)
    rng.shuffle(topo)                      # hidden causal order unrelated to insertion order
    pos = dict((x, i) for i, x in enumerate(topo))
    types = {}
    for x in names:
        r = rng.random()
        types[x] = 'latent' if r < 0.15 else ('sb' if r < 0.2 else 'basic')
    p_dir = rng.choice([0.05, 0.1, 0.25, 0.5])
    p_bi = rng.choice([0.0, 0.05, 0.1, 0.25])
    pairs = [(a, b) for i, a in enumerate(topo) for b in topo[i + 1:]]
    rng.shuffle(pairs)
    edges = []
    for (a, b) in pairs:                   # a precedes b in the hidden order
        r = rng.random()
        if r < p_dir:
            edges.append((a, b, 'directed'))
        elif r < p_dir + p_bi:
            x, y = (a, b) if rng.random() < 0.5 else (b, a)
            edges.append((x, y, 'bidirected'))
    # insertion order of the edges is random but still a permutation
    rng.shuffle(edges)
    # drop a few isolated nodes' worth of structure on purpose by leaving some nodes alone (natural)
    labels = dict((x, 'L' + x) for x in names if rng.random() < 0.5)
    return Spec(names, types, edges, labels)


def names_of(nodes):
    return [x['name'] for x in nodes]


def etuples(edges):
    return [(e['from_'], e['to_'], e['type_']) for e in edges]


def nxtuples(D):
    return [(u, v, d.get('type_', 'directed')) for (u, v, d) in D.edges(data=True)]


def in_order(order, subset):
    s = set(subset)
    return [x for x in order if x in s]


def load_json_graph(fname):
    with open(os.path.join(GRAPH_DIR, fname)) as f:
        return json.load(f)


def legacy_from_web(data):
    """Same conversion as test/script/compute/test_do_calc_engine.py"""
    nodes = []
    for nd in data['nodes']:
        nodes.append({'name': nd['name'], 'label': nd['label'], 'type_': nd['type']})
    edges = []
    for ed in data['edges']:
        edges.append({'from_': ed['from'], 'to_': ed['to'], 'type_': ed['type']})
    return Graph(nodes=nodes, edges=edges)


def val(x):
    """names.nodes / names.edges may be properties or methods."""
    return x() if callable(x) else x


def snapshot(G):
    """Full legacy snapshot of a graph (nodes, edges incl. label/type/metadata)."""
    nodes = [(n['name'], n['label'], n['type_'], n['metadata']) for n in G.nodes]
    edges = [(e['from_'], e['to_'], e['label'], e['type_'], e['metadata']) for e in G.edges]
    return nodes, edges


# ---------- brute force reference rule (baseline semantics) ----------

def brute_subgraph(spec, keep):
    keep = set(keep)
    names = [x for x in spec.names if x in keep]
    edges = [e for e in spec.edges if e[0] in keep and e[1] in keep]
    return Spec(names, dict((x, spec.types[x]) for x in names), edges,
                dict((x, l) for x, l in spec.labels.items() if x in keep))


def brute_transform(spec, over=(), under=(), over_except=(), under_except=()):
    over, under = set(over), set(under)
    oe, ue = set(over_except), set(under_except)
    edges = []
    for (u, v, t) in spec.edges:
        if v in over and u not in oe:
            continue
        if (t == 'directed' and u in under and v not in ue) or (t == 'bidirected' and u in over):
            continue
        edges.append((u, v, t))
    return Spec(spec.names, spec.types, edges, spec.labels)


def brute_do(spec, over=(), under=()):
    return brute_transform(spec, over, under)


def brute_ancestral(spec, x):
    anc = set(x)
    changed = True
    dm = spec.directed()
    while changed:
        changed = False
        for (u, v) in dm:
            if v in anc and u not in anc:
                anc.add(u)
                changed = True
    return brute_subgraph(spec, anc)


class Brute(object):
    """Brute-force queries on a Spec, results in graph order."""

    def __init__(self, spec):
        self.s = spec

    def order(self, subset):
        return in_order(self.s.names, subset)

    def parents(self, xs):
        xs = set(xs)
        return self.order(u for (u, v) in self.s.directed() if v in xs)

    def children(self, xs):
        xs = set(xs)
        return self.order(v for (u, v) in self.s.directed() if u in xs)

    def spouses(self, xs):
        xs = set(xs)
        r = set()
        for (u, v) in self.s.bidirected():
            if u in xs:
                r.add(v)
            if v in xs:
                r.add(u)
        return self.order(r - xs)

    def neighbors(self, xs, t=None):
        xs = set(xs)
        r = set()
        for (u, v, ty) in self.s.edges:
            if t is not None and ty != t:
                continue
            if u in xs:
                r.add(v)
            if v in xs:
                r.add(u)
        return self.order(r - xs)

    def reach(self, xs, t=None, direction='both'):
        seen = set(xs)
        stack = list(xs)
        es = [(u, v) for (u, v, ty) in self.s.edges if t is None or ty == t]
        while stack:
            n = stack.pop()
            for (u, v) in es:
                if direction in ('both', 'forward') and u == n and v not in seen:
                    seen.add(v)
                    stack.append(v)
                if direction in ('both', 'backward') and v == n and u not in seen:
                    seen.add(u)
                    stack.append(u)
        return self.order(seen)

    def anc(self, xs, incl=False):
        r = set(self.reach(xs, 'directed', 'backward'))
        return self.order(r if incl else r - set(xs))

    def desc(self, xs, incl=False):
        r = set(self.reach(xs, 'directed', 'forward'))
        return self.order(r if incl else r - set(xs))

    def c_components(self):
        B = nx.Graph()
        B.add_nodes_from(self.s.names)
        B.add_edges_from(self.s.bidirected())
        comps = [self.order(c) for c in nx.connected_components(B)]
        comps.sort(key=lambda c: self.s.names.index(c[0]))
        return comps

    def moral_pairs(self):
        pairs = set()
        for (u, v, t) in self.s.edges:
            pairs.add(frozenset((u, v)))
        for n in self.s.names:
            pa = self.parents([n])
            for i, a in enumerate(pa):
                for b in pa[i + 1:]:
                    pairs.add(frozenset((a, b)))
        return pairs


def is_linear_extension(order, spec):
    pos = dict((x, i) for i, x in enumerate(order))
    if sorted(order) != sorted(spec.names) or len(order) != len(spec.names):
        return False
    return all(pos[u] < pos[v] for (u, v) in spec.directed())
