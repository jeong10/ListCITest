import glob
import json
import os
import random

import networkx as nx

from src.graph.classes.graph import Graph

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
GRAPH_DIR = os.path.join(ROOT, 'test', 'graph')
BASELINE = os.path.abspath(os.path.join(ROOT, '..', 'fusion_python-baseline'))
RUNNER = os.path.join(HERE, '_runner.py')


class Spec(object):
    def __init__(self, names, edges, types=None):
        self.names = list(names)
        self.edges = list(edges)      # (u, v, 'directed'|'bidirected')
        self.types = dict(types or {})

    def build(self):
        return Graph(nodes=[{'name': n, 'type_': self.types.get(n, 'basic')} for n in self.names],
                     edges=[{'from_': u, 'to_': v, 'type_': t} for (u, v, t) in self.edges])

    def build_json(self):
        return Graph.from_json({
            'nodes': [{'name': n, 'type': self.types.get(n, 'basic')} for n in self.names],
            'edges': [{'from': u, 'to': v, 'type': t} for (u, v, t) in self.edges]})

    def as_job(self):
        return {'names': self.names, 'edges': self.edges, 'types': self.types}

    def has_bi(self):
        return any(t == 'bidirected' for (_, _, t) in self.edges)

    def without_bi(self):
        return Spec(self.names, [e for e in self.edges if e[2] == 'directed'], self.types)

    def latent_dag(self):
        """Explicit DAG: every a<->b replaced by fresh U with U->a, U->b."""
        D = nx.DiGraph()
        D.add_nodes_from(self.names)
        for i, (u, v, t) in enumerate(self.edges):
            if t == 'directed':
                D.add_edge(u, v)
            else:
                lat = '__U%d' % i
                D.add_edge(lat, u)
                D.add_edge(lat, v)
        return D

    def latent_spec(self):
        """Same thing as a Spec whose latent nodes are real 'latent'-typed nodes."""
        names = list(self.names)
        types = dict(self.types)
        edges = []
        for i, (u, v, t) in enumerate(self.edges):
            if t == 'directed':
                edges.append((u, v, t))
            else:
                lat = '__U%d' % i
                names.append(lat)
                types[lat] = 'latent'
                edges.append((lat, u, 'directed'))
                edges.append((lat, v, 'directed'))
        return Spec(names, edges, types)

    # view semantics
    def do(self, over, under):
        over, under = set(over), set(under)
        edges = []
        for (u, v, t) in self.edges:
            if t == 'directed':
                if v in over or u in under:
                    continue
            else:
                if u in over or v in over:
                    continue
            edges.append((u, v, t))
        return Spec(self.names, edges, self.types)

    def sub(self, keep):
        keep = set(keep)
        return Spec([n for n in self.names if n in keep],
                    [e for e in self.edges if e[0] in keep and e[1] in keep], self.types)


def random_admg(rng, nmin=2, nmax=10):
    n = rng.randint(nmin, nmax)
    names = ['V%d' % i for i in range(n)]
    rng.shuffle(names)
    order = list(names)
    rng.shuffle(order)
    p_dir = rng.choice([0.0, 0.1, 0.25, 0.5])
    p_bi = rng.choice([0.0, 0.1, 0.25, 0.4])
    edges = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            r = rng.random()
            if r < p_dir:
                edges.append((a, b, 'directed'))
            elif r < p_dir + p_bi:
                edges.append((a, b, 'bidirected') if rng.random() < .5 else (b, a, 'bidirected'))
    rng.shuffle(edges)
    return Spec(names, edges)


def random_triple(rng, names, zmax=None):
    """disjoint nonempty X, Y and possibly empty Z (lists of names)."""
    n = len(names)
    k = rng.randint(2, n) if n >= 2 else n
    pick = rng.sample(names, k)
    nx_ = rng.randint(1, max(1, min(2, k - 1)))
    ny = rng.randint(1, max(1, min(2, k - nx_)))
    X = pick[:nx_]
    Y = pick[nx_:nx_ + ny]
    Z = pick[nx_ + ny:]
    if rng.random() < 0.3:
        Z = []
    return X, Y, Z


def nx_dsep(D, X, Y, Z):
    f = getattr(nx, 'is_d_separator', None)
    if f is None:
        f = nx.d_separated
    return bool(f(D, set(X), set(Y), set(Z)))


def oracle_m(spec, X, Y, Z):
    return nx_dsep(spec.latent_dag(), X, Y, Z)


def dicts(names):
    return [{'name': n} for n in names]


def corpus_specs():
    out = []
    for p in sorted(glob.glob(os.path.join(GRAPH_DIR, '*.txt'))):
        with open(p) as f:
            data = json.load(f)
        names = [n['name'] for n in data['nodes']]
        types = dict((n['name'], n['type']) for n in data['nodes'])
        edges = [(e['from'], e['to'], e['type']) for e in data['edges']]
        out.append((os.path.basename(p), Spec(names, edges, types), data))
    return out
