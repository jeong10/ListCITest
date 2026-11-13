"""Property tests on seeded random graphs: legacy API, gu helpers and names API vs brute force / networkx."""
import random

import networkx as nx
import pytest

from src.graph.classes.graph import Graph
from src.graph.classes.graph_defs import (directedEdgeType, bidirectedEdgeType, undirectedEdgeType)
from src.inference.utils.graph_utils import GraphUtils as gu, Direction

from gc_helpers import (random_spec, Spec, Brute, names_of, etuples, nxtuples, in_order, snapshot,
                        brute_subgraph, brute_transform, is_linear_extension, val)

SEEDS = list(range(60))
seeds = pytest.mark.parametrize('seed', SEEDS)

ETYPES = {'directed': directedEdgeType, 'bidirected': bidirectedEdgeType}


def pick(rng, names, kmax=3, allow_empty=False):
    lo = 0 if allow_empty else 1
    k = rng.randint(lo, min(kmax, len(names)))
    return rng.sample(names, k)


def dicts(G, names):
    d = dict((n['name'], n) for n in G.nodes)
    return [d[x] for x in names]


def queries(spec, seed, count=4):
    rng = random.Random(seed * 31 + 7)
    return [pick(rng, spec.names, 4) for _ in range(count)]


# ------------------------------------------------------------------ construction / order contract

@seeds
def test_construction_order_matches_networkx(seed):
    spec = random_spec(seed)
    G = spec.build()
    D = spec.nx_ref()
    assert names_of(G.nodes) == list(D.nodes)
    assert etuples(G.edges) == nxtuples(D)
    # node dicts carry label/type_/metadata
    for n, nd in zip(spec.names, G.nodes):
        assert nd['label'] == spec.labels.get(n, n)
        assert nd['type_'] == spec.types[n]
        assert nd['metadata'] == {'m': n}
    for e in G.edges:
        assert set(e) == {'from_', 'to_', 'label', 'type_', 'metadata'}
        assert e['label'] is None and e['metadata'] == {}


@seeds
def test_fresh_lists_shared_metadata(seed):
    G = random_spec(seed).build()
    a, b = G.nodes, G.nodes
    assert a is not b and a == b
    assert all(x is not y for x, y in zip(a, b))
    assert all(x['metadata'] is y['metadata'] for x, y in zip(a, b))
    ea, eb = G.edges, G.edges
    assert ea is not eb
    assert all(x['metadata'] is y['metadata'] for x, y in zip(ea, eb))
    if a:
        a[0]['metadata']['added'] = 1
        assert G.nodes[0]['metadata']['added'] == 1
        a[0]['label'] = 'zzz'            # outer dict is a fresh record
        assert G.nodes[0]['label'] != 'zzz'


def test_node_defaults():
    G = Graph(nodes=[{'name': 'A'}, {'name': 'B', 'label': 'bee', 'type_': 'latent', 'metadata': {'k': 1}}])
    a, b = G.nodes
    assert a == {'name': 'A', 'label': 'A', 'type_': 'basic', 'metadata': {}}
    assert b == {'name': 'B', 'label': 'bee', 'type_': 'latent', 'metadata': {'k': 1}}
    G.addEdges({'from_': 'A', 'to_': 'B'})
    assert G.edges == [{'from_': 'A', 'to_': 'B', 'label': None, 'type_': 'directed', 'metadata': {}}]


@seeds
def test_copy_order_and_metadata_sharing(seed):
    G = random_spec(seed).build()
    C = G.copy()
    assert names_of(C.nodes) == names_of(G.nodes)
    assert etuples(C.edges) == etuples(G.edges)
    for a, b in zip(G.nodes, C.nodes):
        assert a['metadata'] is b['metadata']
    for a, b in zip(G.edges, C.edges):
        assert a['metadata'] is b['metadata']
    # structural independence
    if G.nodes:
        C.deleteNodes(dicts(C, [C.nodes[0]['name']]))
        assert len(G.nodes) == len(C.nodes) + 1


@seeds
def test_random_mutations_keep_networkx_order(seed):
    rng = random.Random(seed + 1000)
    spec = random_spec(seed, max_n=25)
    G = spec.build()
    D = spec.nx_ref()
    fresh = [0]

    def check():
        assert names_of(G.nodes) == list(D.nodes)
        assert etuples(G.edges) == nxtuples(D)

    check()
    for step in range(25):
        op = rng.choice(['delnode', 'deledge', 'addedge', 'addnode', 'copy', 'addedge_unknown'])
        names = list(D.nodes)
        if op == 'delnode' and names:
            xs = pick(rng, names, 3)
            G.deleteNodes(dicts(G, xs))
            D.remove_nodes_from(xs)
        elif op == 'deledge' and D.number_of_edges():
            es = rng.sample(list(D.edges), min(len(D.edges), rng.randint(1, 3)))
            by = dict(((e['from_'], e['to_']), e) for e in G.edges)
            G.deleteEdges([by[e] for e in es])
            D.remove_edges_from(es)
        elif op == 'addedge' and len(names) >= 2:
            u, v = rng.sample(names, 2)
            if D.has_edge(u, v) or D.has_edge(v, u):
                continue
            t = rng.choice(['directed', 'bidirected'])
            G.addEdges({'from_': u, 'to_': v, 'type_': t})
            D.add_edge(u, v, type_=t)
        elif op == 'addedge_unknown' and names:
            u = rng.choice(names)
            fresh[0] += 1
            v = 'NEW%d' % fresh[0]
            G.addEdges({'from_': u, 'to_': v, 'type_': 'directed'})   # silently creates unknown node
            D.add_edge(u, v, type_='directed')
        elif op == 'addnode':
            fresh[0] += 1
            nm = 'N%d' % fresh[0]
            G.addNodes({'name': nm})
            D.add_node(nm)
        elif op == 'copy':
            G = G.copy()
        check()
    # every non-mutated node dict still has full attributes
    for n in G.nodes:
        assert set(n) == {'name', 'label', 'type_', 'metadata'}


@seeds
def test_unknown_nodes_created_at_end(seed):
    spec = random_spec(seed, max_n=10)
    G = spec.build()
    G.addEdges([{'from_': 'QQ', 'to_': 'RR', 'type_': 'directed'}])
    assert names_of(G.nodes) == spec.names + ['QQ', 'RR']
    n = G.nodes[-1]
    assert n['label'] == 'RR' and n['type_'] == 'basic' and n['metadata'] == {}


def test_bidirected_flip_rules():
    ND = [{'name': x} for x in 'XYZ']
    # X->Y exists, add X<->Y  => stored as (Y,X) bidirected
    G = Graph(nodes=ND, edges=[{'from_': 'X', 'to_': 'Y'}])
    G.addEdges({'from_': 'X', 'to_': 'Y', 'type_': 'bidirected'})
    assert etuples(G.edges) == [('X', 'Y', 'directed'), ('Y', 'X', 'bidirected')]
    # X<->Y exists, add X->Y  => existing flipped to (Y,X), directed X->Y added
    G = Graph(nodes=[{'name': x} for x in 'XYZ'], edges=[{'from_': 'X', 'to_': 'Y', 'type_': 'bidirected'}])
    G.addEdges({'from_': 'X', 'to_': 'Y'})
    D = nx.DiGraph()
    D.add_nodes_from('XYZ')
    D.add_edge('Y', 'X', type_='bidirected')
    D.add_edge('X', 'Y', type_='directed')
    assert etuples(G.edges) == nxtuples(D)
    # bidirected is ONE record
    G = Graph(nodes=[{'name': x} for x in 'XYZ'], edges=[{'from_': 'X', 'to_': 'Z', 'type_': 'bidirected'}])
    assert len(G.edges) == 1


def test_duplicate_edges_raise():
    G = Graph(nodes=[{'name': 'A'}, {'name': 'B'}], edges=[{'from_': 'A', 'to_': 'B'}])
    with pytest.raises(Exception):
        G.addEdges({'from_': 'A', 'to_': 'B'})
    G = Graph(nodes=[{'name': 'A'}, {'name': 'B'}], edges=[{'from_': 'A', 'to_': 'B', 'type_': 'bidirected'}])
    with pytest.raises(Exception):
        G.addEdges({'from_': 'A', 'to_': 'B', 'type_': 'bidirected'})
    assert len(G.edges) == 1


def test_delete_missing_edge_follows_graph_flag():
    # fusion_python: raises NetworkXError (legacy); ListCITest sets Graph.ignore_missing_edge_deletion = True
    G = Graph(nodes=[{'name': 'A'}, {'name': 'B'}], edges=[])
    if Graph.ignore_missing_edge_deletion:
        G.deleteEdges({'from_': 'A', 'to_': 'B'})
        assert G.edges == []
    else:
        with pytest.raises(nx.NetworkXError):
            G.deleteEdges({'from_': 'A', 'to_': 'B'})


def test_add_edges_does_not_check_cycles():
    G = Graph(nodes=[{'name': 'A'}, {'name': 'B'}], edges=[{'from_': 'A', 'to_': 'B'}])
    G.addEdges({'from_': 'B', 'to_': 'A'})
    assert etuples(G.edges) == [('A', 'B', 'directed'), ('B', 'A', 'directed')]


@seeds
def test_delete_nodes_removes_incident_edges(seed):
    spec = random_spec(seed)
    G = spec.build()
    xs = queries(spec, seed, 1)[0]
    G.deleteNodes(dicts(G, xs))
    xs = set(xs)
    assert names_of(G.nodes) == [x for x in spec.names if x not in xs]
    D = spec.nx_ref()
    D.remove_nodes_from(xs)
    assert etuples(G.edges) == nxtuples(D)


# ------------------------------------------------------------------ legacy node-set methods

@seeds
def test_legacy_parents_children_neighbors(seed):
    spec = random_spec(seed)
    G, B = spec.build(), Brute(spec)
    for xs in queries(spec, seed):
        nd = dicts(G, xs)
        assert names_of(G.parents(nd)) == B.parents(xs)
        assert names_of(G.children(nd)) == B.children(xs)
        assert names_of(gu.parents(nd, G)) == B.parents(xs)
        assert names_of(gu.children(nd, G)) == B.children(xs)
        assert names_of(G.neighbors(nd)) == B.neighbors(xs)
        for t in ('directed', 'bidirected'):
            assert names_of(G.neighbors(nd, t)) == B.neighbors(xs, t)
            assert names_of(G.neighbors(nd, ETYPES[t])) == B.neighbors(xs, t)
            assert names_of(gu.neighbors(nd, G, ETYPES[t])) == B.neighbors(xs, t)
        # single dict argument
        assert names_of(G.parents(nd[0])) == B.parents(xs[:1])
        assert G.parents([]) == [] and G.neighbors([]) == []


@seeds
def test_legacy_ancestors_descendants_methods(seed):
    """Legacy Graph.ancestors/descendants: directed only, graph order, EXCLUDING the query nodes."""
    spec = random_spec(seed)
    G, B = spec.build(), Brute(spec)
    for xs in queries(spec, seed):
        nd = dicts(G, xs)
        assert names_of(G.ancestors(nd)) == B.anc(xs)
        assert names_of(G.descendants(nd)) == B.desc(xs)


@seeds
def test_gu_helpers(seed):
    spec = random_spec(seed)
    G, B = spec.build(), Brute(spec)
    for xs in queries(spec, seed):
        nd = dicts(G, xs)
        # inclusive, graph order
        assert names_of(gu.ancestors(nd, G)) == B.anc(xs, True)
        assert names_of(gu.descendants(nd, G)) == B.desc(xs, True)
        assert set(names_of(gu.ancestorsPlus(nd, G))) == set(B.anc(xs, True))
        assert set(names_of(gu.descendantsPlus(nd, G))) == set(B.desc(xs, True))
        assert len(gu.ancestorsPlus(nd, G)) == len(B.anc(xs, True))
        assert set(names_of(gu.parentsPlus(nd, G))) == set(B.parents(xs)) | set(xs)
        assert names_of(gu.nonDescendants(nd, G)) == in_order(spec.names, set(spec.names) - set(B.desc(xs, True)))
        # spouses: union over single-node neighbour queries (each excludes only itself)
        exp = []
        for x in xs:
            for y in B.neighbors([x], 'bidirected'):
                if y not in exp:
                    exp.append(y)
        assert names_of(gu.spouses(nd, G)) == exp
        # single dict
        assert names_of(gu.ancestors(nd[0], G)) == B.anc(xs[:1], True)


@seeds
def test_gu_reach_all_types_and_directions(seed):
    spec = random_spec(seed)
    G, B = spec.build(), Brute(spec)
    dirs = {'both': Direction.both, 'forward': Direction.forward, 'backward': Direction.backward}
    for xs in queries(spec, seed, 3):
        nd = dicts(G, xs)
        for t in (None, 'directed', 'bidirected'):
            for dn, d in dirs.items():
                et = ETYPES.get(t)
                got = names_of(gu.reach(nd, G, et, d))
                assert got == B.reach(xs, t, dn), (t, dn)
        assert names_of(gu.reach(nd, G)) == B.reach(xs)


# ------------------------------------------------------------------ subgraph / transform

def same_graph(G, spec):
    ref = spec.build()
    sn, se = snapshot(G)
    rn, re_ = snapshot(ref)
    assert sn == rn
    assert se == re_


@seeds
def test_gu_subgraph_vs_filter(seed):
    spec = random_spec(seed)
    G = spec.build()
    before = snapshot(G)
    rng = random.Random(seed + 5)
    for _ in range(4):
        keep = pick(rng, spec.names, len(spec.names), allow_empty=True)
        S = gu.subgraph(G, dicts(G, keep))
        same_graph(S, brute_subgraph(spec, keep))
        # a subgraph is usable as an ordinary graph (queries run on the materialized form too)
        Bs = Brute(brute_subgraph(spec, keep))
        if keep:
            xs = in_order(S.nodes and names_of(S.nodes), [keep[0]])
            assert names_of(gu.ancestors(dicts(S, xs), S)) == Bs.anc(xs, True)
    assert snapshot(G) == before
    assert gu.subgraph(G, None) is G or snapshot(gu.subgraph(G, None)) == before


@seeds
def test_gu_transform_vs_filter(seed):
    spec = random_spec(seed)
    G = spec.build()
    before = snapshot(G)
    rng = random.Random(seed + 77)
    for _ in range(5):
        over = pick(rng, spec.names, 3, True)
        under = pick(rng, spec.names, 3, True)
        oe = pick(rng, spec.names, 3, True)
        ue = pick(rng, spec.names, 3, True)
        exp = brute_transform(spec, over, under, oe, ue)
        T = gu.transform(G, dicts(G, over), dicts(G, under), dicts(G, oe), dicts(G, ue))
        same_graph(T, exp)
        # no *Except
        T2 = gu.transform(G, dicts(G, over), dicts(G, under))
        same_graph(T2, brute_transform(spec, over, under))
        # None / default arguments
        same_graph(gu.transform(G, dicts(G, over), None), brute_transform(spec, over, ()))
        same_graph(gu.transform(G, None, dicts(G, under)), brute_transform(spec, (), under))
        same_graph(gu.transform(G), spec)
        # queries on the result agree with brute force on the reference graph
        Bt = Brute(exp)
        xs = pick(rng, spec.names, 2)
        assert names_of(gu.ancestors(dicts(T, xs), T)) == Bt.anc(xs, True)
        assert names_of(T.children(dicts(T, xs))) == Bt.children(xs)
        assert names_of(T.neighbors(dicts(T, xs), 'bidirected')) == Bt.neighbors(xs, 'bidirected')
    assert snapshot(G) == before


@seeds
def test_transform_then_subgraph_chain(seed):
    spec = random_spec(seed)
    G = spec.build()
    rng = random.Random(seed + 3)
    cur_spec, cur = spec, G
    for _ in range(4):
        if rng.random() < 0.5:
            over = pick(rng, cur_spec.names, 2, True)
            under = pick(rng, cur_spec.names, 2, True)
            cur = gu.transform(cur, dicts(cur, over), dicts(cur, under))
            cur_spec = brute_transform(cur_spec, over, under)
        else:
            keep = pick(rng, cur_spec.names, len(cur_spec.names), True)
            cur = gu.subgraph(cur, dicts(cur, keep))
            cur_spec = brute_subgraph(cur_spec, keep)
        same_graph(cur, cur_spec)


# ------------------------------------------------------------------ topological sort / c-components

@seeds
def test_topoSort_valid_and_deterministic(seed):
    spec = random_spec(seed)
    G = spec.build()
    for flag in (False, True):
        order = names_of(gu.topoSort(G, flag))
        assert is_linear_extension(order, spec)
        assert order == names_of(gu.topoSort(spec.build(), flag))
        assert order == names_of(gu.topoSort(G, flag))


@seeds
def test_topoSort_sorted_reference(seed):
    """sortFlag=True: legacy toposort_flatten(sort=True) = level by level, names sorted; nodes with no
    directed edge are appended in graph order."""
    spec = random_spec(seed)
    G = spec.build()
    dm = spec.directed()
    deps = {}
    for (u, v) in dm:
        deps.setdefault(v, set()).add(u)
        deps.setdefault(u, set())
    done, order = set(), []
    remaining = dict(deps)
    while remaining:
        level = sorted(k for k, d in remaining.items() if not (d - done))
        assert level
        order.extend(level)
        done.update(level)
        for k in level:
            del remaining[k]
    order += [x for x in spec.names if x not in set(order)]
    assert names_of(gu.topoSort(G, True)) == order


def test_topoSort_isolated_and_bidirected_only():
    G = Graph(nodes=[{'name': x} for x in 'ABCDE'],
              edges=[{'from_': 'B', 'to_': 'C', 'type_': 'bidirected'}, {'from_': 'E', 'to_': 'D'}])
    order = names_of(gu.topoSort(G))
    assert sorted(order) == list('ABCDE') and len(order) == 5
    assert order.index('E') < order.index('D')
    assert names_of(gu.topoSort(Graph())) == []


@seeds
def test_gu_cCompDecomposition(seed):
    spec = random_spec(seed)
    G, B = spec.build(), Brute(spec)
    for flag in (False, True):
        V = names_of(gu.topoSort(G, flag))
        got = [names_of(c) for c in gu.cCompDecomposition(G, flag)]
        # partition correctness
        flat = [x for c in got for x in c]
        assert sorted(flat) == sorted(spec.names)
        exp = B.c_components()
        assert sorted(map(tuple, got)) == sorted(map(tuple, exp))
        # each component in graph order
        for c in got:
            assert c == in_order(spec.names, c)
        # components ordered by topological position of their first discovered node
        mins = [min(V.index(x) for x in c) for c in got]
        assert mins == sorted(mins)


# ------------------------------------------------------------------ name lookups

def test_getNodeByName_case_insensitive():
    G = Graph(nodes=[{'name': 'Abc'}, {'name': 'x_1'}, {'name': 'Y'}])
    assert gu.getNodeByName('abc', G)['name'] == 'Abc'
    assert gu.getNodeByName('ABC', G)['name'] == 'Abc'
    assert gu.getNodeByName(' x_1 ', G)['name'] == 'x_1'
    assert gu.getNodeByName('X_1', G)['name'] == 'x_1'
    assert gu.getNodeByName('nope', G) is None
    assert gu.getNodeByName('', G) is None
    assert gu.getNodeByName(None, G) is None
    assert gu.getNodeByName('y', Graph()) is None


def test_getNodesByName_semantics():
    G = Graph(nodes=[{'name': 'Abc'}, {'name': 'x_1'}, {'name': 'Y'}])
    assert names_of(gu.getNodesByName(['y', 'ABC'], G)) == ['Y', 'Abc']          # follows input order
    assert names_of(gu.getNodesByName('x_1', G)) == ['x_1']
    assert names_of(gu.getNodesByName(['abc', 'missing', 'Y'], G)) == ['Abc', 'Y']  # missing skipped
    assert gu.getNodesByName([], G) == []
    assert gu.getNodesByName(None, G) == []
    assert gu.getNodesByName(['abc'], None) == []
    # duplicates in the input are reproduced (legacy loop appends once per input name)
    assert names_of(gu.getNodesByName(['y', 'Y'], G)) == ['Y', 'Y']
    # returned records are the graph's own (metadata by reference)
    G.nodes[0]['metadata']['k'] = 1
    assert gu.getNodesByName(['abc'], G)[0]['metadata'] == {'k': 1}


def test_getNodesByName_node_dict_inputs_use_their_own_name():
    """Known intentional difference #7 (legacy resolved every dict to the last node)."""
    G = Graph(nodes=[{'name': 'A'}, {'name': 'B'}, {'name': 'C'}])
    ND = G.nodes
    assert names_of(gu.getNodesByName([ND[0], ND[1]], G)) == ['A', 'B']
    assert names_of(gu.getNodesByName([ND[2], ND[0]], G)) == ['C', 'A']


def test_case_colliding_names():
    G = Graph(nodes=[{'name': 'a'}, {'name': 'A'}])
    assert gu.getNodeByName('A', G)['name'] == 'a'            # first match in graph order
    assert names_of(gu.getNodesByName(['a'], G)) == ['A']     # legacy: last node of the colliding pair wins


@seeds
def test_getEdgeByName_and_hasEdge(seed):
    spec = random_spec(seed)
    G = spec.build()
    stored = dict(((u, v), t) for (u, v, t) in spec.edges)
    for (u, v), t in list(stored.items())[:10]:
        e = gu.getEdgeByName(u.lower(), v.upper(), G)
        assert (e['from_'], e['to_'], e['type_']) == (u, v, t)
        assert gu.getEdgeByName(u, v, G, ETYPES[t]) is not None
        other = 'bidirected' if t == 'directed' else 'directed'
        assert gu.getEdgeByName(u, v, G, ETYPES[other]) is None
        assert gu.hasEdge(u, v, G)
    for u in spec.names[:6]:
        for v in spec.names[:6]:
            assert gu.hasEdge(u, v, G) == ((u, v) in stored)
    assert gu.getEdgeByName('nope', spec.names[0], G) is None


# ------------------------------------------------------------------ moralize, undirected, components

@seeds
def test_gu_moralize_edge_set(seed):
    spec = random_spec(seed)
    G, B = spec.build(), Brute(spec)
    M = gu.moralize(G)
    assert names_of(M.nodes) == spec.names
    got = set()
    for e in M.edges:
        assert e['from_'] != e['to_']
        got.add(frozenset((e['from_'], e['to_'])))
    assert got == B.moral_pairs()
    assert len(M.edges) == len(got)                   # no pair twice
    # input graph untouched
    assert etuples(G.edges) == nxtuples(spec.nx_ref())
    # edges that were added by marrying parents are undirected
    orig = set(frozenset((u, v)) for (u, v, t) in spec.edges)
    for e in M.edges:
        if frozenset((e['from_'], e['to_'])) not in orig:
            assert e['type_'] == undirectedEdgeType.id_


@seeds
def test_to_undirected_and_connected_components(seed):
    spec = random_spec(seed)
    G = spec.build()
    if spec.edges:
        with pytest.raises(nx.NetworkXNotImplemented):
            G.connectedComponents()
    U = spec.build()
    U.toUndirected()
    U2 = nx.Graph()
    U2.add_nodes_from(spec.names)
    U2.add_edges_from((u, v) for (u, v, t) in spec.edges)
    got = set(frozenset(names_of(c)) for c in U.connectedComponents())
    assert got == set(frozenset(c) for c in nx.connected_components(U2))
    assert sorted(names_of(U.nodes)) == sorted(spec.names)
    assert set(frozenset((e['from_'], e['to_'])) for e in U.edges) == \
        set(frozenset((u, v)) for (u, v, t) in spec.edges)


# ------------------------------------------------------------------ names API vs brute force / networkx

@seeds
def test_names_ancestors_descendants_vs_networkx(seed):
    spec = random_spec(seed)
    G = spec.build()
    N = G.names
    D = nx.DiGraph()
    D.add_nodes_from(spec.names)
    D.add_edges_from(spec.directed())
    for xs in queries(spec, seed):
        an, de = set(), set()
        for x in xs:
            an |= nx.ancestors(D, x)
            de |= nx.descendants(D, x)
        assert N.ancestors(xs) == in_order(spec.names, an - set(xs))
        assert N.descendants(xs) == in_order(spec.names, de - set(xs))
        assert N.ancestors(xs, include_self=True) == in_order(spec.names, an | set(xs))
        assert N.descendants(xs, include_self=True) == in_order(spec.names, de | set(xs))
        assert N.ancestors(xs[0]) == in_order(spec.names, nx.ancestors(D, xs[0]))
        assert N.descendants(xs[0], include_self=True) == \
            in_order(spec.names, nx.descendants(D, xs[0]) | {xs[0]})
        # any iterable works, results identical
        assert N.ancestors(tuple(xs)) == N.ancestors(xs)
        assert N.ancestors(set(xs)) == N.ancestors(xs)
        assert N.ancestors(x for x in xs) == N.ancestors(xs)


@seeds
def test_names_local_queries_vs_brute(seed):
    spec = random_spec(seed)
    G, B = spec.build(), Brute(spec)
    N = G.names
    for xs in queries(spec, seed):
        assert N.parents(xs) == B.parents(xs)
        assert N.children(xs) == B.children(xs)
        assert N.spouses(xs) == B.spouses(xs)
        assert N.neighbors(xs) == B.neighbors(xs)
        assert N.neighbors(xs, 'directed') == B.neighbors(xs, 'directed')
        assert N.neighbors(xs, 'bidirected') == B.neighbors(xs, 'bidirected')
        assert N.neighbors(xs, directedEdgeType) == B.neighbors(xs, 'directed')
        assert N.neighbors(xs, edge_type=bidirectedEdgeType) == B.neighbors(xs, 'bidirected')
        assert N.parents(xs[0]) == B.parents(xs[:1])
        assert N.spouses(xs[0]) == B.spouses(xs[:1])


@seeds
def test_names_c_components(seed):
    spec = random_spec(seed)
    G, B = spec.build(), Brute(spec)
    cc = G.names.c_components()
    assert cc == B.c_components()
    for c in cc:
        for x in c:
            assert G.names.c_component(x) == c


@seeds
def test_names_topological_order(seed):
    spec = random_spec(seed)
    G = spec.build()
    order = G.names.topological_order()
    assert is_linear_extension(order, spec)
    assert order == G.names.topological_order()
    assert order == spec.build().names.topological_order()


@seeds
def test_names_moralize_vs_brute(seed):
    spec = random_spec(seed)
    G, B = spec.build(), Brute(spec)
    M = G.names.moralize()
    got = M.edges if not hasattr(M, 'names') else None
    # moralize may return a Graph (undirected, no names API) or an edge collection
    if isinstance(M, Graph):
        got = [(e['from_'], e['to_']) for e in M.edges]
    else:
        got = [tuple(e)[:2] for e in M]
    assert set(frozenset(p) for p in got) == B.moral_pairs()
    assert len(got) == len(set(frozenset(p) for p in got))


# ------------------------------------------------------------------ acyclicity

@seeds
def test_names_add_edge_rejects_cycles(seed):
    spec = random_spec(seed, max_n=20)
    G = spec.build()
    B = Brute(spec)
    rng = random.Random(seed + 9)
    existing = set((u, v) for (u, v, t) in spec.edges) | set((v, u) for (u, v, t) in spec.edges)
    tried = 0
    for _ in range(30):
        u = rng.choice(spec.names)
        v = rng.choice(spec.names)
        if (u, v) in existing:
            continue
        before = (names_of(G.nodes), etuples(G.edges), G.struct_version)
        closes = u == v or u in B.desc([v])
        if closes:
            with pytest.raises(ValueError):
                G.names.add_edge(u, v, 'directed')
            assert (names_of(G.nodes), etuples(G.edges), G.struct_version) == before
        else:
            G.names.add_edge(u, v, 'directed')
            spec.edges.append((u, v, 'directed'))
            existing.add((u, v)); existing.add((v, u))
            B = Brute(spec)
            assert is_linear_extension(G.names.topological_order(), spec)
            assert G.struct_version != before[2]
        tried += 1
    # legacy graph rebuilt from final spec agrees with what we built through names.add_edge
    assert set(etuples(G.edges)) == set(spec.edges)
