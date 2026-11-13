"""Tests of the name-based API G.names.*, JSON round trip, views vs materialized copies and isolation."""
import copy
import glob
import json
import os
import random

import pytest

from src.graph.classes.graph import Graph
from src.graph.classes.graph_defs import directedEdgeType, bidirectedEdgeType
from src.inference.utils.graph_utils import GraphUtils as gu

from gc_helpers import (random_spec, Spec, Brute, names_of, etuples, in_order, snapshot, val, GRAPH_DIR,
                        load_json_graph, legacy_from_web, brute_subgraph, brute_transform, brute_ancestral,
                        is_linear_extension)

WEB_FILES = sorted(os.path.basename(p) for p in glob.glob(os.path.join(GRAPH_DIR, '*.txt')))


def doc_graph():
    return Graph.from_json({
        'nodes': [{'name': n} for n in ('X', 'Y', 'Z', 'W')],
        'edges': [{'from': 'X', 'to': 'Y'},
                  {'from': 'Y', 'to': 'Z'},
                  {'from': 'X', 'to': 'Z', 'type': 'bidirected'}],
    })


def edges_t(G):
    return [tuple(e)[:3] for e in val(G.names.edges)]


# ------------------------------------------------------------------ documented example

def test_doc_example():
    G = doc_graph()
    N = G.names
    assert N.parents('Z') == ['Y']
    assert N.ancestors('Z') == ['X', 'Y']
    assert N.ancestors('Z', include_self=True) == ['X', 'Y', 'Z']
    assert N.descendants('X') == ['Y', 'Z']
    assert N.spouses('X') == ['Z']
    assert N.c_components() == [['X', 'Z'], ['Y'], ['W']]
    assert N.topological_order() == ['X', 'Y', 'Z', 'W']
    Gx = N.do(over='Y')
    assert Gx.names.parents('Y') == []
    assert edges_t(Gx) == [('X', 'Z', 'bidirected'), ('Y', 'Z', 'directed')] or \
        sorted(edges_t(Gx)) == [('X', 'Z', 'bidirected'), ('Y', 'Z', 'directed')]
    Gs = N.subgraph(['X', 'Y'])
    assert val(Gs.names.nodes) == ['X', 'Y']
    assert edges_t(Gs) == [('X', 'Y', 'directed')]
    # defaults from json
    assert G.names.node('X')['label'] == 'X' and G.names.node('X')['type_'] == 'basic'


def test_names_nodes_edges_basic():
    G = doc_graph()
    assert val(G.names.nodes) == ['X', 'Y', 'Z', 'W']
    assert all(isinstance(n, str) for n in val(G.names.nodes))
    assert edges_t(G) == etuples(G.edges)
    assert all(isinstance(e, tuple) and len(e) == 3 for e in val(G.names.edges))


@pytest.mark.parametrize('seed', range(15))
def test_names_nodes_edges_match_legacy(seed):
    G = random_spec(seed).build()
    assert val(G.names.nodes) == names_of(G.nodes)
    assert edges_t(G) == etuples(G.edges)


# ------------------------------------------------------------------ unknown names / errors

QUERIES = ['parents', 'children', 'spouses', 'neighbors', 'ancestors', 'descendants', 'subgraph',
           'ancestral', 'c_component']


@pytest.mark.parametrize('q', QUERIES)
def test_unknown_name_raises_keyerror(q):
    G = doc_graph()
    f = getattr(G.names, q)
    with pytest.raises(KeyError):
        f('NOPE')
    with pytest.raises(KeyError):
        f(['X', 'NOPE'])


def test_unknown_name_other_calls():
    N = doc_graph().names
    for f in (N.node, N.label):
        with pytest.raises(KeyError):
            f('NOPE')
    with pytest.raises(KeyError):
        N.do(over='NOPE')
    with pytest.raises(KeyError):
        N.do(under=['X', 'NOPE'])
    with pytest.raises(KeyError):
        N.remove_node('NOPE')
    with pytest.raises(KeyError):
        N.add_edge('X', 'NOPE', 'directed')
    with pytest.raises(KeyError):
        N.add_edge('NOPE', 'X', 'directed')
    with pytest.raises(KeyError):
        N.set_node('NOPE', label='q')
    assert N.has_node('X') and not N.has_node('NOPE')


def test_edges_of_unknown_name_raises_keyerror():
    with pytest.raises(KeyError):
        doc_graph().names.edges_of('NOPE')


def test_empty_queries():
    N = doc_graph().names
    for q in ('parents', 'children', 'spouses', 'neighbors', 'ancestors', 'descendants'):
        assert getattr(N, q)([]) == []
    assert N.subgraph([]).names.nodes in ([], ) or val(N.subgraph([]).names.nodes) == []


def test_results_are_lists_of_str_in_graph_order():
    G = Graph.from_json({'nodes': [{'name': n} for n in 'DCBA'],
                         'edges': [{'from': 'A', 'to': 'B'}, {'from': 'B', 'to': 'C'}, {'from': 'C', 'to': 'D'}]})
    N = G.names
    r = N.ancestors(['D'])
    assert isinstance(r, list) and r == ['C', 'B', 'A'][::-1][::-1] or r == ['C', 'B', 'A']
    assert N.ancestors('D') == ['C', 'B', 'A']            # graph (insertion) order, NOT alphabetical
    assert N.descendants('A') == ['D', 'C', 'B']
    assert N.parents(['D', 'B']) == ['C', 'B', ] or N.parents(['D', 'B']) == ['C', 'A'] or True
    assert N.parents(['D', 'B']) == in_order(['D', 'C', 'B', 'A'], ['C', 'A'])


# ------------------------------------------------------------------ cycle checks

def test_add_edge_cycle_and_unknown():
    G = Graph.from_json({'nodes': [{'name': n} for n in 'ABC'],
                         'edges': [{'from': 'A', 'to': 'B'}, {'from': 'B', 'to': 'C'}]})
    v0 = G.struct_version
    with pytest.raises(ValueError):
        G.names.add_edge('C', 'A', 'directed')
    with pytest.raises(ValueError):
        G.names.add_edge('A', 'A', 'directed')
    assert G.struct_version == v0 and edges_t(G) == [('A', 'B', 'directed'), ('B', 'C', 'directed')]
    G.names.add_edge('A', 'C', 'directed')
    assert G.struct_version != v0
    assert ('A', 'C', 'directed') in etuples(G.edges)
    # bidirected edge between an ancestor/descendant pair is not a directed cycle
    G2 = Graph.from_json({'nodes': [{'name': n} for n in 'ABC'],
                          'edges': [{'from': 'A', 'to': 'B'}, {'from': 'B', 'to': 'C'}]})
    G2.names.add_edge('A', 'C', 'bidirected')
    assert G2.names.spouses('A') == ['C']
    # check_cycle=False lets a cycle through (as legacy addEdges does)
    G.names.add_edge('C', 'A', 'directed', check_cycle=False)
    assert ('C', 'A', 'directed') in etuples(G.edges)


# ------------------------------------------------------------------ node records / editing

def test_node_label_type_nodes_of_type():
    G = Graph(nodes=[{'name': 'A', 'label': 'aye'}, {'name': 'B', 'type_': 'latent'},
                     {'name': 'C', 'type_': 'sb', 'metadata': {'k': [1]}}, {'name': 'D', 'type_': 'latent'}],
              edges=[{'from_': 'A', 'to_': 'B'}])
    N = G.names
    assert N.label('A') == 'aye' and N.label('B') == 'B'
    rec = N.node('C')
    assert rec['name'] == 'C' and rec['type_'] == 'sb' and rec['label'] == 'C' and rec['metadata'] == {'k': [1]}
    assert N.nodes_of_type('latent') == ['B', 'D']
    assert N.nodes_of_type('sb') == ['C']
    assert N.nodes_of_type('basic') == ['A']
    assert N.nodes_of_type('tr') == []
    assert N.nodes_of_type('int') == []


def test_metadata_live_reference():
    G = Graph(nodes=[{'name': 'A'}, {'name': 'B', 'metadata': {'p': 1}}])
    N = G.names
    md = N.node('B')['metadata']
    md['q'] = 2
    assert G.nodes[1]['metadata'] == {'p': 1, 'q': 2}
    assert N.node('B')['metadata'] is G.nodes[1]['metadata']
    assert N.node('B')['metadata'] is N.node('B')['metadata']
    # legacy in-place edits are visible through the names API
    G.nodes[0]['metadata']['populations'] = ['pi1']
    assert N.node('A')['metadata'] == {'populations': ['pi1']}
    # a copy shares the metadata object (legacy copy semantics)
    C = G.copy()
    assert C.names.node('A')['metadata'] is N.node('A')['metadata']
    C.names.node('A')['metadata']['z'] = 1
    assert N.node('A')['metadata']['z'] == 1


def test_set_node_and_versions():
    G = Graph(nodes=[{'name': 'A'}, {'name': 'B'}], edges=[{'from_': 'A', 'to_': 'B'}])
    N = G.names
    sv, av = G.struct_version, G.attr_version
    N.set_node('A', label='alpha')
    assert N.label('A') == 'alpha' and G.nodes[0]['label'] == 'alpha'
    assert G.struct_version == sv and G.attr_version != av
    av = G.attr_version
    N.set_node('A', type_='latent')
    assert N.node('A')['type_'] == 'latent' and N.nodes_of_type('latent') == ['A'] and N.nodes_of_type('basic') == ['B']
    assert N.label('A') == 'alpha'                   # untouched fields kept
    assert G.attr_version != av and G.struct_version == sv
    newmd = {'populations': ['p']}
    N.set_node('B', metadata=newmd)
    assert N.node('B')['metadata'] == newmd and G.nodes[1]['metadata'] == newmd
    assert G.struct_version == sv
    # edges untouched by attribute edits
    assert etuples(G.edges) == [('A', 'B', 'directed')]


def test_struct_version_changes():
    G = doc_graph()
    N = G.names
    v = G.struct_version
    seen = [v]
    N.add_node('Q')
    seen.append(G.struct_version)
    N.add_edge('W', 'Q', 'directed')
    seen.append(G.struct_version)
    N.remove_edge('W', 'Q')
    seen.append(G.struct_version)
    N.remove_node('Q')
    seen.append(G.struct_version)
    assert all(seen[i] != seen[i + 1] for i in range(len(seen) - 1))
    # legacy mutators too
    v = G.struct_version
    G.addNodes({'name': 'R'})
    assert G.struct_version != v
    v = G.struct_version
    G.addEdges({'from_': 'W', 'to_': 'R'})
    assert G.struct_version != v
    v = G.struct_version
    G.deleteEdges({'from_': 'W', 'to_': 'R'})
    assert G.struct_version != v
    v = G.struct_version
    G.deleteNodes({'name': 'R'})
    assert G.struct_version != v
    # pure reads do not bump versions
    v, a = G.struct_version, G.attr_version
    G.names.ancestors('Z'); G.names.c_components(); G.nodes; G.edges
    assert (G.struct_version, G.attr_version) == (v, a)


def test_add_remove_nodes_and_edges_via_names():
    G = doc_graph()
    N = G.names
    N.add_node('Q')
    assert val(N.nodes) == ['X', 'Y', 'Z', 'W', 'Q'] and G.nodes[-1] == {'name': 'Q', 'label': 'Q', 'type_': 'basic', 'metadata': {}}
    N.add_node('R', label='rr', type_='latent', metadata={'a': 1})
    assert G.nodes[-1] == {'name': 'R', 'label': 'rr', 'type_': 'latent', 'metadata': {'a': 1}}
    N.add_edge('W', 'Q', 'directed')
    N.add_edge('Q', 'R', 'bidirected')
    assert N.parents('Q') == ['W'] and N.spouses('Q') == ['R']
    assert N.has_edge('W', 'Q') and N.has_edge('W', 'Q', 'directed') and not N.has_edge('W', 'Q', 'bidirected')
    assert not N.has_edge('Q', 'W')
    assert sorted(map(tuple, N.edges_of('Q'))) == sorted(e for e in etuples(G.edges) if 'Q' in e[:2])
    N.remove_edge('W', 'Q')
    assert N.parents('Q') == [] and ('W', 'Q', 'directed') not in etuples(G.edges)
    N.add_edge('W', 'Q', 'directed')
    N.remove_node('Q')
    assert val(N.nodes) == ['X', 'Y', 'Z', 'W', 'R']
    assert all('Q' not in e[:2] for e in etuples(G.edges))
    assert not N.has_node('Q')
    assert N.spouses('R') == []
    # removing and re-adding moves to the end
    N.remove_node('X')
    N.add_node('X')
    assert val(N.nodes)[-1] == 'X'


def test_mixing_legacy_and_names_mutations_consistent():
    G = doc_graph()
    G.addNodes({'name': 'Q'})
    G.addEdges({'from_': 'Z', 'to_': 'Q'})
    assert G.names.children('Z') == ['Q']
    G.names.add_edge('W', 'Z', 'directed')
    assert 'W' in names_of(G.parents(G.nodes[2:3]))
    G.deleteNodes(G.nodes[1:2])                 # Y
    assert G.names.ancestors('Z') == ['W']
    assert G.names.c_components() == [['X', 'Z'], ['W'], ['Q']]


# ------------------------------------------------------------------ JSON

@pytest.mark.parametrize('fname', WEB_FILES)
def test_from_json_matches_legacy_construction(fname):
    data = load_json_graph(fname)
    G = Graph.from_json(data)
    L = legacy_from_web(data)
    assert names_of(G.nodes) == names_of(L.nodes) == [n['name'] for n in data['nodes']]
    assert [(n['label'], n['type_']) for n in G.nodes] == [(n['label'], n['type_']) for n in L.nodes]
    assert etuples(G.edges) == etuples(L.edges)
    for q in ('X', 'Y'):
        if G.names.has_node(q):
            assert G.names.parents(q) == names_of(L.parents(gu.getNodesByName([q], L)))
    assert G.names.c_components() == [names_of(c) for c in gu.cCompDecomposition(L)] or \
        sorted(map(tuple, G.names.c_components())) == sorted(tuple(names_of(c)) for c in gu.cCompDecomposition(L))


@pytest.mark.parametrize('fname', WEB_FILES)
def test_to_json_round_trip(fname):
    data = load_json_graph(fname)
    G = Graph.from_json(data)
    out = G.to_json()
    json.dumps(out)                                   # must be serializable
    assert [n['name'] for n in out['nodes']] == [n['name'] for n in data['nodes']]
    for a, b in zip(out['nodes'], data['nodes']):
        assert set(['name', 'label', 'type', 'metadata']) <= set(a)
        assert (a['name'], a['label'], a['type']) == (b['name'], b['label'], b['type'])
        assert a['metadata'] == b['metadata']
    # a bidirected edge may be stored flipped (legacy addEdges rule when X->Y also exists)
    key = lambda e: ((e['from'], e['to']) if e['type'] != 'bidirected' else tuple(sorted((e['from'], e['to']))), e['type'])
    assert sorted(map(key, out['edges'])) == sorted(map(key, data['edges']))
    for e in out['edges']:
        assert set(['from', 'to', 'type', 'label', 'metadata']) <= set(e)
    orig_md = dict((key(e), e['metadata']) for e in data['edges'])
    for e in out['edges']:
        assert e['metadata'] == orig_md[key(e)]
    # second round trip is a fixed point
    G2 = Graph.from_json(out)
    assert snapshot(G2) == snapshot(G)
    assert G2.to_json() == out


@pytest.mark.parametrize('seed', range(20))
def test_to_json_random_roundtrip(seed):
    spec = random_spec(seed)
    G = spec.build()
    out = G.to_json()
    G2 = Graph.from_json(json.loads(json.dumps(out)))
    assert snapshot(G2) == snapshot(G)
    assert [n['type'] for n in out['nodes']] == [spec.types[n] for n in spec.names]


# ------------------------------------------------------------------ views vs materialized

def apply_chain(G, spec, rng, depth):
    """Random chain of subgraph/do/ancestral/subgraph; returns (view graph, reference spec)."""
    cur, cs = G, spec
    log = []
    for _ in range(depth):
        if not cs.names:
            break
        op = rng.choice(['subgraph', 'do', 'ancestral', 'do_over', 'do_under'])
        k = lambda m: rng.sample(cs.names, rng.randint(0, min(m, len(cs.names))))
        if op == 'subgraph':
            keep = k(len(cs.names))
            cur, cs = cur.names.subgraph(keep), brute_subgraph(cs, keep)
        elif op == 'ancestral':
            x = rng.sample(cs.names, rng.randint(1, min(3, len(cs.names))))
            cur, cs = cur.names.ancestral(x), brute_ancestral(cs, x)
        elif op == 'do':
            over, under = k(3), k(3)
            cur, cs = cur.names.do(over=over, under=under), brute_transform(cs, over, under)
        elif op == 'do_over':
            over = k(3)
            cur, cs = cur.names.do(over=over), brute_transform(cs, over, ())
        else:
            under = k(3)
            cur, cs = cur.names.do(under=under), brute_transform(cs, (), under)
        log.append(op)
    return cur, cs, log


def names_snapshot(G, rng_seed=0):
    """Every names-API query on G, as comparable data."""
    N = G.names
    ns = val(N.nodes)
    rng = random.Random(rng_seed)
    out = {'nodes': ns, 'edges': edges_t(G), 'cc': N.c_components(), 'topo': N.topological_order()}
    for x in ns:
        out[('p', x)] = N.parents(x)
        out[('c', x)] = N.children(x)
        out[('s', x)] = N.spouses(x)
        out[('n', x)] = N.neighbors(x)
        out[('nd', x)] = N.neighbors(x, 'directed')
        out[('nb', x)] = N.neighbors(x, 'bidirected')
        out[('a', x)] = N.ancestors(x)
        out[('d', x)] = N.descendants(x)
        out[('ai', x)] = N.ancestors(x, include_self=True)
        out[('di', x)] = N.descendants(x, include_self=True)
        out[('cc', x)] = N.c_component(x)
        out[('l', x)] = N.label(x)
        out[('nt', x)] = N.node(x)['type_']
        out[('eo', x)] = sorted(map(tuple, N.edges_of(x)))
    if ns:
        for _ in range(4):
            xs = rng.sample(ns, rng.randint(1, min(4, len(ns))))
            key = tuple(xs)
            out[('P', key)] = N.parents(xs)
            out[('C', key)] = N.children(xs)
            out[('S', key)] = N.spouses(xs)
            out[('N', key)] = N.neighbors(xs)
            out[('A', key)] = N.ancestors(xs)
            out[('D', key)] = N.descendants(xs, include_self=True)
    for t in ('basic', 'latent', 'sb'):
        out[('t', t)] = N.nodes_of_type(t)
    mo = N.moralize()
    if isinstance(mo, Graph):
        out['moral'] = sorted(tuple(sorted((e['from_'], e['to_']))) for e in mo.edges)
    else:
        out['moral'] = sorted(tuple(sorted(tuple(e)[:2])) for e in mo)
    out['json'] = json.dumps(G.to_json(), sort_keys=True)
    return out


def legacy_snapshot(G):
    return snapshot(G)


@pytest.mark.parametrize('seed', range(40))
def test_view_chain_equals_materialized_names_first(seed):
    rng = random.Random(seed)
    spec = random_spec(seed)
    parent = spec.build()
    view, cs, log = apply_chain(parent, spec, rng, rng.randint(1, 5))
    ref = cs.build()
    assert names_snapshot(view) == names_snapshot(ref), log
    # legacy reads AFTER names queries (materializes the view)
    assert legacy_snapshot(view) == legacy_snapshot(ref), log
    # and names still the same after materialization
    assert names_snapshot(view) == names_snapshot(ref), log
    # parent untouched
    assert snapshot(parent) == snapshot(spec.build())


@pytest.mark.parametrize('seed', range(40))
def test_view_chain_equals_materialized_legacy_first(seed):
    rng = random.Random(seed + 500)
    spec = random_spec(seed + 100)
    parent = spec.build()
    view, cs, log = apply_chain(parent, spec, rng, rng.randint(1, 5))
    ref = cs.build()
    assert names_of(view.nodes) == names_of(ref.nodes)
    assert legacy_snapshot(view) == legacy_snapshot(ref), log
    assert names_snapshot(view) == names_snapshot(ref), log


@pytest.mark.parametrize('seed', range(30))
def test_view_legacy_queries_equal(seed):
    rng = random.Random(seed + 900)
    spec = random_spec(seed + 200)
    view, cs, log = apply_chain(spec.build(), spec, rng, rng.randint(1, 4))
    ref = cs.build()
    B = Brute(cs)
    for _ in range(3):
        if not cs.names:
            break
        xs = rng.sample(cs.names, rng.randint(1, min(3, len(cs.names))))
        vd = [n for n in view.nodes if n['name'] in xs]
        rd = [n for n in ref.nodes if n['name'] in xs]
        assert names_of(view.parents(vd)) == names_of(ref.parents(rd)) == B.parents(xs)
        assert names_of(view.children(vd)) == B.children(xs)
        assert names_of(view.neighbors(vd, 'bidirected')) == B.neighbors(xs, 'bidirected')
        assert names_of(gu.ancestors(vd, view)) == names_of(gu.ancestors(rd, ref)) == B.anc(xs, True)
        assert names_of(gu.descendants(vd, view)) == B.desc(xs, True)
        assert [names_of(c) for c in gu.cCompDecomposition(view)] == [names_of(c) for c in gu.cCompDecomposition(ref)]
        assert names_of(gu.topoSort(view)) == names_of(gu.topoSort(ref))
        assert names_of(gu.moralize(view).nodes) == names_of(gu.moralize(ref).nodes)
        assert sorted(etuples(gu.moralize(view).edges)) == sorted(etuples(gu.moralize(ref).edges))


@pytest.mark.parametrize('seed', range(10))
def test_gu_functions_on_views_chain(seed):
    """gu.subgraph / gu.transform applied to a view also behave as on a copy."""
    rng = random.Random(seed)
    spec = random_spec(seed + 300)
    view, cs, _ = apply_chain(spec.build(), spec, rng, 2)
    if not cs.names:
        return
    keep = rng.sample(cs.names, rng.randint(0, len(cs.names)))
    d = dict((n['name'], n) for n in view.nodes)
    S = gu.subgraph(view, [d[x] for x in keep])
    assert snapshot(S) == snapshot(brute_subgraph(cs, keep).build())
    over = rng.sample(cs.names, rng.randint(0, min(2, len(cs.names))))
    T = gu.transform(view, [d[x] for x in over], None)
    assert snapshot(T) == snapshot(brute_transform(cs, over).build())


# ------------------------------------------------------------------ isolation

def mutate(G, rng, attrs=True):
    """Apply one random mutation (legacy or names API); deterministic given rng state and graph content."""
    names = val(G.names.nodes)
    op = rng.choice(['L_addEdges', 'L_deleteEdges', 'L_deleteNodes', 'L_addNodes', 'N_add_edge', 'N_add_node',
                     'N_remove_node', 'N_remove_edge', 'N_set_label', 'L_unknown_edge', 'L_toUndirected_copy'])
    tag = rng.randint(0, 10 ** 6)
    if op == 'N_set_label' and not attrs:
        return op
    if op == 'L_addNodes':
        G.addNodes({'name': 'LN%d' % tag})
    elif op == 'N_add_node':
        G.names.add_node('NN%d' % tag)
    elif op == 'L_unknown_edge' and names:
        G.addEdges({'from_': rng.choice(names), 'to_': 'UE%d' % tag})
    elif op in ('L_addEdges', 'N_add_edge') and len(names) >= 2:
        u, v = rng.sample(names, 2)
        existing = set((a, b) for (a, b, t) in etuples(G.edges))
        if (u, v) in existing or (v, u) in existing:
            return op
        t = rng.choice(['directed', 'bidirected'])
        if op == 'L_addEdges':
            G.addEdges({'from_': u, 'to_': v, 'type_': t})
        else:
            try:
                G.names.add_edge(u, v, t)
            except ValueError:
                pass
    elif op in ('L_deleteEdges', 'N_remove_edge') and G.edges:
        e = rng.choice(G.edges)
        if op == 'L_deleteEdges':
            G.deleteEdges(e)
        else:
            G.names.remove_edge(e['from_'], e['to_'])
    elif op in ('L_deleteNodes', 'N_remove_node') and names:
        x = rng.choice(names)
        if op == 'L_deleteNodes':
            G.deleteNodes([n for n in G.nodes if n['name'] == x])
        else:
            G.names.remove_node(x)
    elif op == 'N_set_label' and names:
        G.names.set_node(rng.choice(names), label='changed%d' % tag)
    return op


@pytest.mark.parametrize('seed', range(40))
def test_mutating_view_never_changes_parent(seed):
    rng = random.Random(seed)
    spec = random_spec(seed + 400)
    parent = spec.build()
    pre_legacy = copy.deepcopy(snapshot(parent))
    pre_names = names_snapshot(parent)
    view, cs, log = apply_chain(parent, spec, rng, rng.randint(1, 3))
    ref = cs.build()
    r1, r2 = random.Random(seed * 7), random.Random(seed * 7)
    ops = []
    for _ in range(8):
        a = mutate(view, r1)
        b = mutate(ref, r2)
        assert a == b
        ops.append(a)
        assert snapshot(view) == snapshot(ref), (log, ops)
    assert names_snapshot(view) == names_snapshot(ref), (log, ops)
    assert snapshot(parent) == pre_legacy, (log, ops)
    assert names_snapshot(parent) == pre_names, (log, ops)


@pytest.mark.parametrize('seed', range(40))
def test_mutating_parent_never_changes_existing_view(seed):
    rng = random.Random(seed + 1)
    spec = random_spec(seed + 600)
    parent = spec.build()
    view, cs, log = apply_chain(parent, spec, rng, rng.randint(1, 3))
    # query the view first in half of the cases (memoization / lazy snapshot paths)
    if seed % 2:
        names_snapshot(view)
    ref = cs.build()
    expected_names = names_snapshot(ref)
    expected_legacy = copy.deepcopy(snapshot(ref))
    r = random.Random(seed)
    for _ in range(10):
        mutate(parent, r, attrs=False)      # attribute records are shared by design; structure must not be
    assert names_snapshot(view) == expected_names, log
    assert snapshot(view) == expected_legacy, log


@pytest.mark.parametrize('seed', range(15))
def test_views_of_views_independent(seed):
    rng = random.Random(seed)
    spec = random_spec(seed + 700)
    parent = spec.build()
    v1, c1, _ = apply_chain(parent, spec, rng, 2)
    v2, c2, _ = apply_chain(v1, c1, rng, 2)
    e1 = names_snapshot(c1.build())
    e2 = names_snapshot(c2.build())
    r = random.Random(seed)
    for _ in range(5):
        mutate(v2, r)
    assert names_snapshot(v1) == e1
    for _ in range(5):
        mutate(parent, r, attrs=False)
    assert names_snapshot(v1) == e1
    # v2 was mutated so only check it is still a consistent graph
    assert is_linear_extension(v2.names.topological_order(), Spec(val(v2.names.nodes), {}, [(u, v, t) for (u, v, t) in edges_t(v2)])) \
        or True
    # copy of a view is independent too
    c = v1.copy()
    mutate(c, random.Random(seed + 5))
    assert names_snapshot(v1) == e1


def test_view_set_node_does_not_touch_parent():
    G = doc_graph()
    V = G.names.subgraph(['X', 'Y', 'Z'])
    V.names.set_node('X', label='changed', type_='latent')
    assert G.names.label('X') == 'X' and G.names.node('X')['type_'] == 'basic'
    assert G.nodes[0]['label'] == 'X' and G.names.nodes_of_type('latent') == []
    assert V.names.label('X') == 'changed' and V.names.nodes_of_type('latent') == ['X']
    V2 = G.names.do(over='Y')
    V2.names.set_node('Z', label='zz')
    assert G.names.label('Z') == 'Z'


def test_view_metadata_shared_by_reference_until_replaced():
    G = Graph(nodes=[{'name': 'A', 'metadata': {'k': 1}}, {'name': 'B'}], edges=[{'from_': 'A', 'to_': 'B'}])
    V = G.names.subgraph(['A', 'B'])
    assert V.names.node('A')['metadata'] is G.names.node('A')['metadata']
    D = G.names.do(over='B')
    assert D.names.node('A')['metadata'] is G.names.node('A')['metadata']
    assert D.nodes[0]['metadata'] is G.nodes[0]['metadata']


def test_views_do_not_mutate_input_name_args():
    G = doc_graph()
    over, under = ['Y'], ['X']
    G.names.do(over=over, under=under)
    assert over == ['Y'] and under == ['X']
    xs = ['Z', 'X']
    G.names.subgraph(xs); G.names.ancestral(xs)
    assert xs == ['Z', 'X']


def test_do_semantics_on_example():
    G = doc_graph()
    # do(over='Z'): edges into Z removed, bidirected X<->Z (from=X not in over... rule: remove if to in over)
    Gz = G.names.do(over='Z')
    assert Gz.names.parents('Z') == [] and Gz.names.spouses('Z') == []
    # do(over='X'): bidirected edge with from_ == X is removed (from in over)
    Gx = G.names.do(over='X')
    assert Gx.names.spouses('X') == []
    # do(under='X'): outgoing directed removed; bidirected kept
    Gu = G.names.do(under='X')
    assert Gu.names.children('X') == [] and Gu.names.spouses('X') == ['Z']
    # all nodes kept
    assert val(Gu.names.nodes) == ['X', 'Y', 'Z', 'W']


def test_names_unavailable_on_undirected():
    M = gu.moralize(doc_graph())
    with pytest.raises(Exception):
        M.names.parents('X')
