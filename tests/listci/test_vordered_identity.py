"""Review regression: W holds the caller's own V / Vordered dict objects (as the legacy list-based ListCI does),
Z holds dicts decoded from the graph."""
import src.testable_implications.conditional_independencies as cim
from src.graph.classes.graph import Graph
from src.inference.utils.graph_utils import GraphUtils as gu

CIA = cim.ConditionalIndependencies


def make():
    return Graph.from_json({
        'nodes': [{'name': n, 'label': n.lower()} for n in ('A', 'B', 'C', 'D', 'E')],
        'edges': [{'from': 'A', 'to': 'B'}, {'from': 'B', 'to': 'C'}, {'from': 'A', 'to': 'D', 'type': 'bidirected'},
                  {'from': 'C', 'to': 'D'}, {'from': 'D', 'to': 'E'}, {'from': 'B', 'to': 'E', 'type': 'bidirected'}]})


def run(monkeypatch, fast, V):
    G = make()

    with monkeypatch.context() as m:
        if not fast:
            m.setattr(cim, 'list_ci_fast', lambda G, V: None)

        return CIA.ListCI(G, G.nodes, V), V


def test_user_vordered_dicts_flow_into_W(monkeypatch):
    G = make()
    names = [n['name'] for n in gu.topoSort(G)]
    out = []

    for fast in (True, False):
        V = [{'name': n} for n in names]          # bare dicts, unlike the graph's node dicts
        res, V = run(monkeypatch, fast, V)
        out.append((res, V))

    (fast_res, fV), (leg_res, lV) = out
    assert fast_res == leg_res
    assert any(ci['W'] for ci in fast_res)

    for ci in fast_res:
        for w in ci['W']:
            assert any(w is v for v in fV)          # identity with the caller's dicts
            assert 'label' not in w


def test_default_V_matches_legacy(monkeypatch):
    a, _ = run(monkeypatch, True, None)
    b, _ = run(monkeypatch, False, None)
    assert a == b
