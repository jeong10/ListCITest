"""Regression tests for the Phase 2 code review (case-insensitive name collisions, dict subclasses,
numeric-string iteratees). The fast paths must agree with the legacy implementations."""
from collections import defaultdict

import pydash
import pytest

import src.path_analysis.d_separation as dsep_mod
import src.adjustment.adjustment_sets_utils as asu
from src.graph.classes.graph import Graph
from src.inference.utils.set_utils import SetUtils as su
from src.projection.projection_utils import ProjectionUtils as pu
from src.path_analysis.d_separation import DSeparation


def nd(*names):
    return [{'name': n} for n in names]


def graph(nodes, edges):
    return Graph.from_json({'nodes': [{'name': n} for n in nodes], 'edges': edges})


def legacy(monkeypatch, fn):
    """Result of ``fn`` with the native fast path disabled."""
    with monkeypatch.context() as m:
        m.setattr(dsep_mod, 'separation_masks', lambda *a: None)
        m.setattr(asu, 'separation_masks', lambda *a: None)
        return fn()


COLLISIONS = [
    (['a', 'A', 'Y'], [{'from': 'A', 'to': 'Y'}], ['A'], ['Y'], []),
    (['X', ' X', 'Y'], [{'from': ' X', 'to': 'Y'}], [' X'], ['Y'], []),
    (['u_{ab}', 'a', 'b'], [{'from': 'a', 'to': 'b', 'type': 'bidirected'}], ['a'], ['b'], []),
    (['m', 'M', 'X', 'Y'], [{'from': 'X', 'to': 'M'}, {'from': 'M', 'to': 'Y'}], ['X'], ['Y'], []),
    (['ab', 'c', 'a', 'bc'], [{'from': 'ab', 'to': 'c', 'type': 'bidirected'},
                              {'from': 'a', 'to': 'bc', 'type': 'bidirected'}], ['ab'], ['c'], []),
]


@pytest.mark.parametrize('nodes,edges,X,Y,Z', COLLISIONS)
def test_name_collisions_match_legacy(monkeypatch, nodes, edges, X, Y, Z):
    G = graph(nodes, edges)
    args = (G, nd(*X), nd(*Y), nd(*Z))
    assert asu.TestSep(*args) == legacy(monkeypatch, lambda: asu.TestSep(*args))
    assert asu.TestSepUnprojected(*args) == legacy(monkeypatch, lambda: asu.TestSepUnprojected(*args))
    assert asu.TestSepUnprojected(*args) == asu.TestSep(pu.unproject(G), *args[1:])
    assert DSeparation.test(*args) == legacy(monkeypatch, lambda: DSeparation.test(*args))


def test_dict_subclass_with_missing_is_not_mutated():
    dd = defaultdict(list)
    other = [{'name': None}]
    assert su.union([dd], other, 'name') == pydash.arrays.union_by([dd], other, iteratee='name')
    assert su.difference([dd], other, 'name') == pydash.arrays.difference_by([dd], other, iteratee='name')
    assert dict(dd) == {}


@pytest.mark.parametrize('key', ['1', ' 1 ', '01', '-1'])
def test_numeric_string_iteratee(key):
    a = [{1: 'a'}, {1: 'b'}, {-1: 'c'}]
    b = [{1: 'a'}]
    assert su.union(a, [], key) == pydash.arrays.union_by(a, [], iteratee=key)
    assert su.difference(a, b, key) == pydash.arrays.difference_by(a, b, iteratee=key)
    assert su.intersection(a, b, key) == pydash.arrays.intersection_by(a, b, iteratee=key)
