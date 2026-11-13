"""m-/d-separation (name API and legacy entry points) against networkx on an explicit latent DAG."""
import random

import pytest

from src.graph.classes.graph import Graph
from src.adjustment.adjustment_sets_utils import TestSep, TestSepUnprojected
from src.path_analysis.d_separation import DSeparation
from src.projection.projection_utils import ProjectionUtils as pu

from sep_helpers import (Spec, random_admg, random_triple, nx_dsep, oracle_m, dicts, corpus_specs)

N_GRAPHS = 500
TRIPLES = 8


def graphs(seed, n=N_GRAPHS, **kw):
    rng = random.Random(seed)
    for _ in range(n):
        yield rng, random_admg(rng, **kw)


# ------------------------------------------------------------------ name API vs oracle
def test_m_separated_random():
    cnt = [0, 0]
    for rng, S in graphs(10):
        G = S.build()
        Gj = S.build_json()
        for _ in range(TRIPLES):
            X, Y, Z = random_triple(rng, S.names)
            exp = oracle_m(S, X, Y, Z)
            cnt[exp] += 1
            assert G.names.m_separated(X, Y, Z) is exp, (S.names, S.edges, X, Y, Z)
            assert Gj.names.m_separated(X, Y, Z) is exp
            assert G.names.m_separated(X, Y, tuple(Z)) is exp
            assert G.names.m_separated(X[0], Y[0], Z) is oracle_m(S, X[:1], Y[:1], Z)
    assert cnt[0] > 300 and cnt[1] > 300      # both outcomes exercised


def test_m_separated_default_z_and_iterables():
    G = Spec(['A', 'B', 'C'], [('A', 'B', 'directed')]).build()
    assert G.names.m_separated('A', 'C') is True
    assert G.names.m_separated('A', 'B') is False
    assert G.names.m_separated(iter(['A']), (n for n in ['C']), set()) is True


def test_d_separated_dags_random():
    for rng, S in graphs(11):
        S = S.without_bi()
        G = S.build()
        D = S.latent_dag()
        for _ in range(TRIPLES):
            X, Y, Z = random_triple(rng, S.names)
            exp = nx_dsep(D, X, Y, Z)
            assert G.names.d_separated(X, Y, Z) is exp, (S.names, S.edges, X, Y, Z)
            assert G.names.m_separated(X, Y, Z) is exp


def test_d_separated_with_explicit_latent_nodes():
    for rng, S in graphs(12):
        L = S.latent_spec()
        G = L.build()
        Gj = L.build_json()
        D = S.latent_dag()
        vis = S.names
        for _ in range(TRIPLES):
            X, Y, Z = random_triple(rng, vis)
            exp = nx_dsep(D, X, Y, Z)
            assert G.names.d_separated(X, Y, Z) is exp
            assert Gj.names.d_separated(X, Y, Z) is exp
            assert G.names.m_separated(X, Y, Z) is exp
            # latent nodes may also be conditioned on / queried
            if L.names[len(vis):]:
                lat = rng.choice(L.names[len(vis):])
                rest = [n for n in vis if n not in X and n not in Y]
                Z2 = list(Z) + [lat]
                assert G.names.d_separated(X, Y, Z2) is nx_dsep(D, X, Y, Z2)


def test_overlap_rule():
    for rng, S in graphs(13, n=150):
        G = S.build()
        v = rng.choice(S.names)
        others = [n for n in S.names if n != v]
        Y = [v] + (rng.sample(others, min(len(others), rng.randint(0, 2))))
        X = [v] + (rng.sample(others, min(len(others), rng.randint(0, 1))))
        Z = [n for n in others if n not in X and n not in Y][:rng.randint(0, 2)]
        assert G.names.m_separated(X, Y, Z) is False
        Gd = S.without_bi().build()
        assert Gd.names.d_separated(X, Y, Z) is False


def test_value_error_and_key_error():
    S = Spec(['A', 'B', 'C'], [('A', 'B', 'directed'), ('B', 'C', 'bidirected')])
    G = S.build()
    with pytest.raises(ValueError):
        G.names.d_separated('A', 'C', [])
    with pytest.raises(ValueError):
        G.names.d_separated('A', 'C', ['B'])
    with pytest.raises(KeyError):
        G.names.m_separated('A', 'Q', [])
    with pytest.raises(KeyError):
        G.names.m_separated('Q', 'A', [])
    with pytest.raises(KeyError):
        G.names.m_separated('A', 'C', ['Q'])
    Gd = S.without_bi().build()
    for args in [('A', 'Q', []), ('Q', 'A', []), ('A', 'C', ['Q'])]:
        with pytest.raises(KeyError):
            Gd.names.d_separated(*args)
        with pytest.raises(KeyError):
            Gd.names.m_separated(*args)


# ------------------------------------------------------------------ views
def test_views_equal_rebuilt():
    for rng, S in graphs(14, n=300):
        G = S.build()
        k = rng.randint(1, len(S.names))
        keep = rng.sample(S.names, k)
        over = rng.sample(S.names, min(len(S.names), rng.randint(0, 3)))
        under = rng.sample(S.names, min(len(S.names), rng.randint(0, 3)))
        cases = [(G.names.subgraph(keep), S.sub(keep)),
                 (G.names.do(over=over, under=under), S.do(over, under)),
                 (G.names.do(over=over), S.do(over, [])),
                 (G.names.do(under=under), S.do([], under)),
                 (G.names.do(over=over, under=under).names.subgraph(keep), S.do(over, under).sub(keep))]
        for V, R in cases:
            RG = R.build()
            for _ in range(4):
                if len(R.names) < 2:
                    break
                X, Y, Z = random_triple(rng, R.names)
                exp = oracle_m(R, X, Y, Z)
                assert V.names.m_separated(X, Y, Z) is exp, (S.edges, keep, over, under, X, Y, Z)
                assert RG.names.m_separated(X, Y, Z) is exp
                if not R.has_bi():
                    assert V.names.d_separated(X, Y, Z) is exp
                else:
                    with pytest.raises(ValueError):
                        V.names.d_separated(X, Y, Z)
        # a view dropping the bidirected edges becomes d_separated-capable
        V = G.names.do(over=S.names)
        assert V.names.d_separated(S.names[:1], S.names[-1:], []) is True   # no edges left at all


def test_view_does_not_change_parent():
    for rng, S in graphs(15, n=100):
        G = S.build()
        before = []
        trip = [random_triple(rng, S.names) for _ in range(5)]
        for X, Y, Z in trip:
            before.append(G.names.m_separated(X, Y, Z))
        G.names.do(over=S.names[:2], under=S.names[-2:])
        G.names.subgraph(S.names[:3])
        assert before == [G.names.m_separated(X, Y, Z) for X, Y, Z in trip]


# ------------------------------------------------------------------ corpus
def test_corpus_files():
    rng = random.Random(16)
    for fname, S, data in corpus_specs():
        G = Graph.from_json(data)
        Gl = S.build()
        names = S.names
        pool = []
        if len(names) <= 7:
            import itertools
            for r in range(1, 3):
                for X in itertools.combinations(names, r):
                    for Y in itertools.combinations([n for n in names if n not in X], 1):
                        rest = [n for n in names if n not in X and n not in Y]
                        pool.append((list(X), list(Y), []))
                        for _ in range(3):
                            pool.append((list(X), list(Y), rng.sample(rest, rng.randint(0, len(rest)))))
        else:
            for _ in range(300):
                pool.append(random_triple(rng, names))
        D = S.latent_dag()
        for X, Y, Z in pool:
            exp = nx_dsep(D, X, Y, Z)
            assert G.names.m_separated(X, Y, Z) is exp, (fname, X, Y, Z)
            assert Gl.names.m_separated(X, Y, Z) is exp, (fname, X, Y, Z)
            assert TestSepUnprojected(G, dicts(X), dicts(Y), dicts(Z)) is exp
            assert DSeparation.test(G, dicts(X), dicts(Y), dicts(Z)) is exp
            if not S.has_bi():
                assert G.names.d_separated(X, Y, Z) is exp
                assert TestSep(G, dicts(X), dicts(Y), dicts(Z)) is exp
            else:
                with pytest.raises(ValueError):
                    G.names.d_separated(X, Y, Z)
            Sd = S.without_bi()
            assert TestSep(G, dicts(X), dicts(Y), dicts(Z)) is nx_dsep(Sd.latent_dag(), X, Y, Z)


# ------------------------------------------------------------------ legacy
def test_legacy_entry_points_random():
    for rng, S in graphs(20):
        G = S.build()
        Sd = S.without_bi()
        Dd = Sd.latent_dag()
        U = pu.unproject(G)
        for _ in range(TRIPLES):
            X, Y, Z = random_triple(rng, S.names)
            dX, dY, dZ = dicts(X), dicts(Y), dicts(Z)
            exp = oracle_m(S, X, Y, Z)
            assert TestSepUnprojected(G, dX, dY, dZ) is exp, (S.names, S.edges, X, Y, Z)
            assert DSeparation.test(G, dX, dY, dZ) is exp
            assert TestSep(U, dX, dY, dZ) is exp
            assert TestSep(G, dX, dY, dZ) is nx_dsep(Dd, X, Y, Z)
            # empty X or Y
            assert DSeparation.test(G, [], dY, dZ) is True
            assert DSeparation.test(G, dX, [], dZ) is True


def overlapping(rng, names):
    k = len(names)
    X = rng.sample(names, rng.randint(1, min(3, k)))
    Y = rng.sample(names, rng.randint(1, min(3, k)))
    Z = rng.sample(names, rng.randint(0, min(3, k)))
    return X, Y, Z


def overlap_oracle(S, X, Y, Z):
    """Spec: a node of X that is in Y and not in Z => False; X nodes in Z block immediately;
    reaching a node of Y minus Z means not separated."""
    Zs = set(Z)
    X2 = [x for x in X if x not in Zs]
    Y2 = [y for y in Y if y not in Zs]
    if set(X2) & set(Y2):
        return False
    if not X2 or not Y2:
        return True
    return oracle_m(S, X2, Y2, Z)


def test_legacy_overlapping_args():
    for rng, S in graphs(21, n=400):
        G = S.build()
        Gd = S.without_bi().build()
        U = pu.unproject(G)
        for _ in range(TRIPLES):
            X, Y, Z = overlapping(rng, S.names)
            dX, dY, dZ = dicts(X), dicts(Y), dicts(Z)
            a = TestSepUnprojected(G, dX, dY, dZ)
            assert a is TestSep(U, dX, dY, dZ)
            assert TestSep(G, dX, dY, dZ) is TestSep(Gd, dX, dY, dZ)
            assert a is overlap_oracle(S, X, Y, Z), (S.names, S.edges, X, Y, Z)
            assert TestSep(G, dX, dY, dZ) is overlap_oracle(S.without_bi(), X, Y, Z)


def test_legacy_inputs_not_mutated():
    for rng, S in graphs(22, n=100):
        G = S.build()
        X, Y, Z = random_triple(rng, S.names)
        dX, dY, dZ = dicts(X), dicts(Y), dicts(Z)
        n0 = [dict(n) for n in G.nodes]
        e0 = [dict(e) for e in G.edges]
        for f in (TestSep, TestSepUnprojected, DSeparation.test):
            f(G, dX, dY, dZ)
        assert dX == dicts(X) and dY == dicts(Y) and dZ == dicts(Z)
        assert [dict(n) for n in G.nodes] == n0
        assert [dict(e) for e in G.edges] == e0


def test_name_collisions_unproject():
    # both edges unproject to a latent called U_{ABC}
    S1 = Spec(['A', 'BC', 'AB', 'C', 'W'], [('A', 'BC', 'bidirected'), ('AB', 'C', 'bidirected'),
                                              ('A', 'W', 'directed'), ('C', 'W', 'directed')])
    # real node literally named like a latent
    S2 = Spec(['A', 'B', 'U_{AB}', 'C'], [('A', 'B', 'bidirected'), ('U_{AB}', 'C', 'directed'),
                                           ('B', 'C', 'directed')])
    S3 = Spec(['A', 'B', 'U_{AB}', 'C'], [('A', 'B', 'bidirected'), ('U_{AB}', 'C', 'directed'),
                                           ('C', 'B', 'directed')])
    for S in (S1, S2, S3):
        G = S.build()
        U = pu.unproject(G)
        rng = random.Random(5)
        for _ in range(60):
            X, Y, Z = overlapping(rng, S.names)
            dX, dY, dZ = dicts(X), dicts(Y), dicts(Z)
            assert TestSepUnprojected(G, dX, dY, dZ) is TestSep(U, dX, dY, dZ), (S.names, X, Y, Z)
        import itertools
        for X in S.names:
            for Y in S.names:
                for r in range(0, 3):
                    for Z in itertools.combinations([n for n in S.names if n not in (X, Y)], r):
                        dX, dY, dZ = dicts([X]), dicts([Y]), dicts(list(Z))
                        assert TestSepUnprojected(G, dX, dY, dZ) is TestSep(U, dX, dY, dZ), (S.names, X, Y, Z)


def test_undirected_graph_testsep_still_runs_like_legacy():
    # smoke: moralized graph must not be routed to the name API (documented as unsupported there)
    S = Spec(['A', 'B', 'C'], [('A', 'B', 'directed'), ('B', 'C', 'directed')])
    G = S.build()
    from src.inference.utils.graph_utils import GraphUtils as gu
    M = gu.moralize(G)
    with pytest.raises(Exception):
        M.names.m_separated('A', 'C', [])
