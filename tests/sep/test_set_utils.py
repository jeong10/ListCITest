"""Differential tests of SetUtils against a pydash reference (written from the spec)."""
import copy
import math
import random

import pydash

from src.inference.utils.set_utils import SetUtils as su


# ---------------------------------------------------------------- reference
class Ref(object):
    @staticmethod
    def union(a, b, it=None):
        if it:
            return pydash.arrays.union_by(a, b, iteratee=it)
        return pydash.arrays.union(a, b)

    @staticmethod
    def difference(a, b, it=None):
        if it:
            return pydash.arrays.difference_by(a, b, iteratee=it)
        return pydash.arrays.difference(a, b)

    @staticmethod
    def intersection(a, b, it=None):
        if it:
            return pydash.arrays.intersection_by(a, b, iteratee=it)
        return pydash.arrays.intersection(a, b)

    @staticmethod
    def belongs(a, b, cmp=None):
        if cmp:
            return any(cmp(a, i) for i in b)
        if isinstance(a, list):
            return any(pydash.predicates.is_equal(a, i) for i in b)
        return pydash.collections.includes(b, a)

    @staticmethod
    def isEmpty(a):
        return pydash.predicates.is_empty(a)

    @staticmethod
    def isSubset(s, t, it=None):
        return len(Ref.difference(s, t, it)) == 0

    @staticmethod
    def equals(a, b, it=None):
        if a is None and b is None:
            return True
        return Ref.isSubset(a, b, it) and Ref.isSubset(b, a, it)

    @staticmethod
    def isEqual(a, b):
        if a is None and b is None:
            return True
        return pydash.predicates.is_equal(a, b)

    @staticmethod
    def unique(a):
        return pydash.arrays.uniq(a)

    @staticmethod
    def uniqWith(a, cmp=None):
        return pydash.arrays.uniq_with(a, cmp)


def same(a, b):
    """Strict structural equality (type-sensitive, NaN == NaN)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return list(a.keys()) == list(b.keys()) and all(same(a[k], b[k]) for k in a)
    return a == b


def run(f, *args):
    try:
        return ('ok', f(*args))
    except Exception as e:  # noqa
        return ('err', type(e))


def check(name, args):
    snap = copy.deepcopy(args)
    got = run(getattr(su, name), *args)
    exp = run(getattr(Ref, name), *copy.deepcopy(args))
    assert got[0] == exp[0], (name, args, got, exp)
    if got[0] == 'err':
        assert got[1] is exp[1], (name, args, got, exp)
    else:
        assert same(got[1], exp[1]), (name, args, got, exp)
    assert same(snap, args), ('mutated input', name, args, snap)


NAME_FUNCS = ['union', 'difference', 'intersection', 'isSubset', 'equals']
NAN = float('nan')


# ---------------------------------------------------------------- generators
def node(rng, pool):
    d = {'name': rng.choice(pool), 'label': rng.choice(['a', 'b', 'c']),
         'type_': rng.choice(['basic', 'latent']), 'metadata': {'x': rng.randint(0, 2)}}
    r = rng.random()
    if r < 0.08:
        del d['name']
    elif r < 0.12:
        d['name'] = None
    elif r < 0.16:
        d['name'] = rng.choice([1, 2.0, True])
    return d


def node_list(rng, pool):
    return [node(rng, pool) for _ in range(rng.randint(0, 7))]


def misc_item(rng):
    return rng.choice([1, 0, 1.0, True, False, None, 'a', 'b', '', NAN, (1, 2), [1], [1, 2], [[1]],
                       {'a': 1}, {'name': 'a'}, {'name': 'b', 'z': 1}, {'a': {'b': 1}}, {'a': {'b': 2}},
                       2, 2.5, 'A'])


def misc_list(rng):
    return [misc_item(rng) for _ in range(rng.randint(0, 7))]


def bad_arg(rng):
    return rng.choice([None, 'abc', {'a': 1}, 5, (1, 2, 1), 'a'])


ITERATEES = [None, 'name', 'label', 'a.b', 'a', 'type_', 'missing']


def test_random_nodes_with_iteratee():
    rng = random.Random(1)
    pool = ['A', 'B', 'C', 'D', 'E']
    for _ in range(2500):
        a, b = node_list(rng, pool), node_list(rng, pool)
        it = rng.choice(['name', 'name', None, 'label'])
        for f in NAME_FUNCS:
            check(f, (a, b, it))
        check('equals', (a, b))
        check('unique', (a,))
        check('isEmpty', (a,))
        check('isEqual', (a, b))
        check('isEqual', (a, copy.deepcopy(a)))
        check('uniqWith', (a,))
        check('belongs', (rng.choice(a) if a else node(rng, pool), b))
        check('belongs', (node(rng, pool), b, lambda x, y: x.get('name') == y.get('name')))


def test_random_misc_items():
    rng = random.Random(2)
    for _ in range(2500):
        a, b = misc_list(rng), misc_list(rng)
        it = rng.choice(ITERATEES)
        for f in NAME_FUNCS:
            check(f, (a, b, it))
            check(f, (a, b))
        check('unique', (a,))
        check('isEmpty', (a,))
        check('isEqual', (a, b))
        check('uniqWith', (a,))
        check('uniqWith', (a, lambda x, y: str(x) == str(y)))
        check('belongs', (misc_item(rng), b))
        check('belongs', (misc_list(rng), b))
        check('belongs', (misc_item(rng), b, lambda x, y: x == y))


def test_random_bad_args():
    rng = random.Random(3)
    for _ in range(1500):
        a = rng.choice([bad_arg(rng), misc_list(rng), node_list(rng, ['A', 'B'])])
        b = rng.choice([bad_arg(rng), misc_list(rng), node_list(rng, ['A', 'B'])])
        it = rng.choice(ITERATEES)
        for f in NAME_FUNCS:
            check(f, (a, b, it))
        check('equals', (a, b))
        check('unique', (a,))
        check('isEmpty', (a,))
        check('isEqual', (a, b))
        check('uniqWith', (a,))
        check('belongs', (a, b))


def test_nested_iteratee_dicts():
    rng = random.Random(4)
    for _ in range(800):
        def mk():
            return [{'a': {'b': rng.randint(0, 2)}, 'name': rng.choice('xy')} for _ in range(rng.randint(0, 5))]
        a, b = mk(), mk()
        for it in ('a.b', 'a', 'name', 'zz'):
            for f in NAME_FUNCS:
                check(f, (a, b, it))


# ---------------------------------------------------------------- hand written
def N(n, **kw):
    d = {'name': n, 'label': n, 'type_': 'basic', 'metadata': {}}
    d.update(kw)
    return d


def test_hand_cases():
    A, B, C = N('A'), N('B'), N('C')
    A2 = N('A', label='other')
    cases = [
        ([], [], 'name'), ([], [A], 'name'), ([A], [], 'name'), ([A, A, B], [B], 'name'),
        ([A, A2], [A], 'name'), ([A, B, C], [C, B, A], 'name'), ([A, B], [A2], None),
        ([{'label': 'x'}], [{'label': 'y'}], 'name'), ([{}], [{}], 'name'),
        ([1, 2, 3], [2], None), (['a', 'b'], ['b', 'a', 'a'], None), ([True, 1, 1.0], [1], None),
        ([NAN], [NAN], None), ([NAN, NAN], [], None), ([None], [None], None),
        ([[1], [1], [2]], [[1]], None), ([(1, 2)], [[1, 2]], None), ([{'a': 1}], [{'a': 1}], None),
        ([A, B], [B, A], 'label'), ([A, B], [B, A], 'a.b'),
    ]
    for a, b, it in cases:
        for f in NAME_FUNCS:
            check(f, (a, b, it))
        check('equals', (a, b))
        check('unique', (a,))
        check('isEmpty', (a,))
        check('isEqual', (a, b))
    for x in (None, 'str', {}, {'a': 1}, [], 0, 5):
        check('isEmpty', (x,))
        check('isEqual', (x, None))
        check('isEqual', (None, x))
    check('isEqual', (None, None))
    check('equals', (None, None))
    check('equals', (None, []))
    check('belongs', ('a', ['a']))
    check('belongs', ('a', 'abc'))
    check('belongs', (None, None))
    check('belongs', ([1], [[1]]))


def test_alias_and_returns_new_list():
    a = [1, 2]
    assert su.union(a, []) is not a
    assert su.difference(a, []) is not a
    assert su.unique(a) is not a
    rng = random.Random(7)
    cmp = lambda x, y: type(x) is type(y)
    for _ in range(300):
        a = misc_list(rng)
        assert same(su.uniqueWith(a), Ref.uniqWith(a))
        assert same(su.uniqueWith(a, cmp), Ref.uniqWith(a, cmp))
