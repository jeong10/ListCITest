"""List-based set operations used all over fusion_python.

Phase 2 rewrite: every operation has a native, hash-based fast path that reproduces pydash's
results exactly (same elements, same order, same duplicate handling).  The pydash calls are kept
only as the fallback for inputs outside the fast-path domain (non-list arguments, items that are
not dicts when an iteratee is given, values that cannot be keyed such as NaN or arbitrary objects,
iteratees other than ``None`` / a plain string key).  The previous implementation is kept,
commented out, at the bottom of this file as reference.

pydash semantics reproduced (7.0.6 and 8.1.0 share these helpers):

* ``difference(a, b)``    every item of ``a`` (duplicates kept, order of ``a``) whose key is not a key of ``b``
* ``intersection(a, b)``  ``[]`` if either list is empty; else ``a`` de-duplicated by key (first occurrence
                          wins) restricted to keys present in ``b``
* ``union(a, b)``         ``a + b`` de-duplicated by key (first occurrence wins)
* ``unique(a)``           ``a`` de-duplicated, first occurrence wins
* equality is plain ``==`` (``pydash.is_equal`` without customizer); with iteratee ``'name'`` the key
  of a dict is ``item.get('name')`` (``None`` when missing)
"""
import pydash


class _Fallback(Exception):
    """Raised inside the fast paths when the inputs are outside the domain they reproduce exactly."""


def _hkey(v):
    """Canonical hashable stand-in for ``v`` such that ``_hkey(a) == _hkey(b)`` iff ``a == b``."""
    t = type(v)

    if t is str or t is int or t is bool or t is bytes or v is None:
        return v

    if t is float:
        if v != v:  # NaN is not equal to itself; hashing it would change the semantics
            raise _Fallback()

        return v

    if t is dict:
        return ('d', frozenset([(k, _hkey(x)) for k, x in v.items()]))

    if t is list:
        return ('l', tuple([_hkey(x) for x in v]))

    if t is tuple:
        return ('t', tuple([_hkey(x) for x in v]))

    raise _Fallback()


def _keys(lst, iteratee):
    """Hashable keys of the items of ``lst``; ``iteratee`` is ``None`` (whole item) or a plain string key."""
    if iteratee is None:
        src = lst
    else:
        # plain dicts only: subclasses may define __missing__ / get differently from pydash's lookup
        src = []

        for x in lst:
            if type(x) is not dict:
                raise _Fallback()

            src.append(x.get(iteratee))

    return [k if type(k) is str else _hkey(k) for k in src]


def _plain_key(iteratee):
    """True for ``None`` and for string keys pydash resolves with one plain dict lookup (no path syntax;
    numeric strings are excluded because pydash retries them as ints)."""
    if iteratee is None:
        return True

    if type(iteratee) is not str or iteratee == '' or '.' in iteratee or '[' in iteratee:
        return False

    try:
        int(iteratee)
    except ValueError:
        return True

    return False


def _both_lists(a, b, iteratee):
    return type(a) is list and type(b) is list and _plain_key(iteratee)


class SetUtils():

    @staticmethod
    def union(a, b, iteratee=None):
        if _both_lists(a, b, iteratee):
            try:
                items = a + b
                seen = set()
                out = []

                for item, k in zip(items, _keys(items, iteratee)):
                    if k not in seen:
                        seen.add(k)
                        out.append(item)

                return out
            except _Fallback:
                pass

        if iteratee is None:
            return pydash.arrays.union(a, b)
        else:
            return pydash.arrays.union_by(a, b, iteratee=iteratee)

    @staticmethod
    def difference(a, b, iteratee=None):
        if _both_lists(a, b, iteratee):
            try:
                if not b:
                    return a[:]

                bkeys = set(_keys(b, iteratee))

                return [item for item, k in zip(a, _keys(a, iteratee)) if k not in bkeys]
            except _Fallback:
                pass

        if iteratee is None:
            return pydash.arrays.difference(a, b)
        else:
            return pydash.arrays.difference_by(a, b, iteratee=iteratee)

    @staticmethod
    def intersection(a, b, iteratee=None):
        if _both_lists(a, b, iteratee):
            try:
                if not a or not b:
                    return []

                bkeys = set(_keys(b, iteratee))
                seen = set()
                out = []

                for item, k in zip(a, _keys(a, iteratee)):
                    if k in seen:
                        continue

                    seen.add(k)

                    if k in bkeys:
                        out.append(item)

                return out
            except _Fallback:
                pass

        if iteratee is None:
            return pydash.arrays.intersection(a, b)
        else:
            return pydash.arrays.intersection_by(a, b, iteratee=iteratee)

    @staticmethod
    def belongs(a, b, comparator=None):
        if comparator is not None:
            for item in b:
                if comparator(a, item):
                    return True

            return False
        elif isinstance(a, list):
            for item in b:
                if a == item:
                    return True

            return False
        elif type(b) is list:
            return a in b
        else:
            return pydash.collections.includes(b, a)

    @staticmethod
    def isEmpty(arr):
        t = type(arr)

        if t is list or t is dict or t is tuple or t is str or arr is None:
            return not arr

        return pydash.predicates.is_empty(arr)

    @staticmethod
    def isSubset(sub, set, iteratee=None):
        if _both_lists(sub, set, iteratee):
            try:
                if not sub:
                    return True

                skeys = frozenset(_keys(set, iteratee))

                for k in _keys(sub, iteratee):
                    if k not in skeys:
                        return False

                return True
            except _Fallback:
                pass

        return len(SetUtils.difference(sub, set, iteratee)) == 0

    @staticmethod
    def equals(a, b, iteratee=None):
        if a is None and b is None:
            return True

        if _both_lists(a, b, iteratee):
            try:
                return set(_keys(a, iteratee)) == set(_keys(b, iteratee))
            except _Fallback:
                pass

        return SetUtils.isSubset(a, b, iteratee) and SetUtils.isSubset(b, a, iteratee)

    @staticmethod
    def isEqual(a, b):
        if a is None and b is None:
            return True

        return a == b

    @staticmethod
    def unique(arr):
        if type(arr) is list:
            try:
                seen = set()
                out = []

                for item, k in zip(arr, _keys(arr, None)):
                    if k not in seen:
                        seen.add(k)
                        out.append(item)

                return out
            except _Fallback:
                pass

        return pydash.arrays.uniq(arr)

    @staticmethod
    def uniqWith(array, comparator=None):
        if comparator is None:
            return SetUtils.unique(array)

        if type(array) is list:
            out = []

            for item in array:
                for seen in out:
                    if comparator(item, seen):
                        break
                else:
                    out.append(item)

            return out

        return pydash.arrays.uniq_with(array, comparator)

    # alias used by ListCITest (ListCI, ListGMP)
    uniqueWith = uniqWith


# --- OLD (pre-Phase-2, reference only; the pydash calls above remain live as the fallback) ---
# class SetUtils():
#
#     @staticmethod
#     def union(a, b, iteratee=None):
#         if iteratee is None:
#             return pydash.arrays.union(a, b)
#         else:
#             return pydash.arrays.union_by(a, b, iteratee=iteratee)
#
#     @staticmethod
#     def difference(a, b, iteratee=None):
#         if iteratee is None:
#             return pydash.arrays.difference(a, b)
#         else:
#             return pydash.arrays.difference_by(a, b, iteratee=iteratee)
#
#     @staticmethod
#     def intersection(a, b, iteratee=None):
#         if iteratee is None:
#             return pydash.arrays.intersection(a, b)
#         else:
#             return pydash.arrays.intersection_by(a, b, iteratee=iteratee)
#
#     @staticmethod
#     def belongs(a, b, comparator=None):
#         if comparator is not None:
#             for item in b:
#                 if comparator(a, item):
#                     return True
#             return False
#         elif isinstance(a, list):
#             for item in b:
#                 if SetUtils.isEqual(a, item):
#                     return True
#             return False
#         else:
#             return pydash.collections.includes(b, a)
#
#     @staticmethod
#     def isEmpty(arr):
#         return pydash.predicates.is_empty(arr)
#
#     @staticmethod
#     def isSubset(sub, set, iteratee=None):
#         return len(SetUtils.difference(sub, set, iteratee)) == 0
#
#     @staticmethod
#     def equals(a, b, iteratee=None):
#         if a is None and b is None:
#             return True
#         return SetUtils.isSubset(a, b, iteratee) and SetUtils.isSubset(b, a, iteratee)
#
#     @staticmethod
#     def isEqual(a, b):
#         if a is None and b is None:
#             return True
#         return pydash.predicates.is_equal(a, b)
#
#     @staticmethod
#     def unique(arr):
#         return pydash.arrays.uniq(arr)
#
#     @staticmethod
#     def uniqWith(array, comparator=None):
#         return pydash.arrays.uniq_with(array, comparator)
# --- END OLD ---
