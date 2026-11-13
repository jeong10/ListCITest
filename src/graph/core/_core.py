"""Hidden bitset core for ``src.graph.classes.graph.Graph``.

Nothing in here is public API: it is an implementation detail of ``Graph``.

Design (see REFACTOR_PLAN.md, Step 1a):

* ``_Core`` is an immutable *snapshot* of a graph's structure.  Node ``i`` is
  the ``i``-th node in insertion order, so iterating the set bits of a mask in
  ascending order yields nodes in ``G.nodes`` order (the order contract).
  Adjacency is kept as Python ints used as bitsets (``pa``, ``ch`` for directed
  edges, ``nb[type]`` for symmetric adjacency of every edge type).
* ``_View`` is a restriction of a core: an ``active`` node mask plus the two
  ``do``-style cuts ``over`` (remove edges into these nodes, and every
  bidirected edge touching them) and ``under`` (remove directed edges out of
  these nodes).  Views are O(1)-ish to create and never copy edges.
* Results of structural queries are memoized on the core, keyed by the view
  and the query mask (masks are plain ints, hence hashable).

Must stay Python 3.7 compatible (no ``int.bit_count``, no walrus, ...).
"""
from __future__ import annotations

import copy
import heapq

DIRECTED = 'directed'
BIDIRECTED = 'bidirected'

_CACHE_LIMIT = 100000


def iter_bits(mask):
    """Yield indices of set bits of ``mask`` in ascending order."""
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def bit_count(mask):
    return bin(mask).count('1')


def correct_name(name):
    """Same normalisation as GraphUtils.correctNodeName (case-insensitive lookups)."""
    if not name:
        return ''
    return name.strip().upper()


def _or_over(adj, mask):
    """OR together ``adj[i]`` for every set bit ``i`` of ``mask``."""
    res = 0
    while mask:
        low = mask & -mask
        res |= adj[low.bit_length() - 1]
        mask ^= low
    return res


def _closure(adj, seed, allowed):
    """Nodes reachable from ``seed`` (inclusive) following ``adj`` inside ``allowed``."""
    seen = seed & allowed
    frontier = seen
    while frontier:
        nxt = _or_over(adj, frontier) & allowed & ~seen
        seen |= nxt
        frontier = nxt
    return seen


class _Core(object):
    """Snapshot of a graph's structure (immutable: Graph replaces node/edge attribute dicts
    copy-on-write, so a snapshot never observes later changes; only ``metadata`` objects are
    shared by reference, as with ``nx.copy()``).

    ``names``  list of node names in insertion order.
    ``attrs``  list of the (live, shared) node attribute dicts, aligned with names.
    ``edges``  list of ``(u_idx, v_idx, edge_attr_dict)`` in DiGraph edge order.
    """

    def __init__(self, names, attrs, edges):
        self.names = names
        self.attrs = attrs
        self.edges = edges
        n = len(names)
        self.n = n
        self.full = (1 << n) - 1
        self.index = dict((nm, i) for i, nm in enumerate(names))
        pa = [0] * n
        ch = [0] * n
        nb = {}
        simple = True
        for u, v, a in edges:
            t = a.get('type_', DIRECTED)
            bu = 1 << u
            bv = 1 << v
            if t == DIRECTED:
                ch[u] |= bv
                pa[v] |= bu
            elif t != BIDIRECTED:
                simple = False
            lst = nb.get(t)
            if lst is None:
                lst = nb[t] = [0] * n
            lst[u] |= bv
            lst[v] |= bu
        self.pa = pa
        self.ch = ch
        self.nb = nb
        self.simple = simple
        self.zero = [0] * n
        self.cache = {}
        self.full_view = _View(self)

    def with_attr(self, i, attr):
        """A core for the same structure whose node ``i`` has the attribute dict ``attr``.

        Shares the adjacency arrays and the memo cache (queries never depend on attributes);
        the original core (and views over it) keep the old attribute dict.
        """
        new = copy.copy(self)
        new.attrs = list(self.attrs)
        new.attrs[i] = attr
        new.full_view = _View(new)
        return new

    def unproject_safe(self):
        """True if replacing every ``<->`` edge by an explicit latent parent named like
        ``ProjectionUtils.unproject`` does would be collision free: no two ``<->`` edges map to the
        same latent name and no latent name equals a real node name.  Only then is
        ``bayes_ball_admg`` guaranteed to equal ``bayes_ball_dag(unproject(G))``.  Cached per core.
        """
        res = getattr(self, '_unproject_safe', None)

        if res is None:
            # legacy lookups are case-insensitive and strip whitespace (correct_name), so latent names
            # are compared under that normalisation, against each other and against the real nodes
            seen = set()
            res = True
            names = self.names
            real = set(correct_name(nm) for nm in names)

            for u, v, a in self.edges:
                if a.get('type_', DIRECTED) != BIDIRECTED:
                    continue

                f, t = names[u], names[v]
                lo, hi = (f, t) if f < t else (t, f)
                latent = correct_name(''.join(['U_{', lo, hi, '}']))

                if latent in seen or latent in real:
                    res = False
                    break

                seen.add(latent)

            self._unproject_safe = res

        return res

    def names_unique_ci(self):
        """True if no two node names are equal after ``correct_name`` (strip + upper-case).

        The legacy separation code finds nodes through the case-insensitive ``getNodeByName`` (first
        match), so the exact-name fast paths are only equivalent when this holds.  Cached per core.
        """
        res = getattr(self, '_names_unique_ci', None)

        if res is None:
            res = len(set(correct_name(nm) for nm in self.names)) == len(self.names)
            self._names_unique_ci = res

        return res

    def mask(self, names):
        """Mask of the given names (KeyError for unknown names)."""
        m = 0
        index = self.index
        for nm in names:
            m |= 1 << index[nm]
        return m

    def decode(self, mask):
        names = self.names
        return [names[i] for i in iter_bits(mask)]


class _View(object):
    """A restriction of a ``_Core`` (see module docstring)."""

    def __init__(self, core, active=None, over=0, under=0):
        self.core = core
        self.active = core.full if active is None else (active & core.full)
        self.over = over & self.active
        self.under = under & self.active
        self.key = (self.active, self.over, self.under)
        self.is_full = (self.active == core.full and not self.over and not self.under)
        self._pa = None
        self._ch = None
        self._sp = None
        self._nb = {}
        self._all = None
        self._cn = None

    # ---- derived views -------------------------------------------------

    def sub(self, mask):
        return _View(self.core, self.active & mask, self.over, self.under)

    def do(self, over_mask, under_mask):
        return _View(self.core, self.active, self.over | over_mask, self.under | under_mask)

    # ---- effective adjacency -------------------------------------------

    def _eff(self, base, kind):
        c = self.core
        act = self.active
        over = self.over
        under = self.under
        res = [0] * c.n
        for i in iter_bits(act):
            bit = 1 << i
            if kind == 'pa':
                if over & bit:
                    continue
                res[i] = base[i] & ~under & act
            elif kind == 'ch':
                if under & bit:
                    continue
                res[i] = base[i] & ~over & act
            else:  # symmetric (bidirected) adjacency
                if over & bit:
                    continue
                res[i] = base[i] & ~over & act
        return res

    @property
    def pa(self):
        if self._pa is None:
            self._pa = self.core.pa if self.is_full else self._eff(self.core.pa, 'pa')
        return self._pa

    @property
    def ch(self):
        if self._ch is None:
            self._ch = self.core.ch if self.is_full else self._eff(self.core.ch, 'ch')
        return self._ch

    def nbr(self, t):
        """Symmetric adjacency (list of masks) of edge type ``t``."""
        res = self._nb.get(t)
        if res is not None:
            return res
        c = self.core
        base = c.nb.get(t, c.zero)
        if self.is_full:
            res = base
        elif t == DIRECTED:
            res = [p | q for p, q in zip(self.pa, self.ch)]
        elif t == BIDIRECTED:
            res = self._eff(base, 'sp')
        else:
            # only reachable for views without do-cuts (see Graph.names.do)
            act = self.active
            res = [(base[i] & act) if (act >> i) & 1 else 0 for i in range(c.n)]
        self._nb[t] = res
        return res

    @property
    def any_nbr(self):
        """Symmetric adjacency over all edge types."""
        if self._all is None:
            c = self.core
            res = [0] * c.n
            for t in c.nb:
                lst = self.nbr(t)
                for i in range(c.n):
                    res[i] |= lst[i]
            self._all = res
        return self._all

    # ---- memoization ----------------------------------------------------

    def _memo(self, op, mask, fn):
        cache = self.core.cache
        key = (self.key, op, mask)
        res = cache.get(key)
        if res is None:
            res = fn()
            if len(cache) >= _CACHE_LIMIT:
                cache.clear()
            cache[key] = res
        return res

    # ---- queries on masks (all results are masks) ------------------------

    def parents(self, mask):
        mask &= self.active
        return self._memo('pa', mask, lambda: _or_over(self.pa, mask))

    def children(self, mask):
        mask &= self.active
        return self._memo('ch', mask, lambda: _or_over(self.ch, mask))

    def neighbors(self, mask, t=None):
        """Neighbors of ``mask`` (edge type ``t`` or any), excluding ``mask`` itself."""
        mask &= self.active
        adj = self.any_nbr if t is None else self.nbr(t)
        return self._memo(('nb', t), mask, lambda: _or_over(adj, mask) & ~mask)

    def ancestors(self, mask):
        """Closure of ``mask`` under 'parent of' (includes ``mask``)."""
        mask &= self.active
        return self._memo('an', mask, lambda: _closure(self.pa, mask, self.active))

    def descendants(self, mask):
        """Closure of ``mask`` under 'child of' (includes ``mask``)."""
        mask &= self.active
        return self._memo('de', mask, lambda: _closure(self.ch, mask, self.active))

    def reach(self, mask, t, direction):
        """Generic reachability (inclusive) along type ``t`` (None = any type).

        ``direction`` is 'both', 'forward' or 'backward'; only meaningful for
        directed edges (``t == DIRECTED``); every other type is symmetric.
        """
        mask &= self.active
        if t == DIRECTED and direction == 'forward':
            adj = self.ch
        elif t == DIRECTED and direction == 'backward':
            adj = self.pa
        else:
            adj = self.any_nbr if t is None else self.nbr(t)
        return self._memo(('reach', t, direction), mask, lambda: _closure(adj, mask, self.active))

    def c_component_of(self, idx):
        """Bidirected-connected component containing node ``idx`` (a mask)."""
        bit = 1 << idx
        return self.reach(bit, BIDIRECTED, 'both')

    def c_components(self):
        """All c-components as masks, ordered by their smallest node index."""
        def compute():
            comps = []
            remaining = self.active
            while remaining:
                low = remaining & -remaining
                comp = self.reach(low, BIDIRECTED, 'both')
                comps.append(comp)
                remaining &= ~comp
            return tuple(comps)
        return self._memo('cc', 0, compute)

    def topo_order(self):
        """Topological order of the active nodes (list of indices).

        Deterministic: among available nodes the smallest index goes first.
        Raises ValueError if the directed part contains a cycle.
        """
        def compute():
            act = self.active
            pa = self.pa
            ch = self.ch
            indeg = {}
            heap = []
            for i in iter_bits(act):
                d = bit_count(pa[i] & act)
                indeg[i] = d
                if d == 0:
                    heap.append(i)
            heapq.heapify(heap)
            order = []
            while heap:
                i = heapq.heappop(heap)
                order.append(i)
                for j in iter_bits(ch[i] & act):
                    indeg[j] -= 1
                    if indeg[j] == 0:
                        heapq.heappush(heap, j)
            if len(order) != len(indeg):
                raise ValueError('The directed part of the graph contains a cycle.')
            return tuple(order)
        return self._memo('topo', 0, compute)

    # ---- separation (Bayes ball) -------------------------------------------
    #
    # Both versions answer "is Y separated from X given Z?" with the state machine of the legacy
    # ``TestSep`` (balls start at X going *up*; a ball that visits a node of Y that is not in Z means
    # "connected"; an observed node blocks 'up' balls and lets 'down' balls turn around).  A state is
    # (node, up|down); visited states are two bitsets and a whole frontier is expanded per round, so
    # the cost is O(rounds * frontier) big-int operations.  ``X``, ``Y``, ``Z`` are masks.

    def bayes_ball_dag(self, X, Y, Z):
        """Non-Markovian version: only directed edges are traversed.

        Unobserved confounders must be *explicit* (latent) nodes; ``<->`` edges are ignored.
        Equals the legacy ``TestSep`` on the same graph.
        """
        act = self.active
        X &= act
        Z &= act
        target = Y & act & ~Z
        pa = self.pa
        ch = self.ch
        zan = Z | self.ancestors(Z)           # observed or ancestor of an observed node
        up_vis = 0
        down_vis = 0
        up_new = X
        down_new = 0

        while True:
            up_new &= ~up_vis
            down_new &= ~down_vis

            if not (up_new or down_new):
                return True

            up_vis |= up_new
            down_vis |= down_new

            if (up_new | down_new) & target:
                return False

            up_free = up_new & ~Z             # unobserved: ball passes on to parents (up) and children (down)
            down_free = down_new & ~Z
            send_up = up_free | (down_new & zan)    # a 'down' ball turns around at observed colliders/ancestors
            up_new = _or_over(pa, send_up)
            down_new = _or_over(ch, up_free | down_free)

    def bayes_ball_admg(self, X, Y, Z):
        """Semi-Markovian version: directed (``->``) and bidirected (``<->``) edges, no latent nodes.

        A ``<->`` edge has an arrowhead at both ends, i.e. it behaves like a hidden common parent:
        wherever the ball is sent up to the parents of ``n`` it is also sent *down* to the spouses of
        ``n``.  Equals ``bayes_ball_dag`` on the graph in which every ``<->`` edge is replaced by an
        explicit latent parent (``ProjectionUtils.unproject``), without building that graph.
        """
        act = self.active
        X &= act
        Z &= act
        target = Y & act & ~Z
        pa = self.pa
        ch = self.ch
        sp = self.nbr(BIDIRECTED)
        zan = Z | self.ancestors(Z)
        up_vis = 0
        down_vis = 0
        up_new = X
        down_new = 0

        while True:
            up_new &= ~up_vis
            down_new &= ~down_vis

            if not (up_new or down_new):
                return True

            up_vis |= up_new
            down_vis |= down_new

            if (up_new | down_new) & target:
                return False

            up_free = up_new & ~Z
            down_free = down_new & ~Z
            send_up = up_free | (down_new & zan)
            up_new = _or_over(pa, send_up)
            down_new = _or_over(ch, up_free | down_free) | _or_over(sp, send_up)

    # ---- lookups ---------------------------------------------------------

    def iter_edges(self):
        """``(u_idx, v_idx, attrs)`` of this view in legacy edge order, without copying."""
        c = self.core
        if self.is_full:
            return c.edges
        act = self.active
        over = self.over
        under = self.under
        out = []
        for u, v, a in c.edges:
            bu = 1 << u
            bv = 1 << v
            if not (act & bu and act & bv):
                continue
            if over or under:
                t = a.get('type_', DIRECTED)
                if over & bv:
                    continue
                if t == DIRECTED and under & bu:
                    continue
                if t == BIDIRECTED and over & bu:
                    continue
            out.append((u, v, a))
        return out

    def cname_maps(self):
        """(first, last): corrected (stripped, upper-cased) name -> node index of the first /
        last active node with that corrected name."""
        if self._cn is None:
            first = {}
            last = {}
            names = self.core.names
            for i in iter_bits(self.active):
                cn = correct_name(names[i])
                first.setdefault(cn, i)
                last[cn] = i
            self._cn = (first, last)
        return self._cn

    # ---- materialization ------------------------------------------------

    def materialize(self):
        """(node_items, edge_items) with *copies* of attribute dicts.

        ``node_items``: list of (name, attr_dict); ``edge_items``: list of
        (from_name, to_name, attr_dict), both in the parent's order.  Edge
        removal follows the legacy ``GraphUtils.transform`` rules.
        """
        c = self.core
        act = self.active
        over = self.over
        under = self.under
        names = c.names
        node_items = [(names[i], dict(c.attrs[i])) for i in iter_bits(act)]
        edge_items = []
        for u, v, a in c.edges:
            bu = 1 << u
            bv = 1 << v
            if not (act & bu and act & bv):
                continue
            if over or under:
                t = a.get('type_', DIRECTED)
                if over & bv:
                    continue
                if t == DIRECTED and under & bu:
                    continue
                if t == BIDIRECTED and over & bu:
                    continue
            edge_items.append((names[u], names[v], dict(a)))
        return node_items, edge_items
