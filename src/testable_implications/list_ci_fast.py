"""Bitset translation of ``ConditionalIndependencies.ListCI`` (Algorithm 2 of the ListCI paper).

``list_ci_fast(G, V)`` returns exactly what the list-of-dicts implementation in
``conditional_independencies.py`` returns (same CIs, same order, same order inside ``W`` / ``Z``), or
``None`` when the input is outside the domain it reproduces exactly; the caller then runs the legacy code.

Everything works on int masks over the hidden graph core (node = core index = graph order).  All
sets of the algorithm (``I``, ``R``, ``C``, ``Z``, ``W``, ``S+`` ...) are masks; the only places where
list ORDER is observable are reproduced explicitly:

* components / parents / descendants lists are in graph order (ascending bit),
* ``gu.spouses(w)`` is the per-node union ``sp(w1) | sp(w2) ...`` in first-seen order (and keeps nodes of ``w``
  that are spouses of other nodes of ``w``),
* ``W`` (= S+ minus Z minus X) is in the consistent ordering ``V``,
* ``Z`` (= mb+) is the members of ``C`` (graph order, X removed) followed by the other parents of ``C`` (graph order),
* ``dCandidates`` = the base set in spouse order, then the remaining descendants in graph order, and the first
  candidate whose separator exists decides ``FindAAC``.

Domain: directed-mode ``Graph``, only ``->`` / ``<->`` edges, no cuts (``do`` views), node names unique after
``strip().upper()``, no latent-name collisions of ``ProjectionUtils.unproject`` (the separator test is the
semi-Markovian Bayes ball, equal to ``TestSep`` on the unprojected graph), every node of ``V`` is a distinct node of ``G``.
Anything unexpected during the run (e.g. a ``None`` component where the legacy code would continue with ``None``)
raises ``_Fallback`` and the whole computation is redone by the legacy code.

Must stay Python 3.7 compatible.
"""
from src.graph.core._core import BIDIRECTED, iter_bits, _or_over, _closure


class _Fallback(Exception):
    pass


def list_ci_fast(G, V):
    """CI list equal to the legacy ``ListCI`` loop over ``V`` (a list of node dicts), or ``None`` (not applicable)."""
    if getattr(G, '_ug', 1) is not None:
        return None

    try:
        view = G._query_view()
    except (TypeError, AttributeError):
        return None

    core = view.core

    if view.over or view.under or not core.simple or not core.names_unique_ci() or not core.unproject_safe():
        return None

    index = core.index
    act0 = view.active
    order = []
    seen = 0

    try:
        for node in V:
            i = index.get(node['name'])

            if i is None or not (act0 >> i) & 1 or (seen >> i) & 1:
                return None

            seen |= 1 << i
            order.append(i)
    except (TypeError, KeyError, IndexError):
        return None

    try:
        return _Run(G, core, V, order).run()
    except _Fallback:
        return None


class _Run(object):

    def __init__(self, G, core, V, order):
        self.G = G
        self.core = core
        self.V = V
        self.order = order
        self.pa = core.pa
        self.ch = core.ch
        self.sp = core.nb.get(BIDIRECTED, core.zero)
        self.pos = [0] * core.n

        for p, i in enumerate(order):
            self.pos[i] = p

        self.CI = []
        self.bugs = 0
        # per-X caches (the active mask changes with X)
        self.act = 0
        self.xbit = 0
        self.xi = 0
        self.cache_an = {}
        self.cache_de = {}
        self.cache_pa = {}
        self.cache_sp = {}
        self.cache_split = {}
        self.nodes = {}        # core index -> node dict shared by the CIs of this call (equal by value to fresh decodes)

    # ---- set algebra on the current induced subgraph (active mask ``act``) -----------------------

    def an(self, mask):
        """Ancestors of ``mask`` including ``mask`` (``ancestorsPlus``)."""
        res = self.cache_an.get(mask)

        if res is None:
            res = self.cache_an[mask] = _closure(self.pa, mask, self.act)

        return res

    def de(self, mask):
        """Descendants of ``mask`` including ``mask``."""
        res = self.cache_de.get(mask)

        if res is None:
            res = self.cache_de[mask] = _closure(self.ch, mask, self.act)

        return res

    def parents(self, mask):
        res = self.cache_pa.get(mask)

        if res is None:
            res = self.cache_pa[mask] = _or_over(self.pa, mask) & self.act

        return res

    def spouses(self, mask):
        """Union of the bidirected neighbours of every node of ``mask`` (nodes of ``mask`` may be included)."""
        res = self.cache_sp.get(mask)

        if res is None:
            res = self.cache_sp[mask] = _or_over(self.sp, mask) & self.act

        return res

    def comp(self, seed, allowed):
        """c-component of ``seed`` inside ``allowed`` (``C(G, X)``); the legacy code yields ``None`` if X is missing."""
        if not (allowed & self.xbit):
            raise _Fallback()

        return _closure(self.sp, seed, allowed)

    def spouse_order(self, mask):
        """``gu.spouses`` as an ordered list of indices: per node of ``mask`` (graph order), its spouses (graph order), first seen wins."""
        out = []
        seen = 0
        sp = self.sp
        act = self.act

        for i in iter_bits(mask):
            new = sp[i] & act & ~seen

            if new:
                seen |= new
                out.extend(iter_bits(new))

        return out

    # ---- Algorithm 2 ----------------------------------------------------------------------------

    def run(self):
        core = self.core
        order = self.order
        prefix = 0

        for j, xi in enumerate(order):
            prefix |= 1 << xi
            self.act = prefix
            self.xi = xi
            self.xbit = 1 << xi
            self.cache_an = {}
            self.cache_de = {}
            self.cache_pa = {}
            self.cache_sp = {}
            self.cache_split = {}

            AnX = self.an(self.xbit)
            I = self.comp(self.xbit, AnX)
            R = self.comp(self.xbit, prefix)

            self.listcix(self.V[j], I, R)

        for _ in range(self.bugs):
            print('Bug: T cannot be empty!')

        return self.CI

    def node(self, i):
        d = self.nodes.get(i)

        if d is None:
            core = self.core
            d = self.nodes[i] = self.G._node_dict(core.names[i], core.attrs[i])

        return d

    def split(self, C):
        """``(Z, Splus)`` masks for an ancestral c-component ``C``: ``Z = mb+(C)`` and ``S+``."""
        res = self.cache_split.get(C)

        if res is None:
            PaC = C | self.parents(C)
            Sp = self.spouses(C) & ~PaC
            Splus = self.act & ~(Sp | self.de(Sp))
            res = self.cache_split[C] = (PaC & ~self.xbit, Splus)

        return res

    def is_admissible(self, C):
        Z, Splus = self.split(C)

        return (Splus & ~(Z | self.xbit)) != 0

    def find_aac(self, I, R):
        """True iff the legacy ``FindAAC`` returns something other than ``None``."""
        if self.is_admissible(I):
            return True

        PaR = R | self.parents(R)
        PaI = I | self.parents(I)
        base = [i for i in self.spouse_order(I) if not (PaI >> i) & 1]
        base_mask = 0

        for i in base:
            base_mask |= 1 << i

        rest = self.de(base_mask) & ~base_mask if base_mask else 0
        xbit = self.xbit
        act = self.act

        for d in base + list(iter_bits(rest)):
            dbit = 1 << d
            XY = xbit | dbit
            Z = (PaR & ~XY) & self.an(XY | PaI)          # FindSeparator: Z = (R' \ XY) cap An+(XY u I)

            if self.separated(dbit, Z):
                # FindAAC returns C(G[An+(I u Z)], X): None if X is not in that subgraph
                return bool(self.an(I | Z) & xbit)

        return False

    def separated(self, Y, Z):
        """Semi-Markovian Bayes ball (``TestSep`` on the unprojected graph): is ``Y`` separated from X given ``Z``?"""
        act = self.act
        pa = self.pa
        ch = self.ch
        sp = self.sp
        target = Y & ~Z
        zan = Z | self.an(Z) if Z else 0
        up_vis = 0
        down_vis = 0
        up_new = self.xbit
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
            up_new = _or_over(pa, send_up) & act
            down_new = (_or_over(ch, up_free | down_free) | _or_over(sp, send_up)) & act

    def listcix(self, X, I, R):
        if not self.find_aac(I, R):
            return

        xbit = self.xbit

        if I == R:
            C = I
            Zmask, Splus = self.split(C)
            Zlist = [i for i in iter_bits(C & ~xbit)] + [i for i in iter_bits(self.parents(C) & ~C & ~xbit)]
            Wmask = Splus & ~(Zmask | xbit)
            pos = self.pos
            Wlist = sorted(iter_bits(Wmask), key=pos.__getitem__)
            V = self.V

            # legacy: W = S+ \ (Z u X) with S+ built from VleqX, i.e. W holds the caller's own V dicts
            # (identity and content, e.g. a user supplied Vordered); Z is decoded from the graph
            self.CI.append({
                'X': X,
                'W': [V[pos[i]] for i in Wlist],
                'Z': [self.node(i) for i in Zlist]
            })

            return

        SpI = self.spouses(I) & ~I
        T = R & SpI

        if not T:
            self.bugs += 1

            return

        s = (T & -T).bit_length() - 1          # T[0]: first node of R (graph order) that is a spouse of I
        sbit = 1 << s
        Rprime = self.comp(xbit, R & ~self.de(sbit))
        Iprime = self.comp(xbit, self.an(I | sbit))

        self.listcix(X, I, Rprime)
        self.listcix(X, Iprime, R)
