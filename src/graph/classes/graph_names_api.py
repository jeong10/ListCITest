"""Name-based public API of ``Graph``, exposed as ``G.names``.

Every method takes node *names* (``str``) or iterables of names and returns
names (``str``) or plain tuples, never dicts.  Lists of names come back in
graph order (node insertion order), which makes results deterministic.

Unlike the legacy dict-based ``Graph`` methods, these methods:

* never rebuild node/edge dicts, they run on a bitset core and are memoized;
* raise ``KeyError`` for unknown node names (the legacy API ignores them);
* ``subgraph`` / ``do`` / ``ancestral`` return cheap *views*: new ``Graph``
  objects that share the parent's structure snapshot and are only copied when
  somebody mutates them or reads their edges through the legacy API.
  Node/edge ``metadata`` dicts are shared by reference with the parent (as with the
  legacy ``Graph.copy()``), but label/type_ edits and replaced edges never reach an
  existing view (attribute dicts are replaced copy-on-write).

Not available on undirected (moralized) graphs.
"""
from __future__ import annotations

from src.graph.core._core import DIRECTED, BIDIRECTED, iter_bits
from src.graph.classes.graph_defs import EdgeType, basicNodeType, directedEdgeType


def _as_names(x):
    if isinstance(x, str):
        return [x]
    return list(x)


class GraphNames(object):

    def __init__(self, graph):
        self._g = graph

    # ---- helpers ---------------------------------------------------------

    def _view(self):
        return self._g._query_view()

    def _mask(self, x, v):
        core = v.core
        m = 0
        for nm in _as_names(x):
            i = core.index.get(nm)
            if i is None or not (v.active >> i) & 1:
                raise KeyError('Unknown node: %r' % (nm,))
            m |= 1 << i
        return m

    @staticmethod
    def _type_id(edge_type):
        if edge_type is None:
            return None
        if isinstance(edge_type, EdgeType):
            return edge_type.id_
        return edge_type

    # ---- inspection --------------------------------------------------------

    def nodes(self):
        v = self._view()
        return v.core.decode(v.active)

    def edges(self):
        """List of ``(from, to, type)`` in legacy ``G.edges`` order."""
        g = self._g
        g._ensure_store()
        return [(u, w, a.get('type_', directedEdgeType.id_)) for u, w, a in g._edge_items()]

    def has_node(self, name):
        v = self._view()
        i = v.core.index.get(name)
        return i is not None and bool((v.active >> i) & 1)

    def has_edge(self, from_, to_, type_=None):
        return self._g.has_edge(from_, to_, type_)

    def edges_of(self, name):
        """Edges incident to ``name`` as ``(from, to, type)`` tuples (KeyError for an unknown node)."""
        if not self.has_node(name):
            raise KeyError('Unknown node: %r' % (name,))
        return [e for e in self.edges() if name in (e[0], e[1])]

    def node(self, name):
        """Node record ``{'name','label','type_','metadata'}`` (metadata is the live dict)."""
        v = self._view()
        i = v.core.index.get(name)
        if i is None or not (v.active >> i) & 1:
            raise KeyError('Unknown node: %r' % (name,))
        return self._g._node_dict(name, v.core.attrs[i])

    def label(self, name):
        return self.node(name)['label']

    def nodes_of_type(self, type_id):
        if isinstance(type_id, EdgeType) or hasattr(type_id, 'id_'):
            type_id = type_id.id_
        v = self._view()
        c = v.core
        return [c.names[i] for i in iter_bits(v.active)
                if c.attrs[i].get('type_', basicNodeType.id_) == type_id]

    # ---- relatives -----------------------------------------------------------

    def parents(self, x):
        v = self._view()
        return v.core.decode(v.parents(self._mask(x, v)))

    def children(self, x):
        v = self._view()
        return v.core.decode(v.children(self._mask(x, v)))

    def spouses(self, x):
        """Bidirected neighbors, excluding the query nodes themselves."""
        v = self._view()
        return v.core.decode(v.neighbors(self._mask(x, v), BIDIRECTED))

    def neighbors(self, x, edge_type=None):
        """Neighbors over edges of ``edge_type`` (any type if None), excluding the query nodes."""
        v = self._view()
        return v.core.decode(v.neighbors(self._mask(x, v), self._type_id(edge_type)))

    def ancestors(self, x, include_self=False):
        """Nodes with a directed path into ``x``.

        With ``include_self=False`` every node of ``x`` is excluded from the
        result (also when it is an ancestor of another node of ``x``).
        """
        v = self._view()
        m = self._mask(x, v)
        res = v.ancestors(m)
        return v.core.decode(res if include_self else res & ~m)

    def descendants(self, x, include_self=False):
        v = self._view()
        m = self._mask(x, v)
        res = v.descendants(m)
        return v.core.decode(res if include_self else res & ~m)

    def c_component(self, x):
        """Union of the bidirected-connected components containing the nodes of ``x``."""
        v = self._view()
        return v.core.decode(v.reach(self._mask(x, v), BIDIRECTED, 'both'))

    def c_components(self):
        """All c-components, ordered by their first node (graph order)."""
        v = self._view()
        return [v.core.decode(m) for m in v.c_components()]

    def topological_order(self):
        """Topological order of the directed part; ties broken by graph order."""
        v = self._view()
        names = v.core.names
        return [names[i] for i in v.topo_order()]

    # ---- separation (Bayes ball) ---------------------------------------------------

    def m_separated(self, x, y, z=()):
        """True if ``x`` and ``y`` are m-separated given ``z`` (semi-Markovian graphs: ``->`` and ``<->``).

        Bidirected edges act as hidden common parents; no latent node is materialized.  For a graph
        without ``<->`` edges this is ordinary d-separation.  ``x`` / ``y`` / ``z``: a name or an
        iterable of names (KeyError for unknown names).  Nodes of ``x`` that are in ``y`` but not in
        ``z`` make the answer False.
        """
        v = self._view()
        return v.bayes_ball_admg(self._mask(x, v), self._mask(y, v), self._mask(z, v) if len(_as_names(z)) else 0)

    def d_separated(self, x, y, z=()):
        """True if ``x`` and ``y`` are d-separated given ``z`` (non-Markovian graphs: only ``->`` edges;
        unobserved confounders are explicit latent nodes).

        Raises ``ValueError`` if the graph has ``<->`` edges (use ``m_separated``, or unproject first).
        """
        v = self._view()

        if any(v.nbr(BIDIRECTED)):
            raise ValueError('The graph has bidirected edges: use m_separated(), or replace them by latent nodes first.')

        return v.bayes_ball_dag(self._mask(x, v), self._mask(y, v), self._mask(z, v) if len(_as_names(z)) else 0)

    # ---- derived graphs (views) -------------------------------------------------

    def subgraph(self, x):
        """Induced subgraph G[x] as a view-backed Graph."""
        v = self._view()
        return self._g._from_view(v.sub(self._mask(x, v)))

    def ancestral(self, x):
        """Subgraph induced by ``x`` and its ancestors."""
        v = self._view()
        return self._g._from_view(v.sub(v.ancestors(self._mask(x, v))))

    def do(self, over=(), under=()):
        """Graph with edges into ``over`` removed (and bidirected edges touching it)
        and directed edges out of ``under`` removed (legacy ``GraphUtils.transform``)."""
        v = self._view()
        o = self._mask(over, v) if len(_as_names(over)) else 0
        u = self._mask(under, v) if len(_as_names(under)) else 0
        if v.core.simple:
            return self._g._from_view(v.do(o, u))
        return self._g._from_materialized(*v.do(o, u).materialize())

    def moralize(self):
        from src.inference.utils.graph_utils import GraphUtils as gu
        return gu.moralize(self._g)

    # ---- editing -------------------------------------------------------------------

    def add_node(self, name, label=None, type_=basicNodeType.id_, metadata=None):
        self._g.addNodes({'name': name,
                          'label': name if label is None else label,
                          'type_': type_,
                          'metadata': {} if metadata is None else metadata})

    def set_node(self, name, label=None, type_=None, metadata=None):
        """Edit a node record (None leaves a field unchanged).

        The attribute dict is replaced, not updated in place, so views created earlier keep
        the old label/type_ (in-place edits of the ``metadata`` dict itself are shared).
        """
        g = self._g
        g._ensure_store()
        if not g._has_node(name):
            raise KeyError('Unknown node: %r' % (name,))
        attrs = dict(g._node_attrs(name))
        if label is not None:
            attrs['label'] = label
        if type_ is not None:
            attrs['type_'] = type_
        if metadata is not None:
            attrs['metadata'] = metadata
        g._replace_node_attrs(name, attrs)

    def remove_node(self, name):
        if not self.has_node(name):
            raise KeyError('Unknown node: %r' % (name,))
        self._g.deleteNodes({'name': name})

    def add_edge(self, from_, to_, type_=directedEdgeType.id_, label=None, metadata=None,
                 check_cycle=True):
        """Add an edge between two existing nodes.

        Raises ``ValueError`` when a directed edge would close a directed cycle
        (disable with ``check_cycle=False``).  Collisions are handled like the
        legacy ``addEdges`` (X<->Y plus X->Y flips the stored bidirected edge).
        """
        type_ = self._type_id(type_)   # accept an EdgeType as well as its id
        g = self._g
        g._ensure_store()
        for nm in (from_, to_):
            if not g._has_node(nm):
                raise KeyError('Unknown node: %r' % (nm,))
        if check_cycle and type_ == DIRECTED and g._directed_path(to_, from_):
            raise ValueError('Adding %s -> %s creates a directed cycle.' % (from_, to_))
        g.addEdges({'from_': from_, 'to_': to_, 'type_': type_, 'label': label,
                    'metadata': {} if metadata is None else metadata})

    def remove_edge(self, from_, to_):
        """Remove an edge. Like the legacy ``deleteEdges`` this raises networkx's ``NetworkXError``
        for a missing edge (unless ``Graph.ignore_missing_edge_deletion`` is set)."""
        self._g.deleteEdges({'from_': from_, 'to_': to_})
