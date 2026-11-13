from __future__ import annotations

import json
import random
from typing import List, Dict, Any

import networkx as nx

from src.graph.classes.graph_defs import basicNodeType, directedEdgeType, bidirectedEdgeType, EdgeType, edgeTypeMap
from src.graph.core._core import _Core, _View, iter_bits, correct_name
from src.graph.classes.graph_names_api import GraphNames
from src.task.classes.task import Task
from src.task.basic_task import BasicTask

from src.common.object_utils import ObjectUtils as ou

# Graph is backed by one of three stores (see REFACTOR_PLAN.md, Step 1):
#   * a directed store (insertion-ordered dicts that mirror nx.DiGraph's
#     iteration order) + a lazily built immutable bitset core for queries;
#   * a real nx.Graph, used only after toUndirected() (moral graphs; rare);
#   * a *view* (parent core + masks) that is materialized into a directed
#     store the first time somebody mutates it or reads its edges.
# The legacy dict-based API (nodes/edges/addEdges/parents/...) keeps its exact
# old semantics; the name-based API lives in ``G.names`` (graph_names_api.py).


class _DiStore(object):
    """Directed store mirroring the iteration order semantics of ``nx.DiGraph``."""

    # Attribute dicts are replaced copy-on-write (never updated in place): cores and views
    # created earlier keep referencing the old dicts, so they never see later changes
    # (only the ``metadata`` objects themselves are shared by reference, as with nx.copy()).

    def __init__(self):
        self.nodes = {}   # name -> attr dict (may lack label/type_/metadata if imported from nx)
        self.succ = {}    # name -> {name -> shared edge attr dict}
        self.pred = {}
        self.cn = None    # cached case-insensitive name maps, see Graph._cname_maps

    def add_node(self, name, attrs):
        self.cn = None

        if name in self.nodes:
            nd = dict(self.nodes[name])
            nd.update(attrs)
            self.nodes[name] = nd
            return False

        self.nodes[name] = dict(attrs)
        self.succ[name] = {}
        self.pred[name] = {}
        return True

    def ensure_node(self, name):
        if name not in self.nodes:
            self.cn = None
            # implicit nodes get the defaults G.nodes reports, stored once so that writes to their
            # metadata persist (also in subgraph views, as in the legacy copy-based subgraph)
            self.nodes[name] = {'label': name, 'type_': basicNodeType.id_, 'metadata': {}}
            self.succ[name] = {}
            self.pred[name] = {}

    def add_edge(self, u, v, attrs):
        self.ensure_node(u)
        self.ensure_node(v)
        d = dict(self.succ[u].get(v) or {})
        d.update(attrs)
        self.succ[u][v] = d
        self.pred[v][u] = d

    def has_edge(self, u, v):
        nb = self.succ.get(u)
        return nb is not None and v in nb

    def remove_edge(self, u, v):
        del self.succ[u][v]
        del self.pred[v][u]

    def remove_incident_edges(self, name):
        for v in list(self.succ[name]):
            self.pred[v].pop(name, None)
        for u in list(self.pred[name]):
            self.succ[u].pop(name, None)
        self.succ[name] = {}
        self.pred[name] = {}

    def remove_node(self, name):
        self.cn = None

        for v in list(self.succ[name]):
            self.pred[v].pop(name, None)
        for u in list(self.pred[name]):
            self.succ[u].pop(name, None)
        del self.succ[name]
        del self.pred[name]
        del self.nodes[name]

    def copy(self):
        s = _DiStore()
        for n, a in self.nodes.items():
            s.nodes[n] = dict(a)
            s.succ[n] = {}
            s.pred[n] = {}
        for u, nb in self.succ.items():
            for v, d in nb.items():
                nd = dict(d)
                s.succ[u][v] = nd
                s.pred[v][u] = nd
        return s


class Graph():

    name = str
    task = Task
    metadata = Dict[str, Any]

    # False: deleting a missing edge raises (fusion_python behaviour);
    # True: it is ignored (ListCITest behaviour).
    # ListCITest: deleting a missing edge is ignored (projection code relies on it)
    ignore_missing_edge_deletion = True

    def __init__(self, graph = None, nodes = [], edges = [], task = None, metadata = dict()):
        self._s = None        # _DiStore (directed mode)
        self._ug = None       # nx.Graph (undirected mode, after toUndirected)
        self._view = None     # _View (lazy view over a parent's core)
        self._sv = 0          # structure version (nodes/edges added or removed)
        self._av = 0          # attribute version (label/type_/metadata edits)
        self._core = None
        self._core_sv = -1
        self._names_api = None

        if graph is not None:
            if isinstance(graph, Graph):
                # known diff #2 (CHANGES-graph-rewrite.md): the legacy code aliased the nx object; we copy.
                other = graph.copy()
                self._s, self._ug, self._view = other._s, other._ug, other._view
            elif isinstance(graph, nx.Graph):
                self._import_nx(graph)
            else:
                raise TypeError('graph must be a Graph or a networkx graph')
        else:
            self._s = _DiStore()
            self.addNodes(nodes)
            self.addEdges(edges)

        if task is None:
            task = BasicTask()

        self.task = task
        # NOTE: the default ``metadata`` dict is shared by all graphs created
        # without one. Parsers rely on this (graph.metadata['populations']).
        self.metadata = metadata

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #

    def _import_nx(self, g):
        if g.is_directed():
            s = _DiStore()
            for n, a in g.nodes(data=True):
                s.nodes[n] = dict(a)
                s.succ[n] = {}
                s.pred[n] = {}
            for u, v, d in g.edges(data=True):
                nd = dict(d)
                s.succ[u][v] = nd
                s.pred[v][u] = nd
            self._s = s
        else:
            self._ug = g

    @classmethod
    def _from_view(cls, view):
        g = cls()
        g._s = None
        g._view = view
        return g

    @classmethod
    def _from_materialized(cls, node_items, edge_items):
        g = cls()
        s = g._s
        for name, attrs in node_items:
            s.nodes[name] = attrs
            s.succ[name] = {}
            s.pred[name] = {}
        for u, v, attrs in edge_items:
            s.succ[u][v] = attrs
            s.pred[v][u] = attrs
        return g

    @classmethod
    def from_json(cls, obj, **kwargs):
        """Build a Graph from the web front end's schema (dict or JSON string):
        ``{'nodes': [{name, label, type, metadata}], 'edges': [{from, to, type, label, metadata}]}``."""
        if isinstance(obj, (str, bytes)):
            obj = json.loads(obj)
        nodes = []
        for n in obj.get('nodes', []):
            nodes.append({'name': n['name'],
                          'label': n.get('label', n['name']),
                          'type_': n.get('type', n.get('type_', basicNodeType.id_)),
                          'metadata': n.get('metadata') or {}})
        edges = []
        for e in obj.get('edges', []):
            edges.append({'from_': e['from'] if 'from' in e else e['from_'],
                          'to_': e['to'] if 'to' in e else e['to_'],
                          'type_': e.get('type', e.get('type_', directedEdgeType.id_)),
                          'label': e.get('label'),
                          'metadata': e.get('metadata') or {}})
        return cls(nodes=nodes, edges=edges, **kwargs)

    def to_json(self):
        """Inverse of ``from_json`` (task/graph-level metadata are not serialized)."""
        return {
            'nodes': [{'name': n['name'], 'label': n['label'], 'type': n['type_'],
                       'metadata': n['metadata']} for n in self.nodes],
            'edges': [{'from': e['from_'], 'to': e['to_'], 'type': e['type_'],
                       'label': e['label'], 'metadata': e['metadata']} for e in self.edges],
        }

    @property
    def names(self):
        """Name-based API, see ``graph_names_api.GraphNames``."""
        if self._names_api is None:
            self._names_api = GraphNames(self)
        return self._names_api

    @property
    def struct_version(self):
        return self._sv

    @property
    def attr_version(self):
        return self._av

    # ------------------------------------------------------------------ #
    # internal store access (three modes)
    # ------------------------------------------------------------------ #

    def _ensure_store(self):
        """Materialize a view into a directed store (no-op otherwise)."""
        if self._view is not None:
            node_items, edge_items = self._view.materialize()
            g = Graph._from_materialized(node_items, edge_items)
            self._s = g._s
            self._view = None
            self._sv += 1

    def _get_core(self):
        if self._view is not None:
            return self._view.core
        if self._core is None or self._core_sv != self._sv:
            s = self._s
            names = list(s.nodes)
            index = dict((nm, i) for i, nm in enumerate(names))
            attrs = [s.nodes[nm] for nm in names]
            edges = []
            for u, nb in s.succ.items():
                iu = index[u]
                for v, d in nb.items():
                    edges.append((iu, index[v], d))
            self._core = _Core(names, attrs, edges)
            self._core_sv = self._sv
        return self._core

    def _query_view(self):
        if self._ug is not None:
            raise TypeError('The name-based API is not available on undirected (moralized) graphs.')
        if self._view is not None:
            return self._view
        return self._get_core().full_view

    def _node_items(self):
        if self._ug is not None:
            return list(self._ug.nodes(data=True))
        if self._view is not None:
            c = self._view.core
            return [(c.names[i], c.attrs[i]) for i in iter_bits(self._view.active)]
        return list(self._s.nodes.items())

    def _edge_items(self):
        """List of (from, to, attrs) in the legacy edge order."""
        if self._ug is not None:
            return list(self._ug.edges(data=True))
        self._ensure_store()
        res = []
        for u, nb in self._s.succ.items():
            for v, d in nb.items():
                res.append((u, v, d))
        return res

    def _has_node(self, name):
        if self._ug is not None:
            return self._ug.has_node(name)
        if self._view is not None:
            i = self._view.core.index.get(name)
            return i is not None and bool((self._view.active >> i) & 1)
        return name in self._s.nodes

    def _edges_touching(self, names, incoming):
        """Edge dicts (fresh, legacy ``edges`` order) of the edges that ``GraphUtils.getIncoming`` (``incoming=True``:
        ``to_`` in names, or a ``<->`` edge with ``from_`` in names) / ``getOutgoing`` (``incoming=False``: non-``<->``
        edges with ``from_`` in names) select, in O(degree) instead of scanning all edges.
        ``None`` when not applicable (undirected graph, view with do-cuts): the caller scans ``self.edges``."""
        if self._ug is not None:
            return None
        if self._view is not None:
            v = self._view
            if v.over or v.under:
                return None
            core = v.core
            act = v.active
        else:
            core = self._get_core()
            act = core.full
        index = core.index
        to_pos, bid_from, nonbid_from = core.edge_index()
        cand = set()
        for nm in names:
            i = index.get(nm)
            if i is None:
                continue
            if incoming:
                cand.update(to_pos[i])
                cand.update(bid_from[i])
            else:
                cand.update(nonbid_from[i])
        cn = core.names
        edges = core.edges
        out = []
        for k in sorted(cand):
            u, v2, a = edges[k]
            if (act >> u) & 1 and (act >> v2) & 1:
                out.append(Graph._edge_dict(cn[u], cn[v2], a))
        return out

    def _node_attrs(self, name):
        if self._ug is not None:
            return self._ug.nodes[name]
        self._ensure_store()
        return self._s.nodes[name]

    def has_edge(self, from_, to_, type=None):
        """Exact (from, to) edge test (symmetric on undirected graphs); ``type`` optionally filters."""
        if self._ug is not None:
            if not self._ug.has_edge(from_, to_):
                return False
            return type is None or self._ug[from_][to_].get('type_', directedEdgeType.id_) == getattr(type, 'id_', type)
        self._ensure_store()
        d = self._s.succ.get(from_, {}).get(to_)
        if d is None:
            return False
        return type is None or d.get('type_', directedEdgeType.id_) == getattr(type, 'id_', type)

    def _directed_path(self, src, dst):
        """True if ``dst`` is reachable from ``src`` (inclusive) over directed edges."""
        self._ensure_store()
        succ = self._s.succ
        seen = {src}
        stack = [src]
        while stack:
            u = stack.pop()
            if u == dst:
                return True
            for v, d in succ[u].items():
                if d.get('type_', directedEdgeType.id_) == directedEdgeType.id_ and v not in seen:
                    seen.add(v)
                    stack.append(v)
        return False

    @staticmethod
    def _node_dict(name, a):
        return {
            'name': name,
            'label': a['label'] if 'label' in a else name,
            'type_': a['type_'] if 'type_' in a else basicNodeType.id_,
            'metadata': a['metadata'] if 'metadata' in a else {}
        }

    @staticmethod
    def _edge_dict(u, v, a):
        return {
            'from_': u,
            'to_': v,
            'label': a['label'] if 'label' in a else None,
            'type_': a['type_'] if 'type_' in a else directedEdgeType.id_,
            'metadata': a['metadata'] if 'metadata' in a else {}
        }

    def _find_edge(self, from_, to_):
        """Edge dict whose orientation is exactly (from_, to_), or None (legacy scan)."""
        if self._ug is None:
            self._ensure_store()
            d = self._s.succ.get(from_, {}).get(to_)
            return None if d is None else self._edge_dict(from_, to_, d)
        for u, v, a in self._ug.edges(data=True):
            if u == from_ and v == to_:
                return self._edge_dict(u, v, a)
        return None

    def _cname_maps(self):
        """(first, last, names, attrs): case-insensitive name -> index maps (first/last match)
        plus the aligned name / attribute lists, for GraphUtils.getNodeByName & friends."""
        if self._ug is not None:
            names = []
            attrs = []
            first = {}
            last = {}
            for i, (nm, a) in enumerate(self._ug.nodes(data=True)):
                names.append(nm)
                attrs.append(a)
                cn = correct_name(nm)
                first.setdefault(cn, i)
                last[cn] = i
            return first, last, names, attrs
        if self._view is None:
            s = self._s
            if s.cn is None:
                names = list(s.nodes)
                first = {}
                last = {}
                for i, nm in enumerate(names):
                    cn = correct_name(nm)
                    first.setdefault(cn, i)
                    last[cn] = i
                s.cn = (first, last, names, [s.nodes[nm] for nm in names])
            return s.cn

        v = self._view
        first, last = v.cname_maps()
        return first, last, v.core.names, v.core.attrs

    def _replace_node_attrs(self, name, attrs):
        """Replace a node's attribute dict copy-on-write (used by names.set_node)."""
        if self._ug is not None:
            self._ug.nodes[name].update(attrs)
            return

        self._ensure_store()
        self._s.nodes[name] = attrs
        self._s.cn = None
        self._av += 1
        self._refresh_core_attr(name)

    def _refresh_core_attr(self, name):
        """After a copy-on-write attribute replacement: hand the graph a core that points to the new
        dict (structure and memo cache are shared; views keep the old core)."""
        c = self._core

        if c is not None and self._core_sv == self._sv:
            self._core = c.with_attr(c.index[name], self._s.nodes[name])

    def _mask_of(self, nodes, core):
        """Mask of the nodes (dicts) that exist; unknown names are ignored (legacy behaviour)."""
        m = 0
        index = core.index
        for node in nodes:
            i = index.get(node['name'])
            if i is not None:
                m |= 1 << i
        return m

    def _decode(self, mask, core):
        names = core.names
        attrs = core.attrs
        return [self._node_dict(names[i], attrs[i]) for i in iter_bits(mask)]

    # ------------------------------------------------------------------ #
    # legacy API
    # ------------------------------------------------------------------ #

    @property
    def nodes(self):
        return [self._node_dict(name, a) for name, a in self._node_items()]

    @nodes.setter
    def nodes(self, nodes):
        # legacy: nx.clear() followed by addNodes
        if self._ug is not None:
            self._ug.clear()
        else:
            self._view = None
            self._s = _DiStore()
            self._sv += 1
        self.addNodes(nodes)

    @property
    def edges(self):
        return [self._edge_dict(u, v, a) for u, v, a in self._edge_items()]

    @edges.setter
    def edges(self, edges):
        self.deleteEdges(self.edges)
        self.addEdges(edges)

    @property
    def nx(self):
        """A networkx view of the graph.

        Directed graphs: a *fresh* ``nx.DiGraph`` is built on every access (changes
        to it are not reflected in this Graph). Undirected graphs: the live ``nx.Graph``.
        """
        if self._ug is not None:
            return self._ug
        self._ensure_store()
        g = nx.DiGraph()
        for n, a in self._s.nodes.items():
            g.add_node(n, **a)
        for u, nb in self._s.succ.items():
            for v, d in nb.items():
                g.add_edge(u, v, **d)
        return g

    @nx.setter
    def nx(self, g):
        self._s = self._ug = self._view = None
        self._import_nx(g)
        self._sv += 1

    def copy(self):
        if self._ug is not None:
            return Graph(self._ug.copy())
        if self._view is not None:
            return Graph._from_view(self._view)
        g = Graph()
        g._s = self._s.copy()
        return g

    def toUndirected(self):
        # legacy: self.nx = nx.Graph(self.nx), built on the same edge order
        self._ug = nx.Graph(self.nx)
        self._s = None
        self._view = None
        self._sv += 1

        # change all edge types

    # ------------------------------------------------------------------ #
    # ListCITest only: random graph generation.
    # The sequence of `random` calls is identical to the original implementation, so seeded
    # experiments stay reproducible. The original used nx.find_cycle(self.nx) after adding an
    # edge; since the graph is acyclic before each insertion that is equivalent to "the target
    # already reaches the source through the stored edges", checked natively (_stored_path).
    # ------------------------------------------------------------------ #

    def _stored_path(self, src, dst):
        """True if ``dst`` is reachable from ``src`` (inclusive) over the stored edges, whatever their type."""
        self._ensure_store()
        succ = self._s.succ
        seen = {src}
        stack = [src]

        while stack:
            u = stack.pop()

            if u == dst:
                return True

            for v in succ[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)

        return False

    def addRandomNodes(self, n, nodeType=basicNodeType.id_):
        nodes = []

        for i in range(n):
            node = {'name': str(i),'label': str(i),'type_': nodeType,'metadata': {}}
            nodes.append(node)

        self.addNodes(nodes)

        return nodes

    def _addRandomDirectedEdges(self, n, m):
        edges = []
        edgeCount = 0

        while (edgeCount < m):
            x = int(random.random() * n)
            y = int(random.random() * n)

            if x == y:
                continue

            edge = {'from_': str(x), 'to_': str(y), 'label': None, 'type_': directedEdgeType.id_, 'metadata': {}}

            if not self.has_edge(edge['from_'], edge['to_']):
                if not self._stored_path(edge['to_'], edge['from_']):
                    self.__addEdge(edge)
                    edgeCount = edgeCount + 1
                    edges.append(edge)
                    continue

            if not self.has_edge(edge['to_'], edge['from_']):
                edge = {'from_': str(y), 'to_': str(x), 'label': None, 'type_': directedEdgeType.id_, 'metadata': {}}

                if not self._stored_path(edge['to_'], edge['from_']):
                    self.__addEdge(edge)
                    edgeCount = edgeCount + 1
                    edges.append(edge)
                    continue

        return edges

    # possible improvements
    # adding directed edges: 1) pick a topological order, 2) add X -> Y where X < Y.
    # adding bidirected edges: 1) pick permutation, 2) add X -- Y (be careful not to overwrite X -> Y)
    def addRandomEdges(self, m, edgeType=directedEdgeType.id_, randomSeed=None):
        if m == 0:
            return

        if randomSeed is not None:
            random.seed(randomSeed)

        n = len(self.nodes)

        edges = []
        edgeCount = 0

        if edgeType == directedEdgeType.id_:
            edges = self._addRandomDirectedEdges(n, m)
        elif edgeType == bidirectedEdgeType.id_:
            while (edgeCount < m):
                x = int(random.random() * n)
                y = int(random.random() * n)

                if x == y:
                    continue

                edge = {'from_': str(x), 'to_': str(y), 'label': None, 'type_': bidirectedEdgeType.id_, 'metadata': {}}

                # if X -- Y or X -> Y exists
                #   check if X -- Y
                #    if true, skip
                #   else (X -> Y)
                #     check if bidirected edge Y -- X exists
                #       if true, skip
                #       else, add Y -- X
                # else
                #   check if Y -- X exists
                #     if yes, skip
                #     else (Y <- X), add X -- Y
                #   else (no edge exists)
                #     add X -- Y
                if self.has_edge(edge['from_'], edge['to_']):
                    existingEdge = self._find_edge(edge['from_'], edge['to_'])

                    if existingEdge['type_'] == bidirectedEdgeType.id_:
                        continue
                    else:
                        if self.has_edge(edge['to_'], edge['from_']):
                            continue
                        else:
                            edge = {'from_': str(y), 'to_': str(x), 'label': None, 'type_': bidirectedEdgeType.id_, 'metadata': {}}

                            self.__addEdge(edge)
                            edgeCount = edgeCount + 1
                            edges.append(edge)
                else:
                    if self.has_edge(edge['to_'], edge['from_']):
                        existingEdge = self._find_edge(edge['to_'], edge['from_'])

                        if existingEdge['type_'] == bidirectedEdgeType.id_:
                            continue
                        else:
                            self.__addEdge(edge)
                            edgeCount = edgeCount + 1
                            edges.append(edge)
                    else:
                        self.__addEdge(edge)
                        edgeCount = edgeCount + 1
                        edges.append(edge)

        return edges

    def toRandomGraph(self, n, m, edgeType=directedEdgeType.id_, randomSeed=None):
        nodes = []

        for i in range(n):
            node = {'name': str(i),'label': str(i),'type_': basicNodeType.id_,'metadata': {}}
            nodes.append(node)

        self.nodes = nodes

        edges = []
        edgeCount = 0

        if randomSeed is not None:
            random.seed(randomSeed)

        if edgeType == directedEdgeType.id_:
            edges = self._addRandomDirectedEdges(n, m)
        elif edgeType == bidirectedEdgeType.id_:
            while (edgeCount < m):
                x = int(random.random() * n)
                y = int(random.random() * n)

                if x == y:
                    continue

                edge = {'from_': str(x), 'to_': str(y), 'label': None, 'type_': bidirectedEdgeType.id_, 'metadata': {}}

                if not self.has_edge(edge['from_'], edge['to_']) and not self.has_edge(edge['to_'], edge['from_']):
                    self.__addEdge(edge)
                    edgeCount = edgeCount + 1
                    edges.append(edge)

        self.edges = edges

    def addNodes(self, nodes):
        nodes = ou.makeArray(nodes)

        if not nodes or len(nodes) == 0:
            return

        for node in nodes:
            # should we throw error if name is not specified?
            # duplicate name/label?

            if 'name' not in node:
                continue

            if 'label' not in node:
                node['label'] = node['name']

            if 'type_' not in node:
                node['type_'] = basicNodeType.id_

            if 'metadata' not in node:
                node['metadata'] = dict()

            self.__addNode(node)

    def deleteNodes(self, nodes):
        nodes = ou.makeArray(nodes)

        if not nodes or len(nodes) == 0:
            return

        # legacy order: first the edges of every listed node, then the nodes themselves
        # (matters for the state left behind when a missing node raises)
        if self._ug is None:
            self._ensure_store()

            for node in nodes:
                if node['name'] in self._s.nodes:
                    self._s.remove_incident_edges(node['name'])
                    self._sv += 1

        for node in nodes:
            self.__deleteNode(node)

    def addEdges(self, edges):
        edges = ou.makeArray(edges)

        if not edges or len(edges) == 0:
            return

        for edge in edges:
            if 'from_' not in edge or 'to_' not in edge:
                continue

            if 'label' not in edge:
                edge['label'] = None

            if 'type_' not in edge:
                edge['type_'] = directedEdgeType.id_

            if 'metadata' not in edge:
                edge['metadata'] = dict()

            # nx has a 'bug': if X -- Y exists, adding X -> Y will overwrite X -- Y, or vice versa
            if self.has_edge(edge['from_'], edge['to_']):
                existingEdge = self._find_edge(edge['from_'], edge['to_'])

                # if adding X -> Y, remove X -- Y and add Y -- X first
                if edge['type_'] == directedEdgeType.id_:
                    if existingEdge['type_'] == bidirectedEdgeType.id_:
                        self.deleteEdges(existingEdge)

                        temp = existingEdge['from_']
                        existingEdge['from_'] = existingEdge['to_']
                        existingEdge['to_'] = temp

                        self.addEdges(existingEdge)
                    else:
                        edgeType = directedEdgeType

                        for id_ in edgeTypeMap:
                            if edge['type_'].upper() == edgeTypeMap[id_].id_.upper():
                                edgeType = edgeTypeMap[id_]
                                break

                        raise Exception('The edge ' + edge['from_'] + ' ' + edgeType.shortId + ' ' + edge['to_'] + ' already exists.')

                # if adding X -- Y, flip from/to and add Y -- X instead
                elif edge['type_'] == bidirectedEdgeType.id_:
                    if existingEdge['type_'] == directedEdgeType.id_:
                        temp = edge['from_']
                        edge['from_'] = edge['to_']
                        edge['to_'] = temp
                    else:
                        edgeType = directedEdgeType

                        for id_ in edgeTypeMap:
                            if edge['type_'].upper() == edgeTypeMap[id_].id_.upper():
                                edgeType = edgeTypeMap[id_]
                                break

                        raise Exception('The edge ' + edge['from_'] + ' ' + edgeType.shortId + ' ' + edge['to_'] + ' already exists.')

            # NOTE: like the legacy code, no cycle check here (use G.names.add_edge for that)
            self.__addEdge(edge)

    def deleteEdges(self, edges):
        edges = ou.makeArray(edges)

        if not edges or len(edges) == 0:
            return

        # Like the legacy fusion_python code, deleting a missing edge raises NetworkXError:
        # algorithms rely on it (e.g. frontdoor GetDep/remoralize). ListCITest's variant
        # ignores missing edges; set ``Graph.ignore_missing_edge_deletion = True`` for that.
        for edge in edges:
            self.__deleteEdge(edge)

    # Node[]
    # Node[]
    def parents(self, nodes):
        nodes = ou.makeArray(nodes)

        if not nodes or len(nodes) == 0:
            return []

        if self._ug is None:
            v = self._query_view()
            return self._decode(v.parents(self._mask_of(nodes, v.core)), v.core)

        # undirected mode: legacy edge scan
        names = dict()
        parents = dict()

        for node in nodes:
            names[node['name']] = True

        for edge in self.edges:
            if edge['type_'] == directedEdgeType.id_ and edge['to_'] in names:
                parents[edge['from_']] = True

        return list(filter(lambda n: n['name'] in parents, self.nodes))

    def ancestors(self, nodes):
        # known diff #1/#3 (CHANGES-graph-rewrite.md): only directed edges are followed (legacy followed <-> as well),
        # and results come back in graph order instead of hash order.
        nodes = ou.makeArray(nodes)

        if not nodes or len(nodes) == 0:
            return []

        if self._ug is None:
            v = self._query_view()
            m = self._require_mask(nodes, v.core)
            return self._decode(v.ancestors(m) & ~m, v.core)

        names = set(n['name'] for n in nodes)
        an = set()
        for node in nodes:
            an.update(nx.ancestors(self._ug, node['name']))
        return [n for n in self.nodes if n['name'] in an and n['name'] not in names]

    def descendants(self, nodes):
        nodes = ou.makeArray(nodes)

        if not nodes or len(nodes) == 0:
            return []

        if self._ug is None:
            v = self._query_view()
            m = self._require_mask(nodes, v.core)
            return self._decode(v.descendants(m) & ~m, v.core)

        names = set(n['name'] for n in nodes)
        de = set()
        for node in nodes:
            de.update(nx.descendants(self._ug, node['name']))
        return [n for n in self.nodes if n['name'] in de and n['name'] not in names]

    def _require_mask(self, nodes, core):
        m = 0
        for node in nodes:
            i = core.index.get(node['name'])
            if i is None or not (self._query_view().active >> i) & 1:
                raise nx.NetworkXError('The node %s is not in the digraph.' % (node['name'],))
            m |= 1 << i
        return m

    def children(self, nodes):
        nodes = ou.makeArray(nodes)

        if self._ug is None:
            v = self._query_view()
            return self._decode(v.children(self._mask_of(nodes, v.core)), v.core)

        names = dict()
        children = dict()

        for node in nodes:
            names[node['name']] = True

        for edge in self.edges:
            if edge['type_'] == directedEdgeType.id_ and edge['from_'] in names:
                children[edge['to_']] = True

        return list(filter(lambda n: n['name'] in children, self.nodes))

    # Node | Node[], EdgeType | str
    # Node[]
    def neighbors(self, nodes, edgeType = None):
        nodes = ou.makeArray(nodes)

        if not nodes or len(nodes) == 0:
            return []

        if isinstance(edgeType, EdgeType):
            edgeType = edgeType.id_

        if self._ug is None:
            v = self._query_view()
            m = self._mask_of(nodes, v.core)
            return self._decode(v.neighbors(m, edgeType if edgeType else None), v.core)

        names = dict()
        neighbors = dict()

        for node in nodes:
            names[node['name']] = True

        # fast path (Phase 4): walk the adjacency of the queried nodes instead of scanning all edges (same set, same node order)
        adj = self._ug.adj
        want = edgeType if edgeType else None
        found = set()

        for nm in names:
            if nm in adj:
                for other, d in adj[nm].items():
                    if want is None or d.get('type_', directedEdgeType.id_) == want:
                        found.add(other)

        return [self._node_dict(n, a) for n, a in self._ug.nodes(data=True) if n in found and n not in names]

        # --- OLD (undirected neighbors by scanning all edges; same result) ---
        # edges = list(filter(lambda e: e['type_'] == edgeType if edgeType else True, self.edges))
        #
        # for edge in edges:
        #     if edge['to_'] in names:
        #         neighbors[edge['from_']] = True
        #     if edge['from_'] in names:
        #         neighbors[edge['to_']] = True
        #
        # return list(filter(lambda n: n['name'] in neighbors and n['name'] not in names, self.nodes))
        # --- END OLD ---

    def connectedComponents(self):
        # legacy: nx.connected_components, which only exists for undirected graphs
        # (raises NetworkXNotImplemented on a directed graph, as before)
        CC = list(nx.connected_components(self.nx))
        nodeByName = dict((n['name'], n) for n in self.nodes)
        ccsNodes = []

        for cc in CC:
            ccsNodes.append([nodeByName[nodeName] for nodeName in cc])

        return ccsNodes

    def print(self):
        if len(self.nodes) > 0:
            print('<Nodes (' + str(len(self.nodes)) + ')>')
        else:
            print('<Nodes (none)>')

        nodeNames = list(map(lambda n: n['name'], self.nodes))

        print('\n'.join(nodeNames))

        if len(self.edges) > 0:
            print('\n<Edges (' + str(len(self.edges)) + ')>')
        else:
            print('\n<Edges (none)>')

        edgeList = list(map(lambda e: e['from_'] + ' ' + edgeTypeMap[e['type_']].shortId  + ' ' + e['to_'], self.edges))

        print('\n'.join(edgeList))

    def __addNode(self, node):
        attrs = {'label': node['label'], 'type_': node['type_'], 'metadata': node['metadata']}

        if self._ug is not None:
            self._ug.add_node(node['name'], **attrs)
            return

        self._ensure_store()
        if self._s.add_node(node['name'], attrs):
            self._sv += 1
        else:
            self._av += 1
            self._refresh_core_attr(node['name'])

    def __deleteNode(self, node):
        name = node['name']

        if self._ug is not None:
            self._ug.remove_node(name)
            return

        self._ensure_store()
        if name not in self._s.nodes:
            raise nx.NetworkXError('The node %s is not in the digraph.' % (name,))
        self._s.remove_node(name)
        self._sv += 1

    def __addEdge(self, edge):
        attrs = {'label': edge['label'], 'type_': edge['type_'], 'metadata': edge['metadata']}

        if self._ug is not None:
            self._ug.add_edge(edge['from_'], edge['to_'], **attrs)
            return

        self._ensure_store()
        self._s.add_edge(edge['from_'], edge['to_'], attrs)
        self._sv += 1

    def __deleteEdge(self, edge):
        u = edge['from_']
        v = edge['to_']

        if self._ug is not None:
            if self._ug.has_edge(u, v) or not Graph.ignore_missing_edge_deletion:
                self._ug.remove_edge(u, v)   # raises NetworkXError if missing
            return

        self._ensure_store()
        if self._s.has_edge(u, v):
            self._s.remove_edge(u, v)
            self._sv += 1
        elif not Graph.ignore_missing_edge_deletion:
            raise nx.NetworkXError('The edge %s-%s not in graph.' % (u, v))


# --------------------------------------------------------------------------
# --- OLD (pre-graph-rewrite, reference only; executable copy: tag graph-rewrite-baseline) ---
# --------------------------------------------------------------------------
# import networkx as nx
# from typing import List, Dict, Any
#
# from src.graph.classes.graph_defs import basicNodeType, directedEdgeType, bidirectedEdgeType, EdgeType, edgeTypeMap
# from src.task.classes.task import Task
# from src.task.basic_task import BasicTask
#
# from src.inference.utils.graph_utils import GraphUtils as gu
# from src.inference.utils.set_utils import SetUtils as su
# from src.common.object_utils import ObjectUtils as ou
#
#
# class Graph():
#
#     name = str
#     task = Task
#     metadata = Dict[str, Any]
#
#     def __init__(self, graph = None, nodes = [], edges = [], task = None, metadata = dict()):
#         if graph is not None:
#             if isinstance(graph, nx.Graph) or isinstance(graph, nx.DiGraph):
#                 self.nx = graph
#             elif isinstance(graph, Graph):
#                 self.nx = graph.nx
#         else:
#             self.nx = nx.DiGraph()
#             self.addNodes(nodes)
#             self.addEdges(edges)
#
#         if task is None:
#             task = BasicTask()
#
#         self.task = task
#         self.metadata = metadata
#
#
#     @property
#     def nodes(self):
#         nodes = self.nx.nodes(data = True) 
#         ns = []
#
#         # convert tuple to list of dictionaries
#         for node in nodes:
#             n = {
#                 'name': node[0],
#                 'label': node[1]['label'] if 'label' in node[1] else node[0],
#                 'type_': node[1]['type_'] if 'type_' in node[1] else basicNodeType.id_,
#                 'metadata': node[1]['metadata'] if 'metadata' in node[1] else {}
#             }
#
#             ns.append(n)
#
#         return ns
#
#         # return self.nx.nodes(data = True)
#
#
#     @nodes.setter
#     def nodes(self, nodes):
#         self.nx.clear()
#         self.addNodes(nodes)
#
#
#     @property
#     def edges(self):
#         edges = self.nx.edges(data = True)
#         es = []
#
#         for edge in edges:
#             e = {
#                 'from_': edge[0],
#                 'to_': edge[1],
#                 'label': edge[2]['label'] if 'label' in edge[2] else None,
#                 'type_': edge[2]['type_'] if 'type_' in edge[2] else directedEdgeType.id_,
#                 'metadata': edge[2]['metadata'] if 'metadata' in edge[2] else {}
#             }
#
#             es.append(e)
#
#         return es
#
#         # return self.nx.edges(data = True)
#
#     @edges.setter
#     def edges(self, edges):
#         self.deleteEdges(self.edges)
#         self.addEdges(edges)
#
#
#     def copy(self):
#         return Graph(self.nx.copy())
#
#
#     def toUndirected(self):
#         # return self.nx.to_undirected()
#         self.nx = nx.Graph(self.nx)
#
#         # change all edge types
#
#
#     def addNodes(self, nodes):
#         nodes = ou.makeArray(nodes)
#
#         if not nodes or len(nodes) == 0:
#             return
#
#         for node in nodes:
#             # should we throw error if name is not specified?
#             # duplicate name/label?
#
#             # if not hasattr(node, 'name'):
#             if 'name' not in node:
#                 continue
#
#             # if not hasattr(node, 'label'):
#             if 'label' not in node:
#                 node['label'] = node['name']
#
#             # if not hasattr(node, 'type_'):
#             if 'type_' not in node:
#                 node['type_'] = basicNodeType.id_
#
#             # if not hasattr(node, 'metadata'):
#             if 'metadata' not in node:
#                 node['metadata'] = dict()
#
#             self.__addNode(node)
#
#
#     def deleteNodes(self, nodes):
#         nodes = ou.makeArray(nodes)
#
#         if not nodes or len(nodes) == 0:
#             return
#
#         self.deleteEdges(gu.getIncoming(nodes, self))
#         self.deleteEdges(gu.getOutgoing(nodes, self))
#
#         # can we delete all at once instead?
#         for node in nodes:
#             self.__deleteNode(node)
#
#
#     def addEdges(self, edges):
#         edges = ou.makeArray(edges)
#
#         if not edges or len(edges) == 0:
#             return
#
#         for edge in edges:
#             # if not hasattr(edge, 'from_') or not hasattr(edge, 'to_'):
#             if 'from_' not in edge or 'to_' not in edge:
#                 continue
#
#             # if not hasattr(edge, 'label'):
#             if 'label' not in edge:
#                 edge['label'] = None
#
#             # if not hasattr(edge, 'type_'):
#             if 'type_' not in edge:
#                 edge['type_'] = directedEdgeType.id_
#
#             # if not hasattr(edge, 'metadata'):
#             if 'metadata' not in edge:
#                 edge['metadata'] = dict()
#
#             # nx has a 'bug': if X -- Y exists, adding X -> Y will overwrite X -- Y, or vice versa
#             if self.nx.has_edge(edge['from_'], edge['to_']):
#                 existingEdge = None
#
#                 for e in self.edges:
#                     if e['from_'] == edge['from_'] and e['to_'] == edge['to_']:
#                         existingEdge = e
#                         break
#
#                 # if adding X -> Y, remove X -- Y and add Y -- X first
#                 if edge['type_'] == directedEdgeType.id_:
#                     if existingEdge['type_'] == bidirectedEdgeType.id_:
#                         self.deleteEdges(existingEdge)
#
#                         temp = existingEdge['from_']
#                         existingEdge['from_'] = existingEdge['to_']
#                         existingEdge['to_'] = temp
#
#                         self.addEdges(existingEdge)
#                     else:
#                         edgeType = directedEdgeType
#
#                         for id_ in edgeTypeMap:
#                             if edge['type_'].upper() == edgeTypeMap[id_].id_.upper():
#                                 edgeType = edgeTypeMap[id_]
#                                 break
#
#                         raise Exception('The edge ' + edge['from_'] + ' ' + edgeType.shortId + ' ' + edge['to_'] + ' already exists.')
#
#                 # if adding X -- Y, flip from/to and add Y -- X instead
#                 elif edge['type_'] == bidirectedEdgeType.id_:
#                     if existingEdge['type_'] == directedEdgeType.id_:
#                         temp = edge['from_']
#                         edge['from_'] = edge['to_']
#                         edge['to_'] = temp
#                     else:
#                         edgeType = directedEdgeType
#
#                         for id_ in edgeTypeMap:
#                             if edge['type_'].upper() == edgeTypeMap[id_].id_.upper():
#                                 edgeType = edgeTypeMap[id_]
#                                 break
#
#                         raise Exception('The edge ' + edge['from_'] + ' ' + edgeType.shortId + ' ' + edge['to_'] + ' already exists.')
#
#             # test if the new edge generates a cycle
#             self.__addEdge(edge)
#
#
#     def deleteEdges(self, edges):
#         edges = ou.makeArray(edges)
#
#         if not edges or len(edges) == 0:
#             return
#
#         # can we delete all at once instead?
#         for edge in edges:
#             self.__deleteEdge(edge)
#
#
#     # Node[]
#     # Node[]
#     def parents(self, nodes):
#         nodes = ou.makeArray(nodes)
#
#         if not nodes or len(nodes) == 0:
#             return []
#
#         names = dict()
#         parents = dict()
#
#         for node in nodes:
#             names[node['name']] = True
#
#         for edge in self.edges:
#             if edge['type_'] == directedEdgeType.id_ and edge['to_'] in names:
#                 parents[edge['from_']] = True
#
#         p = list(filter(lambda n: n['name'] in parents, self.nodes))
#
#         return p
#
#
#     def ancestors(self, nodes):
#         nodes = ou.makeArray(nodes)
#
#         if not nodes or len(nodes) == 0:
#             return []
#
#         AnNames = []
#
#         for node in nodes:
#             AnNames.extend(list(nx.ancestors(self.nx, node['name'])))
#
#         AnNames = su.unique(AnNames)
#
#         An = gu.getNodesByName(AnNames, self)
#         An = su.difference(An, nodes, 'name')
#
#         return An
#
#
#     def descendants(self, nodes):
#         nodes = ou.makeArray(nodes)
#
#         if not nodes or len(nodes) == 0:
#             return []
#
#         DeNames = []
#
#         # bug: nx treats bidirected as directed
#         for node in nodes:
#             DeNames.extend(list(nx.descendants(self.nx, node['name'])))
#
#         DeNames = su.unique(DeNames)
#         De = gu.getNodesByName(DeNames, self)
#         De = su.difference(De, nodes, 'name')
#
#         return De
#
#
#     def children(self, nodes):
#         nodes = ou.makeArray(nodes)
#
#         names = dict()
#         children = dict()
#
#         for node in nodes:
#             names[node['name']] = True
#
#         for edge in self.edges:
#             if edge['type_'] == directedEdgeType.id_ and edge['from_'] in names:
#                 children[edge['to_']] = True
#
#         c = list(filter(lambda n: n['name'] in children, self.nodes))
#
#         return c
#
#
#     # Node | Node[], EdgeType | str
#     # Node[]
#     def neighbors(self, nodes, edgeType = None):
#         nodes = ou.makeArray(nodes)
#
#         if not nodes or len(nodes) == 0:
#             return []
#
#         if isinstance(edgeType, EdgeType):
#             edgeType = edgeType.id_
#
#         names = dict()
#         neighbors = dict()
#
#         for node in nodes:
#             names[node['name']] = True
#
#         edges = list(filter(lambda e: e['type_'] == edgeType if edgeType else True, self.edges))
#
#         for edge in edges:
#             if edge['to_'] in names:
#                 neighbors[edge['from_']] = True
#             if edge['from_'] in names:
#                 neighbors[edge['to_']] = True
#
#         n = list(filter(lambda n: n['name'] in neighbors and n['name'] not in names, self.nodes))
#
#         return n
#
#
#     def connectedComponents(self):
#         # node names
#         # convert to nodes
#         CC = list(nx.connected_components(self.nx))
#         ccsNodes = []
#
#         # cc is a dict
#         for cc in CC:
#             nodeNames = []
#
#             for nodeName in cc:
#                 nodeNames.append(nodeName)
#
#             ccNodes = gu.getNodesByName(nodeNames, self)
#
#             ccsNodes.append(ccNodes)
#
#         return ccsNodes
#
#
#     def print(self):
#         if len(self.nodes) > 0:
#             print('<Nodes (' + str(len(self.nodes)) + ')>')
#         else:
#             print('<Nodes (none)>')
#
#         nodeNames = list(map(lambda n: n['name'], self.nodes))
#
#         print('\n'.join(nodeNames))
#
#         if len(self.edges) > 0:
#             print('\n<Edges (' + str(len(self.edges)) + ')>')
#         else:
#             print('\n<Edges (none)>')
#
#         edgeList = list(map(lambda e: e['from_'] + ' ' + edgeTypeMap[e['type_']].shortId  + ' ' + e['to_'], self.edges))
#
#         print('\n'.join(edgeList))
#
#
#     def __addNode(self, node):
#         self.nx.add_node(node['name'], label = node['label'], type_ = node['type_'], metadata = node['metadata'])
#
#     def __deleteNode(self, node):
#         self.nx.remove_node(node['name'])
#
#     def __addEdge(self, edge):
#         self.nx.add_edge(edge['from_'], edge['to_'], label = edge['label'], type_ = edge['type_'], metadata = edge['metadata'])
#
#     def __deleteEdge(self, edge):
#         self.nx.remove_edge(edge['from_'], edge['to_'])
# --- END OLD ---
