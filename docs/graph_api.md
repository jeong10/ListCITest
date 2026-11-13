# Graph API (`src.graph.classes.graph.Graph`)

`Graph` has two APIs on the same object:

* the **legacy dict API** (`G.nodes`, `G.edges`, `G.parents(nodes)`, `G.addEdges(edges)`, `gu.*`): nodes and
  edges are dicts, behaviour is unchanged from before the rewrite;
* the **name API** `G.names.*` (new, recommended for new code): takes and returns plain node *names*.

```python
from src.graph.classes.graph import Graph

G = Graph.from_json({
    'nodes': [{'name': n} for n in ('X', 'Y', 'Z', 'W')],
    'edges': [{'from': 'X', 'to': 'Y'},
              {'from': 'Y', 'to': 'Z'},
              {'from': 'X', 'to': 'Z', 'type': 'bidirected'}],
})

G.names.parents('Z')                 # ['Y']
G.names.ancestors('Z')               # ['X', 'Y']       (include_self=True adds 'Z')
G.names.spouses('X')                 # ['Z']            (bidirected neighbors)
G.names.c_components()               # [['X', 'Z'], ['Y'], ['W']]
G.names.topological_order()          # ['X', 'Y', 'Z', 'W']
Gx = G.names.do(over='Y')            # G with edges into Y removed (cheap view)
Gs = G.names.subgraph(['X', 'Y'])    # induced subgraph (cheap view)
G.to_json()                          # back to the web front end's schema
```

## Conventions
* Arguments: a name or any iterable of names. Unknown names raise `KeyError`.
* Results: lists of names in **graph order** (node insertion order), so results are deterministic.
* `ancestors(x)` / `descendants(x)` exclude every node of `x` unless `include_self=True`.
* `parents` / `children` do not exclude the query nodes; `spouses` / `neighbors` do.
* `neighbors(x, edge_type=None)`: `edge_type` is an id (`'directed'`, `'bidirected'`, ...) or an `EdgeType`.

## Views
`subgraph`, `do` and `ancestral` return new `Graph` objects that are *views*: they share the parent's
structure snapshot and cost O(V) bit operations instead of a copy. A view is turned into a real copy only
when you mutate it or read its edges through the legacy API. `metadata` dicts are shared by reference with the parent (as with the legacy `Graph.copy()`); label/type_ edits and replaced edges never reach an existing view (attribute dicts are replaced copy-on-write). Structural queries are memoized.

## Editing
`add_node`, `add_edge(u, v, type_, check_cycle=True)`, `remove_node`, `remove_edge`, `set_node(name, label,
type_, metadata)`. `names.add_edge` rejects edges that close a directed cycle (`ValueError`); the legacy
`addEdges` never checked (cycles surface later, e.g. in `topoSort`).
`G.struct_version` changes when nodes/edges are added/removed, `G.attr_version` on attribute edits.

## Not available / notes
* The name API is not available on undirected (moralized) graphs (`toUndirected()` / `gu.moralize`).
* `G.nx` returns a *fresh* `nx.DiGraph` per access (writes to it do not change `G`); for undirected graphs it is the live `nx.Graph`.
* Python >= 3.7.
