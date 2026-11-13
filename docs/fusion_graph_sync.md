# Graph core: sync notes with fusion_python

The graph classes in this repo are copied from `fusion_python` (branch `graph-rewrite`, `src` state as of commit
`bd93be7`), which is the ground truth for the graph core. See `docs/graph_api.md` for the API (legacy dict API plus
the new name-based `G.names` API).

Copied verbatim from fusion_python:
* `src/graph/core/_core.py` (hidden bitset core, mask views, memoized queries)
* `src/graph/classes/graph_names_api.py` (`G.names`)
* `src/inference/utils/graph_utils.py` (hot paths run on the core)
* `src/graph/classes/graph.py` (the commented-out `OLD` block at the bottom is fusion's pre-rewrite class)

ListCITest-only changes in `src/graph/classes/graph.py` (re-apply them when syncing again):
1. `import random`
2. `Graph.ignore_missing_edge_deletion = True` (fusion: `False`, i.e. deleting a missing edge raises `NetworkXError`)
3. `_stored_path`, `addRandomNodes`, `_addRandomDirectedEdges`, `addRandomEdges`, `toRandomGraph`: the exact `random` call
   sequence of the original implementation (seeded experiments are reproducible). The original `nx.find_cycle(self.nx)` check
   became a native reachability test over the stored edges (equivalent because the graph is acyclic before each insertion).

Phase 2 (fusion_python commit `830d4dd`), also migrated:
* copied verbatim: `src/inference/utils/set_utils.py` (hash-based, identical to the pydash results; has the `uniqueWith`
  alias that `ListCI` / `ListGMP` use), `src/graph/core/_core.py` and `src/graph/classes/graph_names_api.py` (Bayes ball:
  `G.names.m_separated` / `d_separated`).
* ported by hand (ListCITest keeps `DSeparation.test(G, X, Y, Z)` with a required `Z`): `separation_masks` + fast path in
  `src/path_analysis/d_separation.py`; `TestSep` fast path + new `TestSepUnprojected` in
  `src/adjustment/adjustment_sets_utils.py`; `FindSeparator` in `conditional_independencies.py` now calls
  `TestSepUnprojected` (the old `pu.unproject` + `TestSep` lines are commented as OLD).
* `tests/sep` (set_utils vs pydash, separation vs networkx); the tests that need fusion's `test/graph` corpus skip here.
* ListCI on 24 graphs (paper + bnlearn sm/md, py3.7): original 2.87 s -> 0.18 s with identical output.

Verification done for the migration (Python 3.12, `PYTHONHASHSEED=0`): CLI output of `main.py {gmp,lmp,clmp}` on
`graphs/paper` + `graphs/bnlearn/sm`, seeded random graph construction / projection, ListCI/ListCIBF results, and the
unit tests are identical to the pre-migration code; ListCI is ~2.8x faster on the paper and bnlearn graphs.
`tests/graph_core` holds the graph tests shared with fusion_python.

Phase 3 (fusion_python commit `1841f9d`, plus the review fix in the same tree): `src/testable_implications/list_ci_fast.py` (copied
verbatim; bitset translation of `ListCI`) and a small dispatch in `ConditionalIndependencies.ListCI` (`list_ci_fast(G, V)` first; the
original list-based loop below it stays as the fallback and reference). Same ordered output (golden: 0 differences); `tests/listci`
holds the identity regression test. On projected bnlearn graphs (U = 0-50 %, 10 samples, py3.7) the new code is 12-119x (sm),
100-1300x (md) and 630-2600x (lg) faster than the original; see fusion_python `bench/results/bnlearn_projection/REPORT.md`.
