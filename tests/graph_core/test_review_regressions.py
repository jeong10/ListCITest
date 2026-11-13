"""Regression tests for the findings of the Step-4 code review (views must stay isolated from later parent changes)."""
import pytest
import networkx as nx

from src.graph.classes.graph import Graph
from src.graph.classes.graph_defs import directedEdgeType
from src.inference.utils.graph_utils import GraphUtils as gu


def chain():
    return Graph(nodes=[{'name': n} for n in 'ABC'],
                 edges=[{'from_': 'A', 'to_': 'B'}, {'from_': 'B', 'to_': 'C'}])


def test_parent_node_attr_change_does_not_reach_existing_views():
    G = chain()
    H = gu.subgraph(G, G.nodes[:2])
    T = gu.transform(G, [G.nodes[1]], [])
    G.addNodes({'name': 'A', 'label': 'ZZ', 'type_': 'latent'})
    assert (H.nodes[0]['label'], H.nodes[0]['type_']) == ('A', 'basic')
    assert (T.nodes[0]['label'], T.nodes[0]['type_']) == ('A', 'basic')
    assert G.nodes[0]['label'] == 'ZZ'


def test_names_set_node_does_not_reach_existing_views():
    G = chain()
    H = G.names.subgraph(['A', 'B'])
    G.names.set_node('A', label='ZZ', type_='latent')
    assert H.names.node('A')['label'] == 'A' and H.names.node('A')['type_'] == 'basic'
    assert G.names.node('A')['label'] == 'ZZ'


def test_parent_edge_replacement_does_not_desync_existing_views():
    G = chain()
    H = gu.subgraph(G, G.nodes)
    with pytest.raises(Exception):
        G.addEdges({'from_': 'A', 'to_': 'B', 'type_': 'directed'})      # duplicate directed edge
    G.addEdges({'from_': 'A', 'to_': 'B', 'type_': 'undirected'})        # replaces the stored edge in the parent
    assert [n['name'] for n in H.parents(H.nodes[1])] == ['A']
    assert [(e['from_'], e['to_'], e['type_']) for e in H.edges] == [('A', 'B', 'directed'), ('B', 'C', 'directed')]
    assert G.edges[0]['type_'] == 'undirected'


def test_metadata_writes_on_subgraph_of_implicit_nodes_persist():
    G = Graph(edges=[{'from_': 'X', 'to_': 'Y'}])
    H = gu.subgraph(G, G.nodes)
    H.nodes[0]['metadata']['k'] = 1
    assert H.nodes[0]['metadata'] == {'k': 1}


def test_names_add_edge_accepts_edgetype_and_rejects_cycle():
    G = chain()
    with pytest.raises(ValueError):
        G.names.add_edge('C', 'A', directedEdgeType)
    G2 = chain()
    G2.names.add_edge('A', 'C', directedEdgeType)
    assert G2.edges[-1]['type_'] == 'directed' and G2.names.parents('C') == ['A', 'B']


def test_delete_nodes_removes_edges_of_listed_nodes_before_raising_for_missing():
    G = chain()
    with pytest.raises(nx.NetworkXError):
        G.deleteNodes([{'name': 'MISSING'}, {'name': 'B'}])
    assert [n['name'] for n in G.nodes] == ['A', 'B', 'C'] and G.edges == []   # legacy state: B's edges already gone


def test_networkx_error_wording():
    G = chain()
    with pytest.raises(nx.NetworkXError, match='not in the digraph'):
        G.deleteNodes({'name': 'NOPE'})
    if not Graph.ignore_missing_edge_deletion:
        with pytest.raises(nx.NetworkXError, match='not in graph'):
            G.deleteEdges({'from_': 'C', 'to_': 'A'})


def test_name_lookups_do_not_rebuild_core_after_mutation():
    G = chain()
    gu.getNodeByName('a', G)              # case-insensitive
    before = G._core
    G.addEdges({'from_': 'A', 'to_': 'C'})
    assert gu.getNodeByName('c', G)['name'] == 'C' and gu.getEdgeByName('A', 'C', G) is not None
    assert G._core is before              # name/edge lookups must not need a rebuilt core
