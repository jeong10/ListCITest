from typing import Dict, Any

# from graph.classes.graph_defs import latentNodeType, directedEdgeType, bidirectedEdgeType

from src.graph.classes.graph import Graph
import itertools

from src.graph.classes.graph_defs import latentNodeType, directedEdgeType, bidirectedEdgeType, basicNodeType

from src.inference.utils.graph_utils import GraphUtils as gu
from src.inference.utils.set_utils import SetUtils as su
from src.common.object_utils import ObjectUtils as ou


class ProjectionUtils():


    # Graph
    # Graph
    @staticmethod
    def projectOverNonLatentNodes(G):
        if not G:
            return None

        V = list(filter(lambda n: n['type_'] != latentNodeType.id_, G.nodes))

        graph = ProjectionUtils.projectOver(G, V)

        return graph


    @staticmethod
    def _projectOverFast(G):
        """Latent elimination of ``projectOver`` on the directed store of a copy of ``G`` (no per-pair graph API calls,
        no core rebuild after each added edge).  Reproduces the legacy loop exactly: latents in ``topoSort`` order of the
        copy; parents/children (directed edges, basic or latent nodes, graph order) taken when the latent's turn starts;
        new edges appended in the legacy order; incident ``<->`` nodes in legacy edge order; ``getEdgeByName`` only looks
        at the exact orientation.  The five ``getEdgeByName`` calls of the legacy step 3 whose results are unused are
        dropped.  Returns ``None`` (caller runs the legacy code) outside the domain: undirected graph, node names that
        collide after ``correctNodeName``, self loops, an existing non-directed edge where a directed one would be added,
        or any ``addEdges`` situation that is not a plain append (existing opposite/other-type edge)."""
        if G._ug is not None:
            return None

        graph = G.copy()
        graph._ensure_store()
        first, last, gnames, gattrs = graph._cname_maps()

        if len(first) != len(gnames):
            return None

        s = graph._s
        nodes, succ, pred = s.nodes, s.succ, s.pred
        # same call as the legacy code: raises the same error on cyclic graphs
        orderedV = gu.topoSort(graph)
        DIR, BI, LAT, BAS = directedEdgeType.id_, bidirectedEdgeType.id_, latentNodeType.id_, basicNodeType.id_
        rank = dict((nm, i) for i, nm in enumerate(nodes))
        basicOrLatent = (BAS, LAT)

        for L in orderedV:
            if L['type_'] != LAT:
                continue

            l = L['name']

            if l in succ[l]:
                return None

            # step 3: every parent -> every child
            parents = sorted((p for p, d in pred[l].items()
                              if d.get('type_', DIR) == DIR and nodes[p].get('type_', BAS) in basicOrLatent),
                             key=rank.__getitem__)
            children = sorted((c for c, d in succ[l].items()
                               if d.get('type_', DIR) == DIR and nodes[c].get('type_', BAS) in basicOrLatent),
                              key=rank.__getitem__)

            for p in parents:
                sp = succ[p]

                for c in children:
                    if p == c:
                        return None

                    d = sp.get(c)

                    if d is None:
                        sp[c] = pred[c][p] = {'label': None, 'type_': DIR, 'metadata': {'directedLatentPath': True}}
                    elif d.get('type_', DIR) != DIR:
                        # addEdges(p -> c) over an existing p <-> c: the <-> edge is deleted and re-added as (c, p)
                        # (end of succ[c]), then p -> c is appended; anything else is left to the legacy code
                        if d.get('type_', DIR) != BI or succ[c].get(p) is not None:
                            return None

                        del sp[c]
                        del pred[c][p]
                        succ[c][p] = pred[p][c] = {'label': d['label'] if 'label' in d else None, 'type_': BI,
                                                   'metadata': d['metadata'] if 'metadata' in d else {}}
                        sp[c] = pred[c][p] = {'label': None, 'type_': DIR, 'metadata': {'directedLatentPath': True}}

            # step 4: nodes reached through <-> edges incident to the latent, in legacy edge order
            inc = []

            for u, d in pred[l].items():
                if d.get('type_', DIR) == BI and u != l:
                    inc.append((rank[u], 0, u))

            for k, (v, d) in enumerate(succ[l].items()):
                if d.get('type_', DIR) == BI and v != l:
                    inc.append((rank[l], k, v))

            inc.sort()
            incident = list(children)
            seen = set(incident)

            for _, _, other in inc:
                if other not in seen:
                    seen.add(other)
                    incident.append(other)

            sl = succ[l]

            for X, Y in itertools.permutations(incident, 2):
                dX = sl.get(X)
                dY = sl.get(Y)
                dirLtoX = dX is not None and dX.get('type_', DIR) == DIR
                dirLtoY = dY is not None and dY.get('type_', DIR) == DIR
                biLX = dX is not None and dX.get('type_', DIR) == BI
                biLY = dY is not None and dY.get('type_', DIR) == BI

                if not biLX:
                    d2 = succ[X].get(l)
                    biLX = d2 is not None and d2.get('type_', DIR) == BI

                if not biLY:
                    d2 = succ[Y].get(l)
                    biLY = d2 is not None and d2.get('type_', DIR) == BI

                if (biLX and dirLtoY) or (dirLtoX and biLY) or (dirLtoX and dirLtoY):
                    dXY = succ[X].get(Y)
                    dYX = succ[Y].get(X)

                    if (dXY is not None and dXY.get('type_', DIR) == BI) or (dYX is not None and dYX.get('type_', DIR) == BI):
                        continue

                    # addEdges(X <-> Y): plain append, or flip to (Y, X) when a directed X -> Y exists
                    if dXY is None:
                        succ[X][Y] = pred[Y][X] = {'label': None, 'type_': BI, 'metadata': {}}
                    elif dXY.get('type_', DIR) == DIR and dYX is None:
                        succ[Y][X] = pred[X][Y] = {'label': None, 'type_': BI, 'metadata': {}}
                    else:
                        return None

            s.remove_node(l)

        graph._sv += 1

        return graph

    # Graph, Node[]
    # Graph
    @staticmethod
    def projectOver(G, V):
        if not G:
            return None
        
        if not V or len(V) == 0:
            return Graph()

        # nodes may not belong to the given graph
        # so map the nodes in G by each node name
        V = list(map(lambda n: gu.getNodeByName(n['name'], G), V))
        V = list(filter(lambda n: n is not None, V))

        # Phase 5 fast path: the same latent elimination simulated directly on the directed store (see
        # _projectOverFast); None = outside its domain, the legacy code below runs unchanged.
        fast = ProjectionUtils._projectOverFast(G)

        if fast is not None:
            return fast

        graph = G.copy()

        # # remove nodes not in V
        # remaining = su.difference(graph.nodes, V, 'name')

        # graph.deleteNodes(remaining)

        orderedV = gu.topoSort(graph)
        latentsInOrder = list(filter(lambda n: n['type_'] == latentNodeType.id_, orderedV))
        
        for latent in latentsInOrder:
            # step 3
            parents = gu.filterBasicNodes(gu.parents(latent, graph), False)
            children = gu.filterBasicNodes(gu.children(latent, graph), False)

            for parent in parents:
                for child in children:
                    dirPtoC = gu.getEdgeByName(parent['name'], child['name'], graph, directedEdgeType)

                    # add latent nodes/edges info to new edge metadata
                    dirPtoL = gu.getEdgeByName(parent['name'], latent['name'], graph, directedEdgeType)
                    dirLtoC = gu.getEdgeByName(latent['name'], child['name'], graph, directedEdgeType)

                    dirPtoLinG = gu.getEdgeByName(parent['name'], latent['name'], G, directedEdgeType)
                    dirLtoCinG = gu.getEdgeByName(latent['name'], child['name'], G, directedEdgeType)

                    edgeToOperate = None

                    # if edge already exists, add latent nodes/edges info to metadata
                    if dirPtoC:
                        edgeToOperate = dirPtoC
                    else:
                        edge = {
                            'from_': parent['name'], 'to_': child['name'], 'type_': directedEdgeType.id_, 'metadata': {
                                'directedLatentPath': True
                                # necessary?
                                # , 'latentNodeIds': [], 'latentEdgeIds': []
                            }
                        }

                        graph.addEdges(edge)

                        edgeToOperate = edge
                    
                    if not edgeToOperate['metadata']:
                        edgeToOperate['metadata'] = {}

            # step 4
            incidentBidirected = list(filter(lambda e: e['type_'] == bidirectedEdgeType.id_, gu.getIncoming(latent, graph)))
            incidentNodesViaBidirected = []

            for edge in incidentBidirected:
                fromNode = gu.getNodeByName(edge['from_'], graph)
                toNode = gu.getNodeByName(edge['to_'], graph)

                if fromNode['name'] is not latent['name']:
                    incidentNodesViaBidirected.append(fromNode)
                elif toNode['name'] is not latent['name']:
                    incidentNodesViaBidirected.append(toNode)

            incidentNodes = su.union(children, incidentNodesViaBidirected, 'name')
            pairs = ou.pairs(incidentNodes, True)
            
            for pair in pairs:
                X = pair[0]
                Y = pair[1]

                dirLtoX = gu.getEdgeByName(latent['name'], X['name'], graph, directedEdgeType)
                dirLtoY = gu.getEdgeByName(latent['name'], Y['name'], graph, directedEdgeType)
                biLX = gu.getEdgeByName(latent['name'], X['name'], graph, bidirectedEdgeType)
                biLY = gu.getEdgeByName(latent['name'], Y['name'], graph, bidirectedEdgeType)

                if not biLX:
                    biLX = gu.getEdgeByName(X['name'], latent['name'], graph, bidirectedEdgeType)

                if not biLY:
                    biLY = gu.getEdgeByName(Y['name'], latent['name'], graph, bidirectedEdgeType)

                # X <-> L -> Y
                case1 = biLX is not None and dirLtoY is not None
                # X <- L <-> Y
                case2 = dirLtoX is not None and biLY is not None
                # X <- L -> Y
                case3 = dirLtoX is not None and dirLtoY is not None

                if case1 or case2 or case3:
                    biXY = gu.getEdgeByName(X['name'], Y['name'], graph, bidirectedEdgeType)
                    biYX = gu.getEdgeByName(Y['name'], X['name'], graph, bidirectedEdgeType)

                    # if edge already exists, add latent nodes/edges info to metadata
                    if biXY is not None or biYX is not None:
                        continue

                    edge = {
                        'from_': X['name'], 'to_': Y['name'], 'type_': bidirectedEdgeType.id_, 'metadata': {}
                    }

                    graph.addEdges(edge)

            graph.deleteNodes(latent)

        return graph


    @staticmethod
    def _unprojectFast(G):
        """``unproject`` on the directed store: every ``<->`` edge (store order) is removed, then the latent nodes
        ``U_{ab}`` are appended in that order, then their two outgoing edges per latent (from_ first).  ``None`` when the
        legacy code must run: undirected graph, self-loop ``<->``, or latent names that repeat or already exist."""
        if G._ug is not None:
            return None

        graph = G.copy()
        graph._ensure_store()
        s = graph._s
        DIR, BI, LAT = directedEdgeType.id_, bidirectedEdgeType.id_, latentNodeType.id_

        bi = [(u, v) for u, nb in s.succ.items() for v, d in nb.items() if d.get('type_', DIR) == BI]
        names = []
        seen = set()

        for u, v in bi:
            if u == v:
                return None

            name = ''.join(['U_{', u, v, '}']) if u < v else ''.join(['U_{', v, u, '}'])

            if name in seen or name in s.nodes:
                return None

            seen.add(name)
            names.append(name)

        for u, v in bi:
            s.remove_edge(u, v)

        for name in names:
            s.add_node(name, {'label': name, 'type_': LAT, 'metadata': {}})

        for name, (u, v) in zip(names, bi):
            s.add_edge(name, u, {'label': None, 'type_': DIR, 'metadata': {}})
            s.add_edge(name, v, {'label': None, 'type_': DIR, 'metadata': {}})

        graph._sv += 1

        return graph

    # Graph
    # Graph
    @staticmethod
    def unproject(G):
        if not G:
            return G

        # Phase 5.4 fast path: one pass on the directed store of the copy (None = legacy code below)
        fast = ProjectionUtils._unprojectFast(G)

        if fast is not None:
            return fast

        graph = G.copy()

        bidirectedEdges = list(filter(lambda e: e['type_'] == bidirectedEdgeType.id_, graph.edges))
        
        latentNodes = []
        latentEdges = []
        
        for edge in bidirectedEdges:
            nodeInfo = ProjectionUtils.bidirectedEdgeToLatentNode(edge, G)
            
            if nodeInfo is None:
                continue

            latentNodes.append(nodeInfo.node)
            latentEdges.append(nodeInfo.fromEdge)
            latentEdges.append(nodeInfo.toEdge)
        
        graph.deleteEdges(bidirectedEdges)
        
        graph.addNodes(latentNodes)
        graph.addEdges(latentEdges)

        return graph


    # Edge, Graph
    # BidirectedEdgeInfo
    @staticmethod
    def bidirectedEdgeToLatentNode(edge, G):
        if not edge or edge['type_'] != bidirectedEdgeType.id_:
            return None
        
        latentNodeName = 'U'
        
        fromComesFirst = edge['from_'] < edge['to_']

        if fromComesFirst:
            # latentNodeName += '_{' + edge.from + '' + edge.to + '}'
            latentNodeName = ''.join(['U_{', edge['from_'], edge['to_'], '}'])
        else:
            # latentNodeName += '_{' + edge.to + '' + edge.from + '}'
            latentNodeName = ''.join(['U_{', edge['to_'], edge['from_'], '}'])

        latentNode = {
            'name': latentNodeName,
            'type_': latentNodeType.id_,
        }
        fromEdge = {
            'from_': latentNodeName,
            'to_': edge['from_'],
            'type_': directedEdgeType.id_
        }
        toEdge = {
            'from_': latentNodeName,
            'to_': edge['to_'],
            'type_': directedEdgeType.id_
        }

        return BidirectedEdgeInfo(latentNode, fromEdge, toEdge)


class BidirectedEdgeInfo():

    node = Dict[str, Any]
    fromEdge = Dict[str, Any]
    toEdge = Dict[str, Any]

    def __init__(self, node, fromEdge, toEdge):
        self.node = node
        self.fromEdge = fromEdge
        self.toEdge = toEdge