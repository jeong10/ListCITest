from typing import List

from src.graph.classes.graph import Graph
from src.path_analysis.classes.path import Path
from src.path_analysis.classes.path_edge import PathEdge
from src.path_analysis.classes.direction import Direction as PathDirection
from src.inference.utils.graph_utils import compareNames

from src.inference.utils.graph_utils import GraphUtils as gu
from src.inference.utils.set_utils import SetUtils as su


class PathSearch():

    backtrack = 1
    forward = 0
    useCaches = True
    paths = List[Path]
    graph = Graph
    # sources = List[Node]
    # targets = List[Node]
    # edgeFilter: (graph: Graph, node: Node, previousInPath: PathEdge) => Edge[]
    # pathFilter: (path: Path) => boolean

    def __init__(self, graph, sources = [], targets = []):
        self.graph = graph
        self.sources = sources
        self.targets = targets
        self.edgeFilter = filterAllEdges
        self.pathFilter = filterAllPaths


    # Path, number
    # Path[]
    def findPaths(self, lastPath, limit):
        
        self.paths = []
        mode = self.forward
        lastSource = None
        visited = dict()

        # Phase 5.6 per-search caches (the graph does not change during a search): node dicts by name, edge-filter
        # results by (node, previous edge) with an edge -> position index, target names.
        # (``useCaches = False`` runs the search without them; used by the differential tests)
        if self.useCaches:
            self._nodeCache = dict()
            self._edgeCache = dict()
            self._targetNames = self._computeTargetNames()
        else:
            self._nodeCache = None
            self._edgeCache = None
            self._targetNames = None

        currentPath = Path([])

        if lastPath is not None and lastPath.length > 0:
            # make a copy of the passed path
            currentPath.copy(lastPath)

            # if the current path is not empty construct a corresponding visited record
            for edge in currentPath.edges:
                visited[edge.target] = True

            lastSource = currentPath.edges[0].source
            visited[currentPath.edges[0].source] = True
            visited[currentPath.lastEdge.target] = True
            mode = self.backtrack

        while limit > 0:
            if mode == self.forward:
                if currentPath.length == 0:
                    # if the path is empty
                    # let bracktrack pick a source and insert an edge in the path to follow
                    mode = self.backtrack
                else:
                    # the path is not empty; look at the last edge
                    lastEdge = currentPath.lastEdge

                    # mark the node reached by this edge
                    visited[lastEdge.source] = True

                    # have we reached a target?
                    if self.isTarget(self._node(lastEdge.target)):
                        # target reached
                        if not self.pathFilter or self.pathFilter(currentPath):
                            self.savePath(currentPath)
                            limit = limit - 1

                        mode = self.backtrack
                    else:
                        # can we move forward?
                        nextEdge = self.getEdgeAfter(self._node(lastEdge.target), lastEdge, None, visited)
                        
                        if nextEdge is not None:
                            currentPath.push(nextEdge)
                        else:
                            # backtrack if no further advance can be made in this path
                            mode = self.backtrack

            if mode == self.backtrack:
                if currentPath.length == 0:
                    # the path is empty
                    # find the last source we started from
                    sourceIdx = len(self.sources) - 1 - self.sources[::-1].index(lastSource) if lastSource is not None else -1
                    
                    # unmark the last source
                    # node?
                    if lastSource is not None:
                        if lastSource['name'] in visited:
                            del visited[lastSource['name']]
                    
                    # are there more sources?
                    if len(self.sources) > sourceIdx + 1:
                        # set the new 'lastSource'
                        lastSource = self.sources[sourceIdx + 1]

                        # does it have outgoing edges?
                        nextEdge = self.getEdgeAfter(lastSource, None, None, visited)
                        
                        if nextEdge is not None:
                            currentPath.push(nextEdge)
                            mode = self.forward

                        # if it does not just let the next iteration to pick the next source
                    else:
                        # no more source, we are done
                        break
                else:
                    # path not empty, get and remove the last edge
                    lastEdge = currentPath.pop()

                    # unmark the last node
                    if lastEdge.target in visited:
                        del visited[lastEdge.target]

                    # get the previous edge in the path
                    previousInPath = currentPath.lastEdge

                    # determine next edge to try
                    source = self._node(lastEdge.source)
                    nextEdge = self.getEdgeAfter(source, previousInPath, lastEdge, visited)
                    
                    if nextEdge is not None:
                        # if there is a next edge add it to the path
                        currentPath.push(nextEdge)
                        mode = self.forward

                    # if there is no next edge let the next iteration pick a new source

        return self.paths


    # Node, PathEdge, PathEdge, Dict
    # PathEdge
    def getEdgeAfter(self, node, previousInPath, previouslySelected = None, visited = dict()):
        if not self.edgeFilter or not node:
            return None
        
        edges, positions = self._edgesFor(node, previousInPath)
        index = 0

        if len(edges) == 0:
            return None

        if previouslySelected is not None:
            # move the iterator right to the previous edge: first position whose (from_, to_, type_) match
            # (Phase 5.6: dictionary lookup instead of the scan of the legacy code, see the OLD block below)
            prevEdge = previouslySelected.edge
            found = positions.get((prevEdge['from_'], prevEdge['to_'], prevEdge['type_']))
            index = len(edges) if found is None else found

            # move onto the next one
            index = index + 1

            # --- OLD (linear scan; same result) ---
            # prevEdge = previouslySelected.edge
            # currEdge = edges[index]
            # while index < len(edges) and not (currEdge['from_'] == prevEdge['from_'] and currEdge['to_'] == prevEdge['to_'] and currEdge['type_'] == prevEdge['type_']):
            #     index = index + 1
            #
            #     if index < len(edges):
            #         currEdge = edges[index]
            #
            # index = index + 1
            # --- END OLD ---

        if index >= len(edges):
            return None
        
        # move forward up to a non-visited edge
        nodeName = edges[index]['from_'] if edges[index]['to_'] == node['name'] else edges[index]['to_']
        
        # while index < len(edges) - 1 and nodeName in visited:
        while index < len(edges) and nodeName in visited:
            index = index + 1

            if index < len(edges):
                nodeName = edges[index]['from_'] if edges[index]['to_'] == node['name'] else edges[index]['to_']
        
        if index >= len(edges):
            return None
        else:
            direction = PathDirection.reversed if edges[index]['to_'] == node['name'] else PathDirection.directed

            return PathEdge(edges[index], direction)


    # str
    # Node
    def _node(self, name):
        cache = getattr(self, '_nodeCache', None)

        if cache is None:
            return gu.getNodeByName(name, self.graph)

        if name not in cache:
            cache[name] = gu.getNodeByName(name, self.graph)

        return cache[name]

    # Node, PathEdge
    # (Edge[], Dict)
    def _edgesFor(self, node, previous):
        """``edgeFilter(graph, node, previous)`` plus {(from_, to_, type_): first position}; both cached per search by the
        identity of ``node`` and ``previous`` (the filters are functions of graph, node and previous only)."""
        cache = getattr(self, '_edgeCache', None)

        if cache is not None:
            key = (id(node), id(previous))
            hit = cache.get(key)

            if hit is not None and hit[0] is node and hit[1] is previous:
                return hit[2], hit[3]

        edges = self.edgeFilter(self.graph, node, previous)
        positions = dict()

        for i, e in enumerate(edges):
            positions.setdefault((e['from_'], e['to_'], e['type_']), i)

        if cache is not None:
            if len(cache) > 50000:
                cache.clear()

            cache[key] = (node, previous, edges, positions)

        return edges, positions

    def _computeTargetNames(self):
        try:
            return set(t['name'] for t in self.targets)
        except (TypeError, KeyError):
            return None

    # Node
    # boolean
    def isTarget(self, node):
        if node is None:
            return False

        names = getattr(self, '_targetNames', None)

        if names is not None:
            return node['name'] in names

        return su.belongs(node, self.targets, compareNames)


    def savePath(self, path):
        pathObj = Path()
        pathObj.copy(path)
        self.paths.append(pathObj)


def filterAllPaths(path):
    return True

def filterAllEdges(graph, node, previous):
    return gu.getOutgoing(node, graph) + gu.getIncoming(node, graph)