from src.inference.utils.graph_utils import compareNames

from src.inference.utils.graph_utils import GraphUtils as gu
from src.inference.utils.set_utils import SetUtils as su
from src.common.object_utils import ObjectUtils as ou

from src.path_analysis.d_separation import DSeparation, separation_masks


def TestSepUnprojected(G, X, Y, Z=[]):
    """``TestSep(ProjectionUtils.unproject(G), X, Y, Z)`` without building the unprojected graph.

    Semi-Markovian Bayes ball: every ``<->`` edge acts as a hidden common parent.  Falls back to the
    explicit unprojection when the native path does not apply (undirected / unusual edge types,
    unknown node names, or ``unproject`` latent-name collisions).
    """
    X = ou.makeArray(X)
    Y = ou.makeArray(Y)
    Z = ou.makeArray(Z)

    fast = separation_masks(G, X, Y, Z)

    if fast is not None and fast[0].core.unproject_safe():
        v, xm, ym, zm = fast

        return v.bayes_ball_admg(xm, ym, zm)

    from src.projection.projection_utils import ProjectionUtils as pu

    return TestSep(pu.unproject(G), X, Y, Z)


def TestSep(G, X, Y, Z=[]):
    X = ou.makeArray(X)
    Y = ou.makeArray(Y)
    Z = ou.makeArray(Z)

    # fast path (Phase 2): non-Markovian Bayes ball on bitsets (directed edges only, like the code below)
    fast = separation_masks(G, X, Y, Z)

    if fast is not None:
        v, xm, ym, zm = fast

        return v.bayes_ball_dag(xm, ym, zm)

    # --- legacy implementation (fallback) ---
    # Bayes Ball
    # https://github.com/lingxuez/bayes-net/blob/9ba18d08f85d04b566c6a103be67242eab8dc95a/src/BN.py#L114
    AnZ = gu.ancestors(Z, G)

    Q = []

    for x in X:
        Q.append((x['name'], 'up'))

    visited = set()

    while len(Q) > 0:
        (name, dir) = Q.pop()
        node = gu.getNodeByName(name, G)

        if (name, dir) not in visited:
            visited.add((name, dir))

            if not su.belongs(node, Z, compareNames) and su.belongs(node, Y, compareNames):
                return False

            if dir == 'up' and not su.belongs(node, Z, compareNames):
                for parent in G.parents(node):
                    Q.append((parent['name'], 'up'))

                for child in G.children(node):
                    Q.append((child['name'], 'down'))

            elif dir == 'down':
                if not su.belongs(node, Z, compareNames):
                    for child in G.children(node):
                        Q.append((child['name'], 'down'))

                if su.belongs(node, Z, compareNames) or su.belongs(node, AnZ, compareNames):
                    for parent in G.parents(node):
                        Q.append((parent['name'], 'up'))

    return True

# def FindSeparator(G, X, Y, I, R):
#     X = ou.makeArray(X)
#     Y = ou.makeArray(Y)
#     I = ou.makeArray(I)
#     R = ou.makeArray(R)

#     XY = su.union(X, Y, 'name')
#     XYI = su.union(XY, I, 'name')
#     Rprime = su.difference(R, XY, 'name')
#     Z = su.intersection(Rprime, gu.ancestorsPlus(XYI, G), 'name')

#     # if TestSep(G, X, Y, Z):
#     if DSeparation.test(G, X, Y, Z):
#         return Z
#     else:
#         return None

def writeNodeNames(nodes):
    return ', '.join(nodes) if len(nodes) > 0 else '\emptyset'

def nodeNamesToString(nodes, sortByName = True):
    names = list(map(lambda n: n['name'] if n is not None and 'name' in n else '\emptyset', nodes))

    if sortByName:
        names = sorted(names)

    return writeNodeNames(names)
