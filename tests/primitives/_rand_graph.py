"""Random ADMG generator shared by the primitive tests."""
from src.graph.classes.graph import Graph


def rand_graph(rng, n):
    names = ['V%d' % i for i in range(n)]
    types = ['basic'] * n
    if rng.random() < 0.2:
        types[rng.randrange(n)] = 'latent'
    nodes = [{'name': nm, 'label': nm, 'type_': t} for nm, t in zip(names, types)]
    order = names[:]
    rng.shuffle(order)
    edges = []
    seen = set()
    p = rng.choice([0.2, 0.35, 0.5])
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                a, b = order[i], order[j]
                edges.append({'from_': a, 'to_': b, 'type_': 'directed'})
                seen.add(frozenset((a, b)))
    for _ in range(rng.choice([0, 0, 1, 2, 3])):
        a, b = rng.sample(names, 2)
        if frozenset((a, b)) in seen:
            continue
        seen.add(frozenset((a, b)))
        edges.append({'from_': a, 'to_': b, 'type_': 'bidirected'})
    return Graph(nodes=nodes, edges=edges)
