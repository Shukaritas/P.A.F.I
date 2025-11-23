def bellman_ford(edges, num_vertices, start):
    """
    Implementación del algoritmo de Bellman–Ford.
    Parámetros:
        edges: lista de tuplas (u, v, peso)
        num_vertices: número total de vértices
        start: nodo origen
    Retorna:
        dist: diccionario con las distancias mínimas
        prev: diccionario con los predecesores
        has_negative_cycle: bool si se detectó un ciclo negativo
    """
    dist = {v: float('inf') for v in range(num_vertices)}
    dist[start] = 0
    prev = {v: None for v in range(num_vertices)}

    # Relajar todas las aristas V-1 veces
    for _ in range(num_vertices - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u

    # Detección de ciclos negativos
    has_negative_cycle = any(dist[u] + w < dist[v] for u, v, w in edges)

    return dist, prev, has_negative_cycle


def reconstruct_path(prev, start, goal):
    """
    Reconstruye el camino más corto usando el diccionario de predecesores.
    """
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    return path if path and path[0] == start else []
