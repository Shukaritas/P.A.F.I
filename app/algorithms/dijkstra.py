def dijkstra(graph, start):
    """
    Implementación del algoritmo de Dijkstra.
    Parámetros:
        graph: dict { nodo: [(vecino, peso), ...] }
        start: nodo origen
    Retorna:
        dist: diccionario con las distancias mínimas desde 'start'
        prev: diccionario con el predecesor de cada nodo
    """
    dist = {v: float('inf') for v in graph}
    dist[start] = 0
    visited = set()
    prev = {v: None for v in graph}

    while len(visited) < len(graph):
        # Seleccionar el nodo no visitado con menor distancia
        u = min((v for v in graph if v not in visited), key=lambda x: dist[x], default=None)
        if u is None or dist[u] == float('inf'):
            break
        visited.add(u)

        # Relajar las aristas
        for neighbor, weight in graph[u]:
            if neighbor not in visited:
                alt = dist[u] + weight
                if alt < dist[neighbor]:
                    dist[neighbor] = alt
                    prev[neighbor] = u

    return dist, prev


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
