class UnionFind:
    """
    Implementación del algoritmo Union–Find (Disjoint Set Union).
    Utilizado para determinar si dos nodos están conectados.
    """

    def __init__(self, vertices):
        self.parent = {v: v for v in vertices}
        self.rank = {v: 0 for v in vertices}

    def find(self, v):
        """Encuentra el representante (raíz) del conjunto que contiene a 'v'."""
        if self.parent[v] != v:
            self.parent[v] = self.find(self.parent[v])  # Compresión de caminos
        return self.parent[v]

    def union(self, a, b):
        """Une los conjuntos que contienen a 'a' y 'b'."""
        rootA = self.find(a)
        rootB = self.find(b)
        if rootA != rootB:
            if self.rank[rootA] < self.rank[rootB]:
                self.parent[rootA] = rootB
            elif self.rank[rootA] > self.rank[rootB]:
                self.parent[rootB] = rootA
            else:
                self.parent[rootB] = rootA
                self.rank[rootA] += 1

    def connected(self, a, b):
        """Verifica si 'a' y 'b' pertenecen al mismo conjunto."""
        return self.find(a) == self.find(b)
