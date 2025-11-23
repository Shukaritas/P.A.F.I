import osmnx as ox
import networkx as nx

def load_osmnx_graph(place_name="San Miguel, Lima, Peru"):
    """
    Descarga el grafo vial de una zona específica usando OSMnx.
    Solo se cargan las vías accesibles por vehículos.
    """
    print(f"Descargando grafo de {place_name}...")
    G = ox.graph_from_place(place_name, network_type='drive')
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    print(f"Grafo cargado: {len(G.nodes)} nodos, {len(G.edges)} aristas.")
    return G
