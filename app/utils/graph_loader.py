import pandas as pd
import networkx as nx
import os

def load_graph(base_path="app/data"):
    """
    Carga los archivos CSV y construye el grafo dirigido.
    - Los nodos provienen de 'nodos_lima_callao.csv'
    - Las aristas provienen de 'aristas_lima_callao.csv'
    - Los hospitales y centros de salud se agregan con atributos extra
    Retorna: grafo nx.DiGraph
    """

    # === 1️⃣ CARGA DE NODOS ===
    nodes_path = os.path.join(base_path, "nodos_lima_callao.csv")
    nodes_df = pd.read_csv(nodes_path, dtype={'node_id': str})

    # === 2️⃣ CARGA DE ARISTAS ===
    edges_path = os.path.join(base_path, "aristas_lima_callao.csv")
    edges_df = pd.read_csv(edges_path, dtype={'source': str, 'target': str})

    # === 3️⃣ CARGA DE CENTROS DE SALUD ===
    hospitals_path = os.path.join(base_path, "Centros_de_Salud.csv")
    hospitals_df = pd.read_csv(hospitals_path, dtype={'OBJECTID': str})

    # === 4️⃣ CONSTRUCCIÓN DEL GRAFO ===
    G = nx.DiGraph()

    # Añadir todos los nodos con sus coordenadas
    for _, row in nodes_df.iterrows():
        G.add_node(row['node_id'], lat=row['lat'], lon=row['lon'], tipo="nodo_vial")

    # Añadir aristas (respetando sentido y peso)
    for _, row in edges_df.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['longitud_m'])
        # Si la vía NO es de un solo sentido ("F" = doble vía)
        if str(row.get('oneway', 'F')).strip().upper() == 'F':
            G.add_edge(row['target'], row['source'], weight=row['longitud_m'])

    # Añadir los hospitales / centros de salud
    for _, row in hospitals_df.iterrows():
        hospital_id = f"HOSP_{row['OBJECTID']}"
        G.add_node(
            hospital_id,
            lat=row['Y'],
            lon=row['X'],
            nombre=row['NOMBRE'],
            tipo=row['Tipo'],
            distrito=row['NOMBDIST'],
            provincia=row['NOMBPROV'],
            departamento=row['NOMBDEP'],
            es_hospital=True
        )

    print(f"Grafo cargado con {len(G.nodes)} nodos y {len(G.edges)} aristas.")
    return G, hospitals_df
