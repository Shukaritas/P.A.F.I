from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Tuple, Optional, List, Dict, Any
from geopy.distance import geodesic
import networkx as nx
import numpy as np
import pandas as pd

from app.utils.graph_loader import load_graph

# -----------------------------
# MODELO DE ENTRADA
# -----------------------------
class RouteRequest(BaseModel):
    latitude: float
    longitude: float
    severity: str              # "leve" | "moderada" | "grave" | "todos"
    algorithm: str = "dijkstra"  # "dijkstra" | "bellman_ford" | "union_find"


# -----------------------------
# FASTAPI + STATIC
# -----------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index():
    return FileResponse("static/index.html")


# -----------------------------
# CARGA DEL GRAFO (LIMA + CALLAO)
# -----------------------------
print("Cargando grafo de Lima y Callao desde CSV...")

# G: grafo vial dirigido (Lima + Callao)
# hospitals_df: catálogo de centros de salud (con columnas X, Y, NOMBRE, Tipo, etc.)
graph, hospitals_df = load_graph(base_path="app/data")

graph_bellman = graph.copy()
for u, v, data in graph_bellman.edges(data=True):
    L = float(data.get("weight", 0.0))
    tipo_via = str(data.get("tipo_via", "")).lower()
    
    # Penaliza calles internas / residenciales
    if tipo_via in ["residential", "service", "living_street"]:
        data["weight"] = L * 2.5
    # Favorece vías principales
    elif tipo_via in ["primary", "secondary", "trunk", "motorway"]:
        data["weight"] = L * 0.8

# Cargamos también el listado de nodos viales para búsquedas rápidas de vecinos
nodes_df = pd.read_csv("app/data/nodos_lima_callao.csv", dtype={"node_id": str})
NODE_IDS = nodes_df["node_id"].values
NODE_LATS = nodes_df["lat"].values
NODE_LONS = nodes_df["lon"].values

print(f"Grafo listo: {len(graph.nodes)} nodos, {len(graph.edges)} aristas.")

# Preprocesamos los tipos de centros de salud para cada severidad
ALL_TYPES = sorted(set(hospitals_df["Tipo"].astype(str)))


def build_severity_types() -> Dict[str, List[str]]:
    grave = []
    moderada_extra = []
    for t in ALL_TYPES:
        tu = t.upper().replace("Í", "I")  # CLÍNICA -> CLINICA
        if "HOSPITAL" in tu:
            grave.append(t)
        if "CLINIC" in tu or "EMERGENCIA" in tu or "URGENCIA" in tu or "POLICLIN" in tu:
            moderada_extra.append(t)

    grave = sorted(set(grave))
    moderada = sorted(set(grave + moderada_extra))
    leve = sorted(set(ALL_TYPES))  

    return {
        "grave": grave,
        "moderada": moderada,
        "leve": leve,
        "todos": leve,
    }

SEVERITY_TYPES = build_severity_types()


def severity_to_types(severity: str) -> List[str]:
    s = (severity or "").lower()
    return SEVERITY_TYPES.get(s, SEVERITY_TYPES["leve"])


# -----------------------------
# HELPERS
# -----------------------------
def nearest_graph_node(lat: float, lon: float) -> str:
    """
    Devuelve el id del nodo vial más cercano (en todo Lima + Callao),
    usando una búsqueda vectorizada en NumPy (rápida incluso con ~200k nodos).
    """
    dlat = NODE_LATS - lat
    dlon = NODE_LONS - lon
    # Distancia aproximada en grados (suficiente para escoger el más cercano)
    d2 = dlat * dlat + dlon * dlon
    idx = int(np.argmin(d2))
    return str(NODE_IDS[idx])


def find_nearest_place(
    lat: float,
    lon: float,
    severity: str,
) -> Optional[Tuple[float, float, str, str]]:
    """
    Busca el establecimiento de salud más cercano en TODO Lima + Callao,
    según la gravedad.
    Retorna (lat, lon, nombre, tipo) o None si no encuentra.
    """
    allowed_types = severity_to_types(severity)
    df = hospitals_df[hospitals_df["Tipo"].isin(allowed_types)]

    if df.empty:
        return None

    min_row = None
    min_dist = float("inf")

    for _, row in df.iterrows():
        plat = row["Y"]
        plon = row["X"]
        try:
            d = geodesic((lat, lon), (plat, plon)).km
        except Exception:
            # Si geodesic falla por algún motivo, usamos distancia euclidiana
            d = ((lat - plat) ** 2 + (lon - plon) ** 2) ** 0.5

        if d < min_dist:
            min_dist = d
            min_row = row

    if min_row is None:
        return None

    return (
        float(min_row["Y"]),
        float(min_row["X"]),
        str(min_row.get("NOMBRE", "Centro sin nombre")),
        str(min_row.get("Tipo", "desconocido")),
    )


def route_as_coords(G: nx.DiGraph, route_nodes: List[str]) -> List[Tuple[float, float]]:
    """Convierte nodos del grafo a lista [(lat, lon), ...] para Leaflet."""
    coords: List[Tuple[float, float]] = []
    for n in route_nodes:
        data = G.nodes[n]
        # En el grafo CSV usamos lat/lon
        coords.append((float(data["lat"]), float(data["lon"])))
    return coords

def get_subgraph(G, node_start, node_end, radius_dist=0.035):
    try:
        n1 = G.nodes[node_start]
        n2 = G.nodes[node_end]
        
        min_lat = min(n1['lat'], n2['lat']) - radius_dist
        max_lat = max(n1['lat'], n2['lat']) + radius_dist
        min_lon = min(n1['lon'], n2['lon']) - radius_dist
        max_lon = max(n1['lon'], n2['lon']) + radius_dist
        
        selected_nodes = [
            n for n, d in G.nodes(data=True) 
            if min_lat <= d['lat'] <= max_lat and min_lon <= d['lon'] <= max_lon
        ]
        
        # Validación de seguridad: si cortamos de más, devolvemos todo
        if node_start not in selected_nodes or node_end not in selected_nodes:
            return G
            
        return G.subgraph(selected_nodes)
    except Exception:
        return G

# -----------------------------
# LÓGICA POR ALGORITMO
# -----------------------------
def compute_route_for_severity(
    orig_node: str,
    orig_lat: float,
    orig_lon: float,
    severity: str,
    algorithm: str,
) -> Dict[str, Any]:
    """
    Calcula una ruta para una severidad y un algoritmo dados.
    Retorna un diccionario con los datos de la ruta o con 'error'.
    """
    target = find_nearest_place(orig_lat, orig_lon, severity)
    if not target:
        return {"error": f"No se encontraron establecimientos para la gravedad '{severity}'."}

    dest_lat, dest_lon, dest_name, dest_type = target
    dest_node = nearest_graph_node(dest_lat, dest_lon)

    algo = (algorithm or "dijkstra").lower()
    blocked_segment_points = None
    blocked_segment_name = None

    try:
        # DIJKSTRA – Ruta más corta real
        if algo == "dijkstra":
            route_nodes = nx.shortest_path(graph, orig_node, dest_node, weight="weight")
            total_m = nx.shortest_path_length(graph, orig_node, dest_node, weight="weight")

        # BELLMAN–FORD – Penaliza calles lentas (según tipo_via)
        elif algo == "bellman_ford":
            sub_G = get_subgraph(graph_bellman, orig_node, dest_node)
            
            route_nodes = nx.bellman_ford_path(sub_G, orig_node, dest_node, weight="weight")
            total_m = nx.bellman_ford_path_length(sub_G, orig_node, dest_node, weight="weight")

        # UNION–FIND – Bloqueo inteligente basado en la ruta
        elif algo == "union_find":
            base_route = nx.shortest_path(graph, orig_node, dest_node, weight="weight")

            sub_G = get_subgraph(graph, orig_node, dest_node)
            blocked_graph = sub_G.copy() 

            if len(base_route) >= 2:
                # Elegimos un tramo "interno"
                if len(base_route) > 3:
                    idx = len(base_route) // 2
                    u = base_route[idx - 1]
                    v = base_route[idx]
                else:
                    u = base_route[0]
                    v = base_route[1]

                # Verificar si existen en el subgrafo recortado (por seguridad)
                if blocked_graph.has_edge(u, v):
                    data = blocked_graph.get_edge_data(u, v)
                    if isinstance(data, dict) and 0 in data:
                        edge_attr = data[0]
                    else:
                        edge_attr = data

                    blocked_segment_name = edge_attr.get("name", "Tramo crítico bloqueado")
                    blocked_segment_points = [
                        (float(blocked_graph.nodes[u]["lat"]), float(blocked_graph.nodes[u]["lon"])),
                        (float(blocked_graph.nodes[v]["lat"]), float(blocked_graph.nodes[v]["lon"])),
                    ]
                    blocked_graph.remove_edge(u, v)

            # Union–Find sobre el grafo PEQUEÑO (Rapidísimo)
            uf = nx.utils.UnionFind(blocked_graph.nodes)
            for a, b in blocked_graph.edges():
                uf.union(a, b)

            if uf[orig_node] != uf[dest_node]:
                return {
                    "error": "Ruta bloqueada: el tramo crítico está cerrado (bloqueo simulado).",
                    "blocked_segment_points": blocked_segment_points,
                    "blocked_segment_name": blocked_segment_name,
                    "severity": severity,
                    "algorithm_used": algo,
                }

            route_nodes = nx.shortest_path(blocked_graph, orig_node, dest_node, weight="weight")
            total_m = nx.shortest_path_length(blocked_graph, orig_node, dest_node, weight="weight")

        else:
            # Por defecto, Dijkstra
            route_nodes = nx.shortest_path(graph, orig_node, dest_node, weight="weight")
            total_m = nx.shortest_path_length(graph, orig_node, dest_node, weight="weight")

        distance_km = round(total_m / 1000, 2)
        avg_speed_kmh = 35
        eta_min = max(1, round((distance_km / avg_speed_kmh) * 60))
        coords = route_as_coords(graph, route_nodes)

        return {
            "hospital_name": dest_name,
            "tipo": dest_type,
            "algorithm_used": algo,
            "severity": severity,
            "distance_km": distance_km,
            "estimated_time_min": eta_min,
            "route_points": coords,
            "blocked_segment_points": blocked_segment_points,
            "blocked_segment_name": blocked_segment_name,
        }

    except nx.NetworkXNoPath:
        return {
            "error": "No se encontró una ruta disponible en el grafo.",
            "severity": severity,
            "algorithm_used": algo,
            "blocked_segment_points": blocked_segment_points,
            "blocked_segment_name": blocked_segment_name,
        }
    except Exception as e:
        return {
            "error": f"Error al calcular la ruta: {type(e).__name__}: {str(e)}",
            "severity": severity,
            "algorithm_used": algo,
            "blocked_segment_points": blocked_segment_points,
            "blocked_segment_name": blocked_segment_name,
        }


# -----------------------------
# ENDPOINT PRINCIPAL
# -----------------------------
@app.post("/route")
async def get_route(req: RouteRequest):
    try:
        # Nota: latitude = Y, longitude = X
        orig_node = nearest_graph_node(req.latitude, req.longitude)
        algo = (req.algorithm or "dijkstra").lower()
        severity = (req.severity or "").lower()

        # MODO "TODOS": leve, moderada y grave en una sola llamada
        if severity == "todos":
            severities = ["leve", "moderada", "grave"]
            routes = []
            for sev in severities:
                result = compute_route_for_severity(orig_node, req.latitude, req.longitude, sev, algo)
                if "error" not in result:
                    routes.append(result)

            if not routes:
                return {"error": "No se encontraron rutas para ningún tipo de gravedad."}

            return {
                "mode": "multi",
                "algorithm_used": algo,
                "routes": routes,
            }

        # MODO NORMAL: una sola severidad
        result = compute_route_for_severity(orig_node, req.latitude, req.longitude, severity, algo)
        if "error" in result and not result.get("route_points"):
            # Error "duro", devolvemos mensaje + posible info de bloqueo
            return {
                "error": result["error"],
                "blocked_segment_points": result.get("blocked_segment_points"),
                "blocked_segment_name": result.get("blocked_segment_name"),
            }

        return result

    except nx.NetworkXNoPath:
        return {"error": "No se encontró una ruta disponible en el grafo."}
    except Exception as e:
        return {"error": f"Error inesperado: {type(e).__name__}: {str(e)}"}
