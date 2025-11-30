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

class RouteRequest(BaseModel):
    latitude: float
    longitude: float
    severity: str              
    algorithm: str = "dijkstra" 

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index():
    return FileResponse("static/index.html")

print("Cargando grafo de Lima y Callao desde CSV...")

graph, hospitals_df = load_graph(base_path="app/data")

nodes_df = pd.read_csv("app/data/nodos_lima_callao.csv", dtype={"node_id": str})
NODE_IDS = nodes_df["node_id"].values
NODE_LATS = nodes_df["lat"].values
NODE_LONS = nodes_df["lon"].values

print(f"Grafo listo: {len(graph.nodes)} nodos, {len(graph.edges)} aristas.")

ALL_TYPES = sorted(set(hospitals_df["Tipo"].astype(str)))


def build_severity_types() -> Dict[str, List[str]]:
    grave = []
    moderada_extra = []
    for t in ALL_TYPES:
        tu = t.upper().replace("Í", "I")
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

def nearest_graph_node(lat: float, lon: float) -> str:
    """
    Devuelve el id del nodo vial más cercano (en todo Lima + Callao),
    usando una búsqueda vectorizada en NumPy (rápida incluso con ~200k nodos).
    """
    dlat = NODE_LATS - lat
    dlon = NODE_LONS - lon
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
        coords.append((float(data["lat"]), float(data["lon"])))
    return coords

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
        if algo == "dijkstra":
            route_nodes = nx.shortest_path(graph, orig_node, dest_node, weight="weight")
            total_m = nx.shortest_path_length(graph, orig_node, dest_node, weight="weight")

        elif algo == "bellman_ford":
            penalized = graph.copy()
            for u, v, data in penalized.edges(data=True):
                L = float(data.get("weight", 0.0))
                tipo_via = str(data.get("tipo_via", "")).lower()

                if tipo_via in ["residential", "service", "living_street"]:
                    data["weight"] = L * 2.5
                elif tipo_via in ["primary", "secondary", "trunk", "motorway"]:
                    data["weight"] = L * 0.8

            route_nodes = nx.bellman_ford_path(penalized, orig_node, dest_node, weight="weight")
            total_m = nx.bellman_ford_path_length(penalized, orig_node, dest_node, weight="weight")

        elif algo == "union_find":
            base_route = nx.shortest_path(graph, orig_node, dest_node, weight="weight")

            blocked_graph = graph.copy()
            if len(base_route) >= 2:
                if len(base_route) > 3:
                    idx = len(base_route) // 2
                    u = base_route[idx - 1]
                    v = base_route[idx]
                else:
                    u = base_route[0]
                    v = base_route[1]

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


@app.post("/route")
async def get_route(req: RouteRequest):
    try:
        orig_node = nearest_graph_node(req.latitude, req.longitude)
        algo = (req.algorithm or "dijkstra").lower()
        severity = (req.severity or "").lower()

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

        result = compute_route_for_severity(orig_node, req.latitude, req.longitude, severity, algo)
        if "error" in result and not result.get("route_points"):
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
