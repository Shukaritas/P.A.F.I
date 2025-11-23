from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Tuple, Optional, List, Dict, Any
from geopy.distance import geodesic
import osmnx as ox
import networkx as nx

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
# CARGA DEL GRAFO (OSMnx)
# -----------------------------
print("Cargando grafo de San Miguel, Lima...")
def load_osmnx_graph(place_name: str = "San Miguel, Lima, Peru"):
    G = ox.graph_from_place(place_name, network_type="drive")
    return G

graph = load_osmnx_graph("San Miguel, Lima, Peru")
print(f"Grafo cargado: {len(graph.nodes)} nodos, {len(graph.edges)} aristas.")


# -----------------------------
# HELPERS
# -----------------------------
def tags_by_severity(severity: str) -> Dict[str, List[str]]:
    s = (severity or "").lower()
    if s == "grave":
        return {"amenity": ["hospital"]}
    elif s == "moderada":
        return {"amenity": ["hospital", "clinic"]}
    elif s == "leve":
        return {"amenity": ["clinic", "doctors", "pharmacy", "health_center"]}
    elif s == "todos":
        # Usado solo si alguna vez se llama directo con "todos"
        return {"amenity": ["hospital", "clinic", "doctors", "pharmacy", "health_center"]}
    else:
        # Por defecto, usar todo
        return {"amenity": ["hospital", "clinic", "doctors", "pharmacy", "health_center"]}


def find_nearest_place(
    lat: float,
    lon: float,
    severity: str,
    place_name: str = "San Miguel, Lima, Peru"
) -> Optional[Tuple[float, float, str, str]]:
    """
    Busca el establecimiento más cercano según la gravedad.
    Retorna (lat, lon, nombre, tipo_amenity) o None si no encuentra.
    """
    tags = tags_by_severity(severity)
    try:
        pois = ox.features_from_place(place_name, tags=tags)
    except Exception:
        return None

    if pois is None or len(pois) == 0:
        return None

    min_place = None
    min_dist = float("inf")
    for _, row in pois.iterrows():
        try:
            geom = row.geometry
            c = geom.centroid
            plat, plon = c.y, c.x
            d = geodesic((lat, lon), (plat, plon)).km
            if d < min_dist:
                min_dist = d
                min_place = (
                    plat,
                    plon,
                    row.get("name", "Centro sin nombre"),
                    row.get("amenity", "desconocido"),
                )
        except Exception:
            continue

    return min_place


def route_as_coords(G: nx.MultiDiGraph, route_nodes: List[int]) -> List[Tuple[float, float]]:
    """Convierte nodos del grafo a lista [(lat, lon), ...] para Leaflet."""
    return [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route_nodes]


# -----------------------------
# LÓGICA POR ALGORITMO
# -----------------------------
def compute_route_for_severity(
    orig_node: int,
    orig_lat: float,
    orig_lon: float,
    severity: str,
    algorithm: str
) -> Dict[str, Any]:
    """
    Calcula una ruta para una severidad y un algoritmo dados.
    Retorna un diccionario con los datos de la ruta o con 'error'.
    """
    target = find_nearest_place(orig_lat, orig_lon, severity, "San Miguel, Lima, Peru")
    if not target:
        return {"error": f"No se encontraron establecimientos para la gravedad '{severity}'."}

    dest_lat, dest_lon, dest_name, dest_type = target
    dest_node = ox.distance.nearest_nodes(graph, dest_lon, dest_lat)

    algo = (algorithm or "dijkstra").lower()
    blocked_segment_points = None
    blocked_segment_name = None

    try:
        # 1️⃣ DIJKSTRA – Ruta más corta real
        if algo == "dijkstra":
            route_nodes = nx.shortest_path(graph, orig_node, dest_node, weight="length")
            total_m = nx.shortest_path_length(graph, orig_node, dest_node, weight="length")

        # 2️⃣ BELLMAN–FORD – Penaliza calles lentas
        elif algo == "bellman_ford":
            penalized = graph.copy()
            for u, v, k, data in penalized.edges(keys=True, data=True):
                L = data.get("length", 0.0)
                highway = data.get("highway", "")
                if isinstance(highway, list):
                    highway = highway[0]

                # Penaliza calles internas / residenciales
                if highway in ["residential", "service", "living_street"]:
                    data["length"] = L * 2.5
                # Favorece vías principales
                elif highway in ["primary", "secondary"]:
                    data["length"] = L * 0.8

            route_nodes = nx.bellman_ford_path(penalized, orig_node, dest_node, weight="length")
            total_m = nx.bellman_ford_path_length(penalized, orig_node, dest_node, weight="length")

        # 3️⃣ UNION–FIND – Bloqueo inteligente basado en la ruta
        elif algo == "union_find":
            # a) Ruta base con Dijkstra en el grafo normal
            base_route = nx.shortest_path(graph, orig_node, dest_node, weight="length")

            blocked_graph = graph.copy()
            if len(base_route) >= 2:
                # Elegimos un tramo "interno" de la ruta (no el primero ni el último, cuando sea posible)
                if len(base_route) > 3:
                    idx = len(base_route) // 2
                    u = base_route[idx - 1]
                    v = base_route[idx]
                else:
                    u = base_route[0]
                    v = base_route[1]

                edge_data = blocked_graph.get_edge_data(u, v, default=None)
                if edge_data:
                    # Tomamos el primer sub-arco entre u y v
                    first_key = next(iter(edge_data))
                    data = edge_data[first_key]
                    blocked_segment_name = data.get("name", "Tramo crítico bloqueado")

                    blocked_segment_points = [
                        (blocked_graph.nodes[u]["y"], blocked_graph.nodes[u]["x"]),
                        (blocked_graph.nodes[v]["y"], blocked_graph.nodes[v]["x"]),
                    ]

                    # Removemos todos los arcos entre u y v (bloqueo de ese tramo)
                    for key in list(edge_data.keys()):
                        if blocked_graph.has_edge(u, v, key):
                            blocked_graph.remove_edge(u, v, key=key)

            # Union–Find sobre el grafo bloqueado
            uf = nx.utils.UnionFind(blocked_graph.nodes)
            for a, b, _ in blocked_graph.edges(keys=True):
                uf.union(a, b)

            if uf[orig_node] != uf[dest_node]:
                # No hay conexión posible con el tramo bloqueado
                return {
                    "error": "Ruta bloqueada: el tramo crítico está cerrado (bloqueo simulado).",
                    "blocked_segment_points": blocked_segment_points,
                    "blocked_segment_name": blocked_segment_name,
                    "severity": severity,
                    "algorithm_used": algo,
                }

            # Si aún hay conexión, buscamos la ruta alternativa
            route_nodes = nx.shortest_path(blocked_graph, orig_node, dest_node, weight="length")
            total_m = nx.shortest_path_length(blocked_graph, orig_node, dest_node, weight="length")

        else:
            # Por defecto, Dijkstra
            route_nodes = nx.shortest_path(graph, orig_node, dest_node, weight="length")
            total_m = nx.shortest_path_length(graph, orig_node, dest_node, weight="length")

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
        orig_node = ox.distance.nearest_nodes(graph, req.longitude, req.latitude)
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
