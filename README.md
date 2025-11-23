# 🚑 P.A.F.I – Plataforma de Asistencia Frente a Incidencias

Aplicación desarrollada con **FastAPI**, **Leaflet** y **OSMnx** para calcular rutas óptimas hacia centros de salud según la gravedad del accidente y utilizando distintos algoritmos de rutas.

Incluye:
- Algoritmo **Dijkstra** (ruta más rápida real)
- Algoritmo **Bellman–Ford** (penalización de calles lentas)
- Algoritmo **Union–Find** (detección de rutas bloqueadas)

---

## 📦 Requisitos

- **Windows 10/11**
- **Python 3.11** (exactamente 3.11, no 3.12 ni 3.14)
- Conexión a internet (la primera vez OSMnx descarga datos del mapa)

---

## 🚀 Ejecutar la aplicación (Método recomendado)

Solo debes ejecutar en consola:

```bash
run.bat
```

Una vez iniciado, abre en tu navegador:

'http://127.0.0.1:8000'

---

## ⚙️ Ejecutar manualmente (opcional)

Si deseas correr el proyecto sin el `run.bat`:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
