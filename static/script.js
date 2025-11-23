// Inicializar mapa
const map = L.map('map').setView([-12.06, -77.08], 13);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

let marker = null;
let routeLayers = [];
let blockedLayer = null;

const infoEl = document.getElementById('info');
const severitySelect = document.getElementById('severity');
const algorithmSelect = document.getElementById('algorithm');
const findRouteBtn = document.getElementById('findRouteBtn');

// Helpers
function clearLayers() {
  routeLayers.forEach(layer => map.removeLayer(layer));
  routeLayers = [];
  if (blockedLayer) {
    map.removeLayer(blockedLayer);
    blockedLayer = null;
  }
}

function getColorForAlgorithm(algorithm) {
  switch (algorithm) {
    case 'bellman_ford':
      return '#ef4444'; // rojo
    case 'union_find':
      return '#f97316'; // naranja
    default:
      return '#3b82f6'; // azul
  }
}

function getColorForSeverity(severity) {
  switch (severity) {
    case 'leve':
      return '#22c55e'; // verde
    case 'moderada':
      return '#f97316'; // naranja
    case 'grave':
      return '#ef4444'; // rojo
    default:
      return '#3b82f6';
  }
}

function drawBlockedSegment(blockedPoints, name) {
  if (!blockedPoints || blockedPoints.length !== 2) return;

  const latlngs = blockedPoints.map(p => [p[0], p[1]]);
  blockedLayer = L.polyline(latlngs, {
    color: '#dc2626',
    weight: 7,
    opacity: 0.9,
    dashArray: '6, 6'
  }).addTo(map);
}

map.on('click', (e) => {
  if (marker) map.removeLayer(marker);
  marker = L.marker(e.latlng).addTo(map);

  infoEl.innerHTML = `
    Ubicación seleccionada:<br>
    <b>Lat:</b> ${e.latlng.lat.toFixed(5)}<br>
    <b>Lon:</b> ${e.latlng.lng.toFixed(5)}<br>
    Ahora selecciona gravedad y algoritmo y presiona <b>Calcular ruta</b>.
  `;
});

findRouteBtn.addEventListener('click', async () => {
  if (!marker) {
    alert('Haz clic en el mapa para elegir una ubicación.');
    return;
  }

  const severity = severitySelect.value;
  const algorithm = algorithmSelect.value;
  const coords = marker.getLatLng();

  clearLayers();

  infoEl.innerHTML = `<i>Calculando ruta con <b>${algorithm}</b>...</i>`;

  try {
    const res = await fetch('/route', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        latitude: coords.lat,
        longitude: coords.lng,
        severity: severity,
        algorithm: algorithm
      })
    });

    const data = await res.json();

    // Manejo de errores globales
    if (data.error && !data.routes) {
      infoEl.innerHTML = `<b style="color:#dc2626;">${data.error}</b>`;
      if (data.blocked_segment_points) {
        drawBlockedSegment(data.blocked_segment_points, data.blocked_segment_name);
      }
      return;
    }

    // MODO MULTI (severity = "todos")
    if (data.routes && Array.isArray(data.routes)) {
      const allLatLngs = [];
      let html = `<b>Algoritmo:</b> ${data.algorithm_used}<br><br>`;
      html += `<b>Rutas por gravedad:</b><ul class="info-list">`;

      data.routes.forEach(route => {
        const color = getColorForSeverity(route.severity);
        const latlngs = route.route_points.map(p => [p[0], p[1]]);
        const poly = L.polyline(latlngs, {
          color,
          weight: 5,
          opacity: 0.9
        }).addTo(map);
        
        if (route.route_points && route.route_points.length > 0) {
          const destinationPoint = route.route_points[route.route_points.length - 1];
          const destLatLng = [destinationPoint[0], destinationPoint[1]];
          
          const destMarker = L.circleMarker(destLatLng, {
            radius: 8,
            color: color,
            fillColor: color,
            fillOpacity: 1,
            weight: 2
          }).bindTooltip(route.hospital_name, { permanent: false, direction: 'right' }).addTo(map);

          routeLayers.push(destMarker);
        }
        

        routeLayers.push(poly);
        allLatLngs.push(...latlngs);

        html += `
          <li>
            <span class="pill pill-${route.severity}">${route.severity}</span>
            ${route.hospital_name} – ${route.distance_km} km (~${route.estimated_time_min} min)
          </li>
        `;
      });

      html += `</ul>`;
      infoEl.innerHTML = html;

      if (allLatLngs.length > 0) {
        map.fitBounds(allLatLngs);
      }

      // Si alguna ruta incluye un tramo bloqueado (Union–Find), lo dibujamos:
      const blockedRoute = data.routes.find(r => r.blocked_segment_points && r.blocked_segment_points.length === 2);
      if (blockedRoute) {
        drawBlockedSegment(blockedRoute.blocked_segment_points, blockedRoute.blocked_segment_name);
        infoEl.innerHTML += `<br><br><b>Bloqueo simulado:</b> ${blockedRoute.blocked_segment_name || 'Tramo crítico de la ruta'}`;
      }

      return;
    }

    // MODO NORMAL (una sola severidad)
    const color = getColorForAlgorithm(data.algorithm_used || algorithm);
    const latlngs = data.route_points.map(p => [p[0], p[1]]);
    const poly = L.polyline(latlngs, {
      color,
      weight: 5,
      opacity: 0.9
    }).addTo(map);

    if (data.route_points && data.route_points.length > 0) {
      const destinationPoint = data.route_points[data.route_points.length - 1];
      const destLatLng = [destinationPoint[0], destinationPoint[1]];
      
      const destMarker = L.circleMarker(destLatLng, {
        radius: 8,
        color: color,
        fillColor: color,
        fillOpacity: 1,
        weight: 2
      }).bindTooltip(data.hospital_name, { permanent: false, direction: 'right' }).addTo(map);

      routeLayers.push(destMarker);
    }

    routeLayers.push(poly);
    map.fitBounds(poly.getBounds());

    let html = `
      <b>Destino:</b> ${data.hospital_name}<br>
      <b>Tipo:</b> ${data.tipo || 'No especificado'}<br>
      <b>Algoritmo:</b> ${data.algorithm_used || algorithm}<br>
      <b>Distancia:</b> ${data.distance_km} km<br>
      <b>Tiempo estimado:</b> ${data.estimated_time_min} min
    `;

    if (data.blocked_segment_points && data.blocked_segment_points.length === 2) {
      drawBlockedSegment(data.blocked_segment_points, data.blocked_segment_name);
      html += `<br><b>Bloqueo simulado:</b> ${data.blocked_segment_name || 'Tramo crítico de la ruta'}`;
    }

    infoEl.innerHTML = html;
  } catch (err) {
    infoEl.innerHTML = `<b style="color:#dc2626;">Error de conexión o cálculo: ${err}</b>`;
  }
});