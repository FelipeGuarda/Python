import { MapContainer, TileLayer, GeoJSON, CircleMarker, Popup } from "react-leaflet";
import { C, SP_COLORS } from "../constants/colors.js";
import { DEFAULT_MAP_CENTER } from "../constants/map_defaults.js";
import { FitBounds } from "./FitBounds.jsx";

// ── Mini Leaflet map for per-species detection bubble maps ──
export function SpeciesMap({ boundary, stations, colorIdx, center }) {
  const color = SP_COLORS[colorIdx] ?? SP_COLORS[0];
  const maxCount = stations ? Math.max(...stations.map(s => s.count), 1) : 1;
  return (
    <MapContainer
      center={center ?? DEFAULT_MAP_CENTER}
      zoom={13}
      style={{ width: "100%", height: "100%", borderRadius: 8 }}
      zoomControl={false}
      scrollWheelZoom={false}
    >
      <TileLayer
        attribution='Imagery &copy; <a href="https://www.esri.com/">Esri</a>'
        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
      />
      {boundary && <FitBounds geojson={boundary} />}
      {boundary && (
        <GeoJSON data={boundary} style={{ color: "#FFFFFF", weight: 2.5, fillOpacity: 0, dashArray: "8 6" }} />
      )}
      {(stations || []).map(st =>
        st.count === 0 ? (
          <CircleMarker key={st.tc} center={[st.lat, st.lon]}
            radius={3}
            pathOptions={{ color: "#ffffff60", weight: 1, fillColor: "#888888", fillOpacity: 0.25 }}
          />
        ) : (
          <CircleMarker key={st.tc} center={[st.lat, st.lon]}
            radius={4 + Math.sqrt(st.count / maxCount) * 10}
            pathOptions={{ color: "#ffffff", weight: 1.5, fillColor: color, fillOpacity: 0.85 }}
          >
            <Popup>
              <div style={{ fontFamily: "'Trebuchet MS', sans-serif", fontSize: 12, color: C.text }}>
                <strong>TC{String(st.tc).padStart(2, "0")}</strong>: {st.count} detecciones
              </div>
            </Popup>
          </CircleMarker>
        )
      )}
    </MapContainer>
  );
}
