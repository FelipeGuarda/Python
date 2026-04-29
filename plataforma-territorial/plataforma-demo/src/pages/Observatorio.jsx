import { useState, useEffect } from "react";
import { MapContainer, TileLayer, GeoJSON, CircleMarker, Marker, Popup } from "react-leaflet";
import L from "leaflet";
import { getFireRiskCurrent, getStationSummary, getGeography } from "../api.js";
import { C } from "../constants/colors.js";
import { DEFAULT_MAP_CENTER, DEFAULT_MAP_ZOOM } from "../constants/map_defaults.js";
import { useAPI } from "../hooks/useAPI.js";
import { FitBounds } from "../components/FitBounds.jsx";
import { Card } from "../components/Card.jsx";
import { SectionLabel } from "../components/SectionLabel.jsx";
import { RiskGauge } from "../components/RiskGauge.jsx";
import { StatBlock } from "../components/StatBlock.jsx";

// Custom weather station icon
const weatherIcon = L.divIcon({
  className: "",
  html: `<div style="width:28px;height:28px;background:${C.amber};border:2px solid white;border-radius:50%;display:flex;align-items:center;justify-content:center;box-shadow:0 2px 6px rgba(0,0,0,0.3)">
    <span style="color:white;font-weight:bold;font-size:11px">EM</span>
  </div>`,
  iconSize: [28, 28],
  iconAnchor: [14, 14],
  popupAnchor: [0, -16],
});

export function Observatorio() {
  const [boundary, setBoundary] = useState(null);
  const [showBoundary, setShowBoundary] = useState(true);
  const [showCams, setShowCams] = useState(true);
  const [lightboxImg, setLightboxImg] = useState(null);
  const { data: riskData } = useAPI(getFireRiskCurrent, null, []);
  const { data: stations } = useAPI(getStationSummary, null, []);
  const { data: geo } = useAPI(getGeography, null, []);

  const riskTotal = riskData?.rule_based?.total ? Math.round(riskData.rule_based.total) : null;
  const wx = riskData?.weather || {};

  useEffect(() => {
    fetch("/data/boundary.geojson").then(r => r.json()).then(setBoundary);
  }, []);

  const boundaryStyle = {
    color: "#FFFFFF",
    weight: 2.5,
    opacity: 0.9,
    fillOpacity: 0,
    dashArray: "8 6",
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 300px", gap: 16, height: "calc(100vh - 56px)", padding: 16 }}>
      {/* Map Area */}
      <Card style={{ padding: 0, overflow: "hidden", position: "relative" }}>
        <MapContainer
          center={geo?.reserve?.center ?? DEFAULT_MAP_CENTER}
          zoom={geo?.reserve?.zoom ?? DEFAULT_MAP_ZOOM}
          style={{ width: "100%", height: "100%", borderRadius: 8 }}
          zoomControl={false}
        >
          <TileLayer
            attribution='Imagery &copy; <a href="https://www.esri.com/">Esri</a>'
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          />

          {boundary && <FitBounds geojson={boundary} />}

          {/* Reserve boundary */}
          {showBoundary && boundary && (
            <GeoJSON data={boundary} style={boundaryStyle} />
          )}

          {/* Weather station */}
          {showCams && geo?.weather_station && (
            <Marker position={[geo.weather_station.lat, geo.weather_station.lon]} icon={weatherIcon}>
              <Popup>
                <div style={{ fontFamily: "'Trebuchet MS', sans-serif", minWidth: 160 }}>
                  <div style={{ fontWeight: 700, fontSize: 13, color: C.text, marginBottom: 6 }}>
                    {geo.weather_station.name || "Estación Meteorológica"}
                  </div>
                  {geo.weather_station.model && (
                    <div style={{ fontSize: 11, color: C.muted }}>{geo.weather_station.model}</div>
                  )}
                  <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>
                    {geo.weather_station.lat.toFixed(6)}, {geo.weather_station.lon.toFixed(6)}
                  </div>
                </div>
              </Popup>
            </Marker>
          )}

          {/* Camera trap stations */}
          {showCams && stations && stations.map(st => (
            <CircleMarker
              key={st.canonical_name}
              center={[st.latitude, st.longitude]}
              radius={7}
              pathOptions={{
                color: C.white,
                weight: 2,
                fillColor: st.total_observations > 0 ? C.deepGreen : C.muted,
                fillOpacity: st.total_observations > 0 ? 0.9 : 0.55,
              }}
            >
              <Popup maxWidth={360}>
                <div style={{ fontFamily: "'Trebuchet MS', sans-serif", minWidth: 316 }}>
                  <div style={{ fontWeight: 700, fontSize: 13, color: C.text, marginBottom: 6 }}>
                    {st.canonical_name}
                    <span style={{ fontWeight: 400, color: C.muted, fontSize: 10, marginLeft: 6 }}>
                      {st.total_observations} detecciones
                    </span>
                  </div>
                  {/* Empty-state message: no identified animal detections */}
                  {st.species.length === 0 && (
                    <div style={{ fontSize: 11, color: C.muted, fontStyle: "italic", padding: "4px 0" }}>
                      Sin detecciones identificadas
                    </div>
                  )}
                  {/* Species list */}
                  {st.species.length > 0 && (
                    <div style={{ marginBottom: 8 }}>
                      {st.species.slice(0, 5).map(sp => (
                        <div key={sp.name} style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: C.muted, padding: "2px 0", borderBottom: `1px solid ${C.paleMint}` }}>
                          <span style={{ color: C.text }}>{sp.name}</span>
                          <span style={{ fontWeight: 600 }}>{sp.count}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  {/* Thumbnail images */}
                  {st.images.length > 0 && (
                    <div style={{ display: "flex", gap: 6, marginTop: 6, flexWrap: "wrap" }}>
                      {st.images.map((img, i) => (
                        <img
                          key={i}
                          src={img.url}
                          alt={img.campaign}
                          onClick={() => setLightboxImg(img.url)}
                          style={{ width: 250, height: 188, objectFit: "cover", borderRadius: 4, border: `1px solid ${C.mint}`, flexShrink: 0, cursor: "pointer" }}
                        />
                      ))}
                    </div>
                  )}
                </div>
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>

        {/* Lightbox */}
        {lightboxImg && (
          <div
            onClick={() => setLightboxImg(null)}
            style={{ position: "fixed", inset: 0, zIndex: 9999, background: "rgba(0,0,0,0.8)", display: "flex", alignItems: "center", justifyContent: "center", cursor: "zoom-out" }}
          >
            <img
              src={lightboxImg}
              alt="Ampliado"
              style={{ maxWidth: "90vw", maxHeight: "90vh", borderRadius: 6, boxShadow: "0 8px 40px rgba(0,0,0,0.6)" }}
            />
          </div>
        )}

        {/* Legend overlay */}
        <div style={{ position: "absolute", bottom: 12, left: 12, zIndex: 1000, background: "rgba(255,255,255,0.92)", borderRadius: 6, padding: "10px 14px", fontSize: 11 }}>
          <div style={{ fontWeight: 700, color: C.text, marginBottom: 6, fontSize: 10, letterSpacing: 1 }}>CAPAS</div>
          <label style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4, cursor: "pointer", color: C.text }}>
            <input type="checkbox" checked={showBoundary} onChange={() => setShowBoundary(!showBoundary)} /> Límite reserva
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer", color: C.text }}>
            <input type="checkbox" checked={showCams} onChange={() => setShowCams(!showCams)} /> Estaciones
          </label>
        </div>

        {/* Station count */}
        <div style={{ position: "absolute", top: 12, right: 12, zIndex: 1000, background: "rgba(255,255,255,0.92)", borderRadius: 6, padding: "8px 12px", fontSize: 11, color: C.muted }}>
          {stations ? `${stations.length} estaciones` : "Cargando..."}
        </div>
      </Card>

      {/* Right sidebar */}
      <div style={{ display: "flex", flexDirection: "column", gap: 12, overflowY: "auto" }}>
        <Card>
          <SectionLabel>Estado actual</SectionLabel>
          <div style={{ fontFamily: "'Georgia', serif", fontSize: 15, fontWeight: 700, color: C.text, marginBottom: 10 }}>
            Bosque Pehuén
          </div>
          <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.5 }}>
            {new Date().toLocaleDateString("es-CL", { day: "numeric", month: "long", year: "numeric" })}
          </div>
        </Card>
        <Card>
          <SectionLabel>Riesgo de incendio</SectionLabel>
          <RiskGauge value={riskTotal ?? 0} />
          {riskData?.rule_based?.label && (
            <div style={{ fontSize: 11, color: C.muted, textAlign: "center", marginTop: 4 }}>
              {riskData.rule_based.label}
            </div>
          )}
          {riskData?.timestamp && (
            <div style={{ fontSize: 10, color: C.lightMuted, textAlign: "center", marginTop: 6 }}>
              {new Date(riskData.timestamp).toLocaleDateString("es-CL", { day: "numeric", month: "short" })}{" "}
              {new Date(riskData.timestamp).toLocaleTimeString("es-CL", { hour: "2-digit", minute: "2-digit" })}
            </div>
          )}
        </Card>
        <Card>
          <SectionLabel>Meteorología</SectionLabel>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 8 }}>
            <StatBlock value={wx.temperature_c != null ? wx.temperature_c.toFixed(1) : "—"} unit="°C" label="Temperatura" />
            <StatBlock value={wx.relative_humidity_pct != null ? Math.round(wx.relative_humidity_pct) : "—"} unit="%" label="Humedad" />
            <StatBlock value={wx.wind_speed_kmh != null ? wx.wind_speed_kmh.toFixed(0) : "—"} unit="km/h" label="Viento" />
            <StatBlock value={wx.days_without_rain ?? "—"} unit="días" label="Sin lluvia" color={C.amber} />
          </div>
        </Card>
        <Card>
          <SectionLabel>Resumen estaciones</SectionLabel>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 8 }}>
            <StatBlock value={stations ? stations.length : "..."} label="Cámaras trampa" />
            <StatBlock value="1" label="Estación meteo" />
          </div>
        </Card>
      </div>
    </div>
  );
}
