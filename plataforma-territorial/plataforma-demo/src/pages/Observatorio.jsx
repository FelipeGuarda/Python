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
import styles from "./Observatorio.module.css";

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

const MAP_HOST_STYLE = { width: "100%", height: "100%", borderRadius: 8 };
const CARD_MAP_STYLE = { padding: 0, overflow: "hidden", position: "relative" };
const BOUNDARY_STYLE = {
  color: "#FFFFFF",
  weight: 2.5,
  opacity: 0.9,
  fillOpacity: 0,
  dashArray: "8 6",
};

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

  return (
    <div className={styles.container}>
      {/* Map Area */}
      <Card style={CARD_MAP_STYLE}>
        <MapContainer
          center={geo?.reserve?.center ?? DEFAULT_MAP_CENTER}
          zoom={geo?.reserve?.zoom ?? DEFAULT_MAP_ZOOM}
          style={MAP_HOST_STYLE}
          zoomControl={false}
        >
          <TileLayer
            attribution='Imagery &copy; <a href="https://www.esri.com/">Esri</a>'
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          />

          {boundary && <FitBounds geojson={boundary} />}

          {/* Reserve boundary */}
          {showBoundary && boundary && (
            <GeoJSON data={boundary} style={BOUNDARY_STYLE} />
          )}

          {/* Weather station */}
          {showCams && geo?.weather_station && (
            <Marker position={[geo.weather_station.lat, geo.weather_station.lon]} icon={weatherIcon}>
              <Popup>
                <div className={styles.popupContainer} style={{ minWidth: 160 }}>
                  <div className={styles.popupTitle}>
                    {geo.weather_station.name || "Estación Meteorológica"}
                  </div>
                  {geo.weather_station.model && (
                    <div className={styles.popupModel}>{geo.weather_station.model}</div>
                  )}
                  <div className={styles.popupCoords}>
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
                <div className={styles.popupContainer} style={{ minWidth: 316 }}>
                  <div className={styles.popupTitle}>
                    {st.canonical_name}
                    <span className={styles.popupTitleAside}>
                      {st.total_observations} detecciones
                    </span>
                  </div>
                  {/* Empty-state message: no identified animal detections */}
                  {st.species.length === 0 && (
                    <div className={styles.popupEmpty}>
                      Sin detecciones identificadas
                    </div>
                  )}
                  {/* Species list */}
                  {st.species.length > 0 && (
                    <div className={styles.spList}>
                      {st.species.slice(0, 5).map(sp => (
                        <div key={sp.name} className={styles.spRow}>
                          <span className={styles.spRowName}>{sp.name}</span>
                          <span className={styles.spRowCount}>{sp.count}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  {/* Thumbnail images */}
                  {st.images.length > 0 && (
                    <div className={styles.thumbsRow}>
                      {st.images.map((img, i) => (
                        <img
                          key={i}
                          src={img.url}
                          alt={img.campaign}
                          onClick={() => setLightboxImg(img.url)}
                          className={styles.thumb}
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
            className={styles.lightboxBackdrop}
          >
            <img
              src={lightboxImg}
              alt="Ampliado"
              className={styles.lightboxImg}
            />
          </div>
        )}

        {/* Legend overlay */}
        <div className={styles.legendOverlay}>
          <div className={styles.legendTitle}>CAPAS</div>
          <label className={`${styles.legendCheck} ${styles.legendCheckSpaced}`}>
            <input type="checkbox" checked={showBoundary} onChange={() => setShowBoundary(!showBoundary)} /> Límite reserva
          </label>
          <label className={styles.legendCheck}>
            <input type="checkbox" checked={showCams} onChange={() => setShowCams(!showCams)} /> Estaciones
          </label>
        </div>

        {/* Station count */}
        <div className={styles.stationCountBadge}>
          {stations ? `${stations.length} estaciones` : "Cargando..."}
        </div>
      </Card>

      {/* Right sidebar */}
      <div className={styles.sidebar}>
        <Card>
          <SectionLabel>Estado actual</SectionLabel>
          <div className={styles.sidebarTitle}>
            Bosque Pehuén
          </div>
          <div className={styles.sidebarDate}>
            {new Date().toLocaleDateString("es-CL", { day: "numeric", month: "long", year: "numeric" })}
          </div>
        </Card>
        <Card>
          <SectionLabel>Riesgo de incendio</SectionLabel>
          <RiskGauge value={riskTotal ?? 0} />
          {riskData?.rule_based?.label && (
            <div className={styles.riskLabel}>
              {riskData.rule_based.label}
            </div>
          )}
          {riskData?.timestamp && (
            <div className={styles.riskTimestamp}>
              {new Date(riskData.timestamp).toLocaleDateString("es-CL", { day: "numeric", month: "short" })}{" "}
              {new Date(riskData.timestamp).toLocaleTimeString("es-CL", { hour: "2-digit", minute: "2-digit" })}
            </div>
          )}
        </Card>
        <Card>
          <SectionLabel>Meteorología</SectionLabel>
          <div className={styles.statGrid}>
            <StatBlock value={wx.temperature_c != null ? wx.temperature_c.toFixed(1) : "—"} unit="°C" label="Temperatura" />
            <StatBlock value={wx.relative_humidity_pct != null ? Math.round(wx.relative_humidity_pct) : "—"} unit="%" label="Humedad" />
            <StatBlock value={wx.wind_speed_kmh != null ? wx.wind_speed_kmh.toFixed(0) : "—"} unit="km/h" label="Viento" />
            <StatBlock value={wx.days_without_rain ?? "—"} unit="días" label="Sin lluvia" color={C.amber} />
          </div>
        </Card>
        <Card>
          <SectionLabel>Resumen estaciones</SectionLabel>
          <div className={styles.statGridTight}>
            <StatBlock value={stations ? stations.length : "..."} label="Cámaras trampa" />
            <StatBlock value="1" label="Estación meteo" />
          </div>
        </Card>
      </div>
    </div>
  );
}
