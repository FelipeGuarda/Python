import { useState, useEffect, useRef, Component } from "react";
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadialBarChart, RadialBar, PieChart, Pie, Cell } from "recharts";
import { MapContainer, TileLayer, GeoJSON, CircleMarker, Marker, Popup, useMap } from "react-leaflet";
import L from "leaflet";

// ── Color System (designer's green palette) ──
const C = {
  deepGreen: "#004D3C",
  medGreen: "#006B54",
  lightGreen: "#5A9A7A",
  mint: "#D4E8D9",
  paleMint: "#E8F0EB",
  bg: "#F2F7F4",
  white: "#FFFFFF",
  text: "#004D3C",
  muted: "#5A7D6E",
  lightMuted: "#9BB5A8",
  amber: "#D4913B",
  red: "#C4573A",
  warmSand: "#E8DFD0",
};

// ── Error Boundary ──
class ErrorBoundary extends Component {
  constructor(props) { super(props); this.state = { error: null }; }
  static getDerivedStateFromError(error) { return { error }; }
  render() {
    if (this.state.error) return (
      <div style={{ padding: 40, color: C.red, fontFamily: "monospace", fontSize: 13 }}>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>Error al cargar la vista</div>
        <div style={{ color: C.muted }}>{this.state.error.message}</div>
        <button onClick={() => this.setState({ error: null })}
          style={{ marginTop: 16, padding: "8px 16px", background: C.deepGreen, color: C.white, border: "none", borderRadius: 6, cursor: "pointer" }}>
          Reintentar
        </button>
      </div>
    );
    return this.props.children;
  }
}

// ── Mock Data ──

const weeklyRisk = [
  { dia: "Lun", riesgo: 32, temp: 18, humedad: 55 },
  { dia: "Mar", riesgo: 28, temp: 16, humedad: 62 },
  { dia: "Mie", riesgo: 45, temp: 22, humedad: 38 },
  { dia: "Jue", riesgo: 67, temp: 26, humedad: 28 },
  { dia: "Vie", riesgo: 58, temp: 24, humedad: 32 },
  { dia: "Sab", riesgo: 42, temp: 20, humedad: 48 },
  { dia: "Dom", riesgo: 35, temp: 17, humedad: 58 },
];

const speciesData = [
  { nombre: "Raton cola larga", detecciones: 487, tendencia: "estable" },
  { nombre: "Zorro culpeo", detecciones: 156, tendencia: "alza" },
  { nombre: "Jabali", detecciones: 134, tendencia: "alza" },
  { nombre: "Liebre europea", detecciones: 98, tendencia: "estable" },
  { nombre: "Chingue", detecciones: 67, tendencia: "baja" },
  { nombre: "Guina", detecciones: 23, tendencia: "estable" },
  { nombre: "Puma", detecciones: 8, tendencia: "estable" },
  { nombre: "Monito del monte", detecciones: 5, tendencia: "baja" },
];

const activityByHour = Array.from({ length: 24 }, (_, i) => ({
  hora: i,
  actividad: i >= 20 || i <= 4 ? 15 + Math.random() * 25 : 2 + Math.random() * 5,
}));

const stations = [
  { id: 1, name: "Estacion Meteorologica", type: "weather", x: 45, y: 35, temp: "14.2°C", hum: "62%", wind: "12 km/h" },
  { id: 2, name: "Camara Araucaria Norte", type: "camera", x: 30, y: 25, lastDetection: "Zorro culpeo", time: "Hace 3h" },
  { id: 3, name: "Camara Rio Turbio", type: "camera", x: 62, y: 55, lastDetection: "Jabali", time: "Hace 45min" },
  { id: 4, name: "Camara Sendero Pehuen", type: "camera", x: 38, y: 60, lastDetection: "Guina", time: "Hace 12h" },
  { id: 5, name: "Camara Laguna Sur", type: "camera", x: 55, y: 40, lastDetection: "Puma", time: "Hace 2d" },
  { id: 6, name: "Camara Bosque Antiguo", type: "camera", x: 25, y: 45, lastDetection: "Chingue", time: "Hace 6h" },
];

const fireZones = [
  { cx: 50, cy: 30, r: 18, risk: "alto" },
  { cx: 30, cy: 50, r: 14, risk: "medio" },
  { cx: 65, cy: 60, r: 12, risk: "bajo" },
];

const chatMessages = [
  { role: "user", text: "Cual es el riesgo de incendio hoy?" },
  { role: "assistant", text: "El indice de riesgo de incendio para hoy en Bosque Pehuen es de 67/100 (ALTO). Este valor se calcula con la formula FRI = 0.35*T + 0.25*(100-H) + 0.20*V + 0.20*D, donde T=26°C, H=28%, V=18 km/h, y D=12 dias sin lluvia. El factor principal hoy es la baja humedad relativa." },
  { role: "user", text: "Que especies se han detectado esta semana?" },
  { role: "assistant", text: "Esta semana se registraron 47 detecciones en las 5 camaras trampa activas. Las especies mas frecuentes fueron: Raton cola larga (18), Zorro culpeo (9), Jabali (8), Liebre europea (5). Destaca 1 deteccion de Puma en la Camara Laguna Sur hace 2 dias, lo que es consistente con su patron de movimiento estacional." },
];

// ── Reusable Components ──
function NavBar({ page, setPage }) {
  const pages = [
    { id: "observatorio", label: "Observatorio", icon: "M" },
    { id: "dashboard", label: "Dashboard", icon: "D" },
    { id: "asistente", label: "Asistente", icon: "A" },
    { id: "reportes", label: "Reportes", icon: "R" },
  ];
  return (
    <div style={{ background: C.deepGreen, padding: "0 24px", display: "flex", alignItems: "center", height: 56, gap: 0 }}>
      <div style={{ fontFamily: "'Georgia', serif", color: C.white, fontSize: 15, fontWeight: 700, marginRight: 40, letterSpacing: -0.3 }}>
        Plataforma Territorial <span style={{ color: C.lightGreen, fontWeight: 400, fontSize: 12 }}>FMA</span>
      </div>
      <div style={{ display: "flex", gap: 0, flex: 1 }}>
        {pages.map(p => (
          <button key={p.id} onClick={() => setPage(p.id)} style={{
            background: page === p.id ? "rgba(255,255,255,0.12)" : "transparent",
            border: "none", color: page === p.id ? C.white : C.lightGreen,
            padding: "16px 20px", cursor: "pointer", fontSize: 13,
            fontFamily: "'Trebuchet MS', sans-serif", fontWeight: page === p.id ? 700 : 400,
            borderBottom: page === p.id ? `2px solid ${C.white}` : "2px solid transparent",
            transition: "all 0.2s",
          }}>{p.label}</button>
        ))}
      </div>
      <div style={{ color: C.lightMuted, fontSize: 11, fontFamily: "monospace", letterSpacing: 1 }}>
        BOSQUE PEHUEN
      </div>
    </div>
  );
}

function Card({ children, style = {} }) {
  return (
    <div style={{ background: C.white, borderRadius: 8, padding: 20, boxShadow: "0 1px 4px rgba(0,77,60,0.06)", ...style }}>
      {children}
    </div>
  );
}

function SectionLabel({ children }) {
  return (
    <div style={{ fontSize: 10, fontFamily: "monospace", color: C.lightMuted, letterSpacing: 3, textTransform: "uppercase", marginBottom: 6 }}>
      {children}
    </div>
  );
}

function StatBlock({ value, label, unit = "", color = C.text }) {
  return (
    <div style={{ textAlign: "center" }}>
      <div style={{ fontSize: 32, fontWeight: 700, fontFamily: "'Georgia', serif", color, lineHeight: 1 }}>
        {value}<span style={{ fontSize: 14, fontWeight: 400 }}>{unit}</span>
      </div>
      <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>{label}</div>
    </div>
  );
}

function RiskGauge({ value }) {
  const getColor = (v) => v >= 60 ? C.red : v >= 40 ? C.amber : C.medGreen;
  const getLabel = (v) => v >= 60 ? "ALTO" : v >= 40 ? "MODERADO" : "BAJO";
  const color = getColor(value);
  const angle = (value / 100) * 180;
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "10px 0" }}>
      <svg width="180" height="100" viewBox="0 0 180 100">
        <path d="M 10 90 A 80 80 0 0 1 170 90" fill="none" stroke={C.paleMint} strokeWidth="12" strokeLinecap="round" />
        <path d="M 10 90 A 80 80 0 0 1 170 90" fill="none" stroke={color} strokeWidth="12" strokeLinecap="round"
          strokeDasharray={`${angle / 180 * 251.3} 251.3`} />
        <text x="90" y="75" textAnchor="middle" fontSize="28" fontWeight="700" fontFamily="Georgia" fill={color}>{value}</text>
        <text x="90" y="92" textAnchor="middle" fontSize="10" fill={C.muted}>/100</text>
      </svg>
      <div style={{ fontSize: 13, fontWeight: 700, color, letterSpacing: 2, marginTop: 4 }}>{getLabel(value)}</div>
    </div>
  );
}

// ── WEATHER CONSTANTS ──
const WEATHER_VARS = [
  { id: "temperature_air",  label: "Temperatura del aire",   unit: "°C",    color: C.red },
  { id: "relative_humidity",label: "Humedad relativa",       unit: "%",     color: C.medGreen },
  { id: "precipitation",    label: "Precipitación",          unit: "mm",    color: C.deepGreen, type: "bar" },
  { id: "solar_radiation",  label: "Radiación solar",        unit: "W/m²",  color: C.amber },
  { id: "BP_mbar_Avg",      label: "Presión atmosférica",    unit: "hPa",   color: C.lightGreen },
  { id: "PtoRocio_Avg",     label: "Punto de rocío",         unit: "°C",    color: C.lightMuted },
  { id: "T107_10cm_Avg",    label: "Suelo a 10cm",           unit: "°C",    color: "#7BAE91" },
  { id: "T107_50cm_Avg",    label: "Suelo a 50cm",           unit: "°C",    color: "#A8CFA8" },
  { id: "albedo_Avg",       label: "Albedo",                 unit: "%",     color: C.warmSand },
  { id: "DT_Avg",           label: "Profundidad de nieve",   unit: "cm",    color: "#7EC8E3" },
];

const WIND_SPEED_COLORS = ["#BDE8FF", "#5BC0EB", "#1A6B9A", "#2ECC71", "#E67E22", "#E74C3C"];
const RESOLUTIONS = [
  { id: "15min", label: "15 min" },
  { id: "D",     label: "Diaria" },
  { id: "ME",    label: "Mensual" },
  { id: "Q",     label: "Estacional" },
];
const VAR_FRIENDLY = {
  ...Object.fromEntries(WEATHER_VARS.map(v => [v.id, v.label])),
  wind_speed: "Velocidad del viento",
};

// ── WIND ROSE SVG ──
function WindRose({ data }) {
  if (!data || data.length === 0) return null;
  const cx = 115, cy = 115, maxR = 90;
  const maxPct = Math.max(...data.map(d => d.total_pct), 0.1);

  function arcPath(r1, r2, a1Deg, a2Deg) {
    const toRad = d => (d * Math.PI) / 180;
    const a1 = toRad(a1Deg), a2 = toRad(a2Deg);
    const x2 = cx + r2 * Math.cos(a1), y2 = cy + r2 * Math.sin(a1);
    const x3 = cx + r2 * Math.cos(a2), y3 = cy + r2 * Math.sin(a2);
    const lg = (a2 - a1 > Math.PI) ? 1 : 0;
    if (r1 < 1) {
      return `M ${cx} ${cy} L ${x2.toFixed(1)} ${y2.toFixed(1)} A ${r2} ${r2} 0 ${lg} 1 ${x3.toFixed(1)} ${y3.toFixed(1)} Z`;
    }
    const x1 = cx + r1 * Math.cos(a1), y1 = cy + r1 * Math.sin(a1);
    const x4 = cx + r1 * Math.cos(a2), y4 = cy + r1 * Math.sin(a2);
    return `M ${x1.toFixed(1)} ${y1.toFixed(1)} L ${x2.toFixed(1)} ${y2.toFixed(1)} A ${r2} ${r2} 0 ${lg} 1 ${x3.toFixed(1)} ${y3.toFixed(1)} L ${x4.toFixed(1)} ${y4.toFixed(1)} A ${r1} ${r1} 0 ${lg} 0 ${x1.toFixed(1)} ${y1.toFixed(1)} Z`;
  }

  const sectors = [];
  data.forEach((d, i) => {
    const mid = i * 22.5 - 90; // rotate so 0° = North = up
    const a1 = mid - 11.25, a2 = mid + 11.25;
    let r0 = 0;
    d.bins.forEach((bin, j) => {
      const r = (bin.pct / maxPct) * maxR;
      if (r > 0.3) {
        sectors.push(
          <path key={`${i}-${j}`} d={arcPath(r0, r0 + r, a1, a2)}
            fill={WIND_SPEED_COLORS[j]} stroke="white" strokeWidth={0.4} opacity={0.9} />
        );
      }
      r0 += r;
    });
  });

  const cardinal = [
    { label: "N", dx: 0,        dy: -(maxR + 12) },
    { label: "E", dx: maxR + 14, dy: 4 },
    { label: "S", dx: 0,        dy: maxR + 16 },
    { label: "O", dx: -(maxR + 14), dy: 4 },
  ];

  return (
    <svg width={230} height={230} style={{ display: "block", margin: "0 auto" }}>
      {[0.25, 0.5, 0.75, 1].map(f => (
        <circle key={f} cx={cx} cy={cy} r={maxR * f} fill="none" stroke={C.paleMint} strokeWidth={0.8} strokeDasharray="3 3" />
      ))}
      <line x1={cx} y1={cy - maxR} x2={cx} y2={cy + maxR} stroke={C.paleMint} strokeWidth={0.5} />
      <line x1={cx - maxR} y1={cy} x2={cx + maxR} y2={cy} stroke={C.paleMint} strokeWidth={0.5} />
      {sectors}
      {cardinal.map(c => (
        <text key={c.label} x={cx + c.dx} y={cy + c.dy} textAnchor="middle" fontSize={10} fontWeight={700} fill={C.muted}>{c.label}</text>
      ))}
    </svg>
  );
}

// ── METEO TAB ──
function MeteoTab() {
  const today = new Date().toISOString().split("T")[0];
  const yearAgo = new Date(Date.now() - 365 * 86400e3).toISOString().split("T")[0];

  const [current, setCurrent] = useState(null);
  const [histData, setHistData] = useState([]);
  const [windRose, setWindRose] = useState(null);
  const [stats, setStats] = useState({});
  const [loading, setLoading] = useState(false);
  const [selectedVars, setSelectedVars] = useState(["temperature_air", "relative_humidity", "precipitation"]);
  const [showWind, setShowWind] = useState(false);
  const [resolution, setResolution] = useState("D");
  const [startDate, setStartDate] = useState(yearAgo);
  const [endDate, setEndDate] = useState(today);
  const [appliedDates, setAppliedDates] = useState({ start: yearAgo, end: today });

  useEffect(() => {
    fetch("/api/weather/current").then(r => r.json()).then(setCurrent).catch(() => {});
  }, []);

  useEffect(() => {
    if (!appliedDates.start || !appliedDates.end) return;
    setLoading(true);
    const varList = [
      ...selectedVars,
      ...(showWind ? ["wind_speed", "wind_direction"] : []),
    ].join(",");
    const p = new URLSearchParams({ start: appliedDates.start, end: appliedDates.end, resolution, variables: varList });
    fetch(`/api/weather/history?${p}`)
      .then(r => r.json())
      .then(json => {
        setHistData(json.data || []);
        setWindRose(json.wind_rose || null);
        setStats(json.stats || {});
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [selectedVars, showWind, resolution, appliedDates]);

  const toggleVar = id =>
    setSelectedVars(prev => prev.includes(id) ? prev.filter(v => v !== id) : [...prev, id]);

  const tickFmt = ts => {
    if (!ts) return "";
    const [year, month, dayPart] = ts.split("-");
    const day = parseInt(dayPart);
    const MONTHS = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"];
    const m = parseInt(month) - 1;
    if (resolution === "ME") return `${MONTHS[m]} ${year.slice(2)}`;
    if (resolution === "Q") return `T${Math.ceil((m + 1) / 3)} ${year.slice(2)}`;
    if (resolution === "15min") return `${ts.slice(5, 7)}/${ts.slice(8, 10)} ${ts.slice(11, 16)}`;
    return `${day} ${MONTHS[m]}`;
  };

  const varConf = id => WEATHER_VARS.find(v => v.id === id) || { label: id, unit: "", color: C.deepGreen };

  const commonChartProps = {
    margin: { top: 4, right: 4, left: -10, bottom: 0 },
  };
  const commonXAxis = (
    <XAxis dataKey="timestamp" tickFormatter={tickFmt}
      tick={{ fontSize: 9, fill: C.muted }} interval="preserveStartEnd"
      axisLine={{ stroke: C.mint }} tickLine={false} />
  );
  const commonTooltip = (label, unit) => (
    <Tooltip contentStyle={{ borderRadius: 6, fontSize: 11, border: `1px solid ${C.mint}` }}
      labelFormatter={tickFmt}
      formatter={v => [v != null ? v.toFixed(2) : "—", label]} />
  );

  return (
    <div style={{ padding: 16, overflowY: "auto", height: "100%" }}>
      {/* Current conditions */}
      <Card style={{ marginBottom: 12 }}>
        <SectionLabel>
          Condiciones actuales
          {current?.timestamp ? ` — ${new Date(current.timestamp).toLocaleString("es-CL", { dateStyle: "medium", timeStyle: "short" })}` : ""}
        </SectionLabel>
        {current ? (
          <div style={{ display: "flex", gap: 28, flexWrap: "wrap", marginTop: 6 }}>
            {[
              { v: current.temperature_air,     decimals: 1, u: "°C",   l: "Temperatura" },
              { v: current.relative_humidity,   decimals: 0, u: "%",    l: "Humedad" },
              { v: current.wind_speed_kmh,      decimals: 1, u: "km/h", l: "Viento" },
              { v: current.BP_mbar_Avg,         decimals: 0, u: " hPa", l: "Presión" },
              { v: current.precipitation,       decimals: 1, u: " mm",  l: "Precip. 15min" },
              { v: current.solar_radiation,     decimals: 0, u: " W/m²",l: "Radiación solar" },
            ].map(item => item.v != null && (
              <div key={item.l} style={{ textAlign: "center", minWidth: 64 }}>
                <div style={{ fontSize: 20, fontWeight: 700, color: C.text, fontFamily: "'Georgia', serif", lineHeight: 1 }}>
                  {Number(item.v).toFixed(item.decimals)}<span style={{ fontSize: 12, fontWeight: 400 }}>{item.u}</span>
                </div>
                <div style={{ fontSize: 10, color: C.muted, marginTop: 3 }}>{item.l}</div>
              </div>
            ))}
          </div>
        ) : (
          <div style={{ fontSize: 12, color: C.muted, marginTop: 6 }}>Conectando con la API...</div>
        )}
      </Card>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 270px", gap: 12 }}>
        {/* Left: controls + charts */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {/* Controls */}
          <Card>
            <SectionLabel>Variables</SectionLabel>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "5px 14px", marginTop: 6, marginBottom: 12 }}>
              {WEATHER_VARS.map(v => (
                <label key={v.id} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 12, color: C.text, cursor: "pointer" }}>
                  <input type="checkbox" checked={selectedVars.includes(v.id)} onChange={() => toggleVar(v.id)} />
                  {v.label}
                </label>
              ))}
              <label style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 12, color: C.text, cursor: "pointer" }}>
                <input type="checkbox" checked={showWind} onChange={() => setShowWind(v => !v)} />
                Viento (rosa)
              </label>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
              <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                <span style={{ fontSize: 11, color: C.muted }}>Desde:</span>
                <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)}
                  style={{ border: `1px solid ${C.mint}`, borderRadius: 4, padding: "4px 8px", fontSize: 12, color: C.text }} />
                <span style={{ fontSize: 11, color: C.muted }}>Hasta:</span>
                <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)}
                  style={{ border: `1px solid ${C.mint}`, borderRadius: 4, padding: "4px 8px", fontSize: 12, color: C.text }} />
                <button onClick={() => setAppliedDates({ start: startDate, end: endDate })}
                  style={{ padding: "5px 14px", background: C.deepGreen, color: C.white, border: "none", borderRadius: 4, fontSize: 12, cursor: "pointer" }}>
                  Aplicar
                </button>
              </div>
              <div style={{ display: "flex", gap: 12 }}>
                {RESOLUTIONS.map(r => (
                  <label key={r.id} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 12, color: C.text, cursor: "pointer" }}>
                    <input type="radio" name="resolution" value={r.id} checked={resolution === r.id} onChange={() => setResolution(r.id)} />
                    {r.label}
                  </label>
                ))}
              </div>
            </div>
          </Card>

          {loading && (
            <div style={{ textAlign: "center", padding: 24, color: C.muted, fontSize: 13 }}>Cargando datos...</div>
          )}

          {/* Per-variable charts */}
          {!loading && selectedVars.map(varId => {
            const conf = varConf(varId);
            const chartData = histData.filter(d => d[varId] != null);
            if (chartData.length === 0) return null;
            return (
              <Card key={varId}>
                <SectionLabel>{conf.label} ({conf.unit})</SectionLabel>
                <ResponsiveContainer width="100%" height={150}>
                  {conf.type === "bar" ? (
                    <BarChart data={chartData} {...commonChartProps}>
                      <CartesianGrid strokeDasharray="3 3" stroke={C.paleMint} />
                      {commonXAxis}
                      <YAxis tick={{ fontSize: 9, fill: C.muted }} axisLine={{ stroke: C.mint }} tickLine={false} unit={` ${conf.unit}`} width={52} />
                      {commonTooltip(conf.label, conf.unit)}
                      <Bar dataKey={varId} fill={conf.color} radius={[2, 2, 0, 0]} />
                    </BarChart>
                  ) : (
                    <LineChart data={chartData} {...commonChartProps}>
                      <CartesianGrid strokeDasharray="3 3" stroke={C.paleMint} />
                      {commonXAxis}
                      <YAxis tick={{ fontSize: 9, fill: C.muted }} axisLine={{ stroke: C.mint }} tickLine={false} unit={` ${conf.unit}`} width={52} />
                      {commonTooltip(conf.label, conf.unit)}
                      <Line type="monotone" dataKey={varId} stroke={conf.color} strokeWidth={1.5} dot={false} connectNulls={false} />
                    </LineChart>
                  )}
                </ResponsiveContainer>
              </Card>
            );
          })}

          {/* Wind speed chart (separate from variable selector) */}
          {!loading && showWind && histData.some(d => d.wind_speed != null) && (
            <Card>
              <SectionLabel>Velocidad del viento (km/h)</SectionLabel>
              <ResponsiveContainer width="100%" height={150}>
                <BarChart data={histData.filter(d => d.wind_speed != null)} {...commonChartProps}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.paleMint} />
                  {commonXAxis}
                  <YAxis tick={{ fontSize: 9, fill: C.muted }} axisLine={{ stroke: C.mint }} tickLine={false} unit=" km/h" width={52} />
                  <Tooltip contentStyle={{ borderRadius: 6, fontSize: 11, border: `1px solid ${C.mint}` }}
                    labelFormatter={tickFmt} formatter={v => [v != null ? v.toFixed(1) : "—", "Viento"]} />
                  <Bar dataKey="wind_speed" fill={C.lightGreen} radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          )}
        </div>

        {/* Right: wind rose + stats */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {showWind && windRose && (
            <Card>
              <SectionLabel>Rosa de vientos</SectionLabel>
              <WindRose data={windRose} />
              <div style={{ display: "flex", flexWrap: "wrap", gap: "3px 8px", marginTop: 8 }}>
                {["0–3","3–6","6–9","9–12","12–15","≥15"].map((r, i) => (
                  <div key={r} style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 10, color: C.muted }}>
                    <div style={{ width: 9, height: 9, borderRadius: 2, background: WIND_SPEED_COLORS[i], flexShrink: 0 }} />
                    {r} km/h
                  </div>
                ))}
              </div>
            </Card>
          )}

          {!loading && Object.keys(stats).length > 0 && (
            <Card>
              <SectionLabel>Estadísticas del período</SectionLabel>
              <table style={{ width: "100%", fontSize: 11, borderCollapse: "collapse", marginTop: 8 }}>
                <thead>
                  <tr>
                    {["Variable", "Media", "Mín", "Máx"].map(h => (
                      <th key={h} style={{ textAlign: h === "Variable" ? "left" : "right", color: C.muted, fontWeight: 600, paddingBottom: 4, fontSize: 10 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(stats).map(([key, s]) => (
                    <tr key={key} style={{ borderTop: `1px solid ${C.paleMint}` }}>
                      <td style={{ color: C.text, padding: "4px 0" }}>{VAR_FRIENDLY[key] || key}</td>
                      <td style={{ textAlign: "right", fontFamily: "monospace", color: C.text }}>{s.mean?.toFixed(1)}</td>
                      <td style={{ textAlign: "right", fontFamily: "monospace", color: C.muted }}>{s.min?.toFixed(1)}</td>
                      <td style={{ textAlign: "right", fontFamily: "monospace", color: C.muted }}>{s.max?.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {!showWind && !loading && Object.keys(stats).length === 0 && (
            <Card>
              <div style={{ color: C.lightMuted, fontSize: 12, textAlign: "center", padding: "24px 0", lineHeight: 1.6 }}>
                Selecciona variables,<br />define el período<br />y haz clic en Aplicar.
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

// ── PAGE: Observatorio ──
// Helper: fit map to boundary when GeoJSON loads
function FitBounds({ geojson }) {
  const map = useMap();
  useEffect(() => {
    if (geojson) {
      const layer = L.geoJSON(geojson);
      map.fitBounds(layer.getBounds(), { padding: [30, 30] });
    }
  }, [geojson, map]);
  return null;
}

// Custom weather station icon
const weatherIcon = L.divIcon({
  className: "",
  html: `<div style="width:28px;height:28px;background:${C.amber};border:2px solid white;border-radius:50%;display:flex;align-items:center;justify-content:center;box-shadow:0 2px 6px rgba(0,0,0,0.3)">
    <span style="color:white;font-weight:bold;font-size:13px">E</span>
  </div>`,
  iconSize: [28, 28],
  iconAnchor: [14, 14],
  popupAnchor: [0, -16],
});

function Observatorio() {
  const [boundary, setBoundary] = useState(null);
  const [cameras, setCameras] = useState(null);
  const [showBoundary, setShowBoundary] = useState(true);
  const [showCams, setShowCams] = useState(true);

  useEffect(() => {
    fetch("/data/boundary.geojson").then(r => r.json()).then(setBoundary);
    fetch("/data/camera_trap_stations.geojson").then(r => r.json()).then(setCameras);
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
          center={[-39.4417, -71.7420]}
          zoom={14}
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
          {showCams && (
            <Marker position={[-39.4417, -71.7420]} icon={weatherIcon}>
              <Popup>
                <div style={{ fontFamily: "'Trebuchet MS', sans-serif", minWidth: 160 }}>
                  <div style={{ fontWeight: 700, fontSize: 13, color: C.text, marginBottom: 6 }}>Estación Meteorológica</div>
                  <div style={{ fontSize: 11, color: C.muted }}>Campbell Scientific CR800</div>
                  <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>-39.4417, -71.7420</div>
                </div>
              </Popup>
            </Marker>
          )}

          {/* Camera trap stations */}
          {showCams && cameras && cameras.features.map(f => {
            const { id, tc, grid_id, altitude_m, sd_card } = f.properties;
            const [lon, lat] = f.geometry.coordinates;
            return (
              <CircleMarker
                key={id}
                center={[lat, lon]}
                radius={7}
                pathOptions={{
                  color: C.white,
                  weight: 2,
                  fillColor: C.deepGreen,
                  fillOpacity: 0.9,
                }}
              >
                <Popup>
                  <div style={{ fontFamily: "'Trebuchet MS', sans-serif", minWidth: 160 }}>
                    <div style={{ fontWeight: 700, fontSize: 13, color: C.text, marginBottom: 6 }}>
                      {id} <span style={{ fontWeight: 400, color: C.muted, fontSize: 11 }}>Grilla {grid_id}</span>
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4, fontSize: 11, color: C.muted }}>
                      <div>Altitud: <b style={{ color: C.text }}>{altitude_m} m</b></div>
                      <div>SD: <b style={{ color: C.text }}>{sd_card}</b></div>
                      <div style={{ gridColumn: "1/-1" }}>{lat.toFixed(5)}, {lon.toFixed(5)}</div>
                    </div>
                  </div>
                </Popup>
              </CircleMarker>
            );
          })}
        </MapContainer>

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
          {cameras ? `${cameras.features.length} cámaras trampa` : "Cargando..."}
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
          <RiskGauge value={67} />
        </Card>
        <Card>
          <SectionLabel>Meteorología</SectionLabel>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 8 }}>
            <StatBlock value="14.2" unit="°C" label="Temperatura" />
            <StatBlock value="62" unit="%" label="Humedad" />
            <StatBlock value="12" unit="km/h" label="Viento" />
            <StatBlock value="12" unit="días" label="Sin lluvia" color={C.amber} />
          </div>
        </Card>
        <Card>
          <SectionLabel>Resumen estaciones</SectionLabel>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 8 }}>
            <StatBlock value={cameras ? cameras.features.length : "..."} label="Cámaras trampa" />
            <StatBlock value="1" label="Estación meteo" />
          </div>
        </Card>
      </div>
    </div>
  );
}

// ── PAGE: Dashboard ──
function Dashboard() {
  const [tab, setTab] = useState("riesgo");
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const tabs = [
    { id: "riesgo", label: "Riesgo de Incendio" },
    { id: "meteo", label: "Meteorologia" },
    { id: "camaras", label: "Camaras Trampa" },
    { id: "fauna", label: "Fauna" },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "calc(100vh - 56px)" }}>
      {/* Tabs */}
      <div style={{ display: "flex", gap: 0, borderBottom: `2px solid ${C.mint}`, padding: "0 16px", flexShrink: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            background: tab === t.id ? C.white : "transparent", border: "none",
            borderBottom: tab === t.id ? `2px solid ${C.deepGreen}` : "2px solid transparent",
            padding: "10px 20px", cursor: "pointer", fontSize: 13,
            fontFamily: "'Trebuchet MS', sans-serif", color: tab === t.id ? C.text : C.muted,
            fontWeight: tab === t.id ? 700 : 400, borderRadius: "6px 6px 0 0",
            marginBottom: -2, transition: "all 0.2s",
          }}>{t.label}</button>
        ))}
      </div>

      {/* Tab content */}
      <div style={{ flex: 1, overflowY: "auto", padding: 16 }}>
      {tab === "riesgo" && mounted && (
        <div style={{ display: "grid", gridTemplateColumns: "300px 1fr", gap: 16 }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <Card>
              <SectionLabel>Indice actual</SectionLabel>
              <RiskGauge value={67} />
              <div style={{ fontSize: 11, color: C.muted, textAlign: "center", marginTop: 8, lineHeight: 1.5 }}>
                Formula: FRI = 0.35*T + 0.25*(100-H) + 0.20*V + 0.20*D
              </div>
            </Card>
            <Card>
              <SectionLabel>Factores contributivos</SectionLabel>
              <div style={{ marginTop: 8 }}>
                {[{ label: "Temperatura", value: 26, max: 40, color: C.red },
                  { label: "Humedad (inv)", value: 72, max: 100, color: C.amber },
                  { label: "Viento", value: 18, max: 50, color: C.medGreen },
                  { label: "Dias sin lluvia", value: 12, max: 30, color: C.amber }].map(f => (
                  <div key={f.label} style={{ marginBottom: 10 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 3 }}>
                      <span style={{ color: C.text }}>{f.label}</span>
                      <span style={{ color: C.muted, fontFamily: "monospace" }}>{f.value}</span>
                    </div>
                    <div style={{ height: 6, background: C.paleMint, borderRadius: 3 }}>
                      <div style={{ height: 6, background: f.color, borderRadius: 3, width: `${(f.value / f.max) * 100}%`, transition: "width 0.5s" }} />
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </div>
          <Card>
            <SectionLabel>Riesgo ultimos 7 dias</SectionLabel>
            <div style={{ fontFamily: "'Georgia', serif", fontSize: 15, fontWeight: 700, color: C.text, marginBottom: 16 }}>Tendencia semanal</div>
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={weeklyRisk}>
                <defs>
                  <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={C.red} stopOpacity={0.2} />
                    <stop offset="95%" stopColor={C.red} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={C.paleMint} />
                <XAxis dataKey="dia" tick={{ fontSize: 11, fill: C.muted }} axisLine={{ stroke: C.mint }} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: C.muted }} axisLine={{ stroke: C.mint }} />
                <Tooltip contentStyle={{ borderRadius: 6, border: `1px solid ${C.mint}`, fontSize: 12 }} />
                <Area type="monotone" dataKey="riesgo" stroke={C.red} fill="url(#riskGrad)" strokeWidth={2} dot={{ r: 4, fill: C.red }} />
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {tab === "meteo" && mounted && <MeteoTab />}

      {tab === "camaras" && mounted && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          <Card>
            <SectionLabel>Actividad por hora</SectionLabel>
            <div style={{ fontSize: 13, fontWeight: 700, color: C.text, marginBottom: 12, fontFamily: "'Georgia', serif" }}>Patron de actividad (todas las camaras)</div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={activityByHour}>
                <CartesianGrid strokeDasharray="3 3" stroke={C.paleMint} />
                <XAxis dataKey="hora" tick={{ fontSize: 9, fill: C.muted }} axisLine={{ stroke: C.mint }} />
                <YAxis tick={{ fontSize: 10, fill: C.muted }} axisLine={{ stroke: C.mint }} />
                <Tooltip contentStyle={{ borderRadius: 6, fontSize: 11 }} />
                <Bar dataKey="actividad" fill={C.deepGreen} radius={[2, 2, 0, 0]} name="Detecciones" />
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 11, color: C.muted, marginTop: 8, fontStyle: "italic" }}>
              Pico nocturno consistente con fauna nativa de habitos crepusculares y nocturnos
            </div>
          </Card>
          <Card>
            <SectionLabel>Estado de camaras</SectionLabel>
            <div style={{ marginTop: 8 }}>
              {stations.filter(s => s.type === "camera").map(s => (
                <div key={s.id} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 0", borderBottom: `1px solid ${C.paleMint}` }}>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: C.text }}>{s.name}</div>
                    <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>
                      Ultima: {s.lastDetection} — {s.time}
                    </div>
                  </div>
                  <div style={{ width: 8, height: 8, borderRadius: "50%", background: C.medGreen }} title="Activa" />
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {tab === "fauna" && mounted && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: 16 }}>
          <Card>
            <SectionLabel>Detecciones por especie (ultimo mes)</SectionLabel>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={speciesData} layout="vertical" margin={{ left: 100 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={C.paleMint} horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 10, fill: C.muted }} axisLine={{ stroke: C.mint }} />
                <YAxis type="category" dataKey="nombre" tick={{ fontSize: 11, fill: C.text }} axisLine={{ stroke: C.mint }} width={100} />
                <Tooltip contentStyle={{ borderRadius: 6, fontSize: 11 }} />
                <Bar dataKey="detecciones" radius={[0, 4, 4, 0]} name="Detecciones">
                  {speciesData.map((entry, i) => (
                    <Cell key={i} fill={["Puma", "Guina"].includes(entry.nombre) ? C.amber : ["Jabali", "Liebre europea"].includes(entry.nombre) ? C.red : C.deepGreen} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div style={{ display: "flex", gap: 16, marginTop: 8 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 10, color: C.muted }}>
                <div style={{ width: 10, height: 10, borderRadius: 2, background: C.deepGreen }} /> Nativa
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 10, color: C.muted }}>
                <div style={{ width: 10, height: 10, borderRadius: 2, background: C.amber }} /> Prioritaria
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 10, color: C.muted }}>
                <div style={{ width: 10, height: 10, borderRadius: 2, background: C.red }} /> Invasora
              </div>
            </div>
          </Card>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <Card>
              <SectionLabel>Resumen del mes</SectionLabel>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 10 }}>
                <StatBlock value="978" label="Total detecciones" />
                <StatBlock value="8" label="Especies" />
                <StatBlock value="5" label="Camaras activas" />
                <StatBlock value="31" label="Dias muestreados" />
              </div>
            </Card>
            <Card>
              <SectionLabel>Alertas de especies</SectionLabel>
              <div style={{ marginTop: 8 }}>
                <div style={{ padding: "8px 10px", background: `${C.amber}15`, borderRadius: 6, marginBottom: 6, borderLeft: `3px solid ${C.amber}` }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: C.text }}>Puma detectado</div>
                  <div style={{ fontSize: 11, color: C.muted }}>Camara Laguna Sur — hace 2 dias</div>
                </div>
                <div style={{ padding: "8px 10px", background: `${C.red}12`, borderRadius: 6, borderLeft: `3px solid ${C.red}` }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: C.text }}>Jabali: tendencia al alza</div>
                  <div style={{ fontSize: 11, color: C.muted }}>+23% vs mes anterior — 3 estaciones</div>
                </div>
              </div>
            </Card>
          </div>
        </div>
      )}
      </div>
    </div>
  );
}

// ── PAGE: Asistente ──
function Asistente() {
  const [messages, setMessages] = useState(chatMessages);
  const [input, setInput] = useState("");
  const [typing, setTyping] = useState(false);
  const endRef = useRef(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  const handleSend = () => {
    if (!input.trim()) return;
    const userMsg = input.trim();
    setMessages(prev => [...prev, { role: "user", text: userMsg }]);
    setInput("");
    setTyping(true);
    setTimeout(() => {
      setMessages(prev => [...prev, {
        role: "assistant",
        text: "Esta es una demostracion del Asistente. En la version final, cada respuesta consultara la base de datos en tiempo real y citara la formula o metodologia utilizada para generar la informacion. Transparencia metodologica total.",
      }]);
      setTyping(false);
    }, 1500);
  };

  const suggestions = [
    "Como se calcula el indice de riesgo?",
    "Cuantas detecciones de Puma hay este ano?",
    "Cual es la tendencia de temperatura?",
    "Que camaras tienen mas actividad?",
  ];

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 280px", gap: 16, height: "calc(100vh - 56px)", padding: 16 }}>
      <Card style={{ display: "flex", flexDirection: "column", padding: 0, overflow: "hidden" }}>
        {/* Header */}
        <div style={{ padding: "14px 20px", borderBottom: `1px solid ${C.paleMint}`, background: C.white }}>
          <div style={{ fontSize: 15, fontWeight: 700, color: C.text, fontFamily: "'Georgia', serif" }}>Asistente Territorial</div>
          <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>Consulta datos y metodologias en lenguaje natural</div>
        </div>

        {/* Messages */}
        <div style={{ flex: 1, overflowY: "auto", padding: 20, display: "flex", flexDirection: "column", gap: 12, background: C.bg }}>
          {messages.map((m, i) => (
            <div key={i} style={{ display: "flex", justifyContent: m.role === "user" ? "flex-end" : "flex-start" }}>
              <div style={{
                maxWidth: "80%", padding: "10px 14px", borderRadius: 12, fontSize: 13, lineHeight: 1.6,
                background: m.role === "user" ? C.deepGreen : C.white,
                color: m.role === "user" ? C.white : C.text,
                borderBottomRightRadius: m.role === "user" ? 2 : 12,
                borderBottomLeftRadius: m.role === "user" ? 12 : 2,
                boxShadow: m.role === "assistant" ? "0 1px 3px rgba(0,0,0,0.06)" : "none",
              }}>
                {m.text}
                {m.role === "assistant" && (
                  <div style={{ fontSize: 10, color: C.lightMuted, marginTop: 6, fontStyle: "italic" }}>
                    Fuente: base de datos local + formula documentada
                  </div>
                )}
              </div>
            </div>
          ))}
          {typing && (
            <div style={{ display: "flex", justifyContent: "flex-start" }}>
              <div style={{ background: C.white, padding: "10px 14px", borderRadius: 12, fontSize: 13, color: C.muted, boxShadow: "0 1px 3px rgba(0,0,0,0.06)" }}>
                Consultando datos...
              </div>
            </div>
          )}
          <div ref={endRef} />
        </div>

        {/* Input */}
        <div style={{ padding: "12px 16px", borderTop: `1px solid ${C.paleMint}`, background: C.white, display: "flex", gap: 8 }}>
          <input value={input} onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleSend()}
            placeholder="Pregunta sobre datos o metodologias..."
            style={{ flex: 1, border: `1px solid ${C.mint}`, borderRadius: 8, padding: "10px 14px", fontSize: 13, outline: "none", fontFamily: "inherit" }}
          />
          <button onClick={handleSend} style={{
            background: C.deepGreen, color: C.white, border: "none", borderRadius: 8,
            padding: "10px 20px", cursor: "pointer", fontSize: 13, fontWeight: 600,
          }}>Enviar</button>
        </div>
      </Card>

      {/* Sidebar */}
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <Card>
          <SectionLabel>Preguntas sugeridas</SectionLabel>
          <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 8 }}>
            {suggestions.map((s, i) => (
              <button key={i} onClick={() => { setInput(s); }}
                style={{ background: C.paleMint, border: "none", borderRadius: 6, padding: "8px 10px",
                  fontSize: 11, color: C.text, textAlign: "left", cursor: "pointer", lineHeight: 1.4,
                  transition: "background 0.2s" }}
                onMouseEnter={e => e.target.style.background = C.mint}
                onMouseLeave={e => e.target.style.background = C.paleMint}>
                {s}
              </button>
            ))}
          </div>
        </Card>
        <Card>
          <SectionLabel>Principio de transparencia</SectionLabel>
          <div style={{ fontSize: 12, color: C.muted, lineHeight: 1.6, marginTop: 6 }}>
            Cada respuesta que involucre un valor calculado muestra su formula, los datos de entrada y el modelo que lo produjo. Sin cajas negras.
          </div>
        </Card>
        <Card>
          <SectionLabel>Capacidades</SectionLabel>
          <div style={{ marginTop: 6, display: "flex", flexDirection: "column", gap: 6 }}>
            {["Consultar riesgo de incendio", "Buscar detecciones de especies", "Explicar metodologias", "Analizar tendencias climaticas"].map((c, i) => (
              <div key={i} style={{ fontSize: 11, color: C.text, display: "flex", alignItems: "center", gap: 6 }}>
                <div style={{ width: 5, height: 5, borderRadius: "50%", background: C.medGreen, flexShrink: 0 }} />
                {c}
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

// ── PAGE: Reportes ──
function Reportes() {
  const [generating, setGenerating] = useState(false);
  const [draft, setDraft] = useState("");
  const [period, setPeriod] = useState("Febrero 2026");

  const sampleDraft = `Resumen mensual — Bosque Pehuen, ${period}

Durante ${period.toLowerCase()}, las condiciones en Bosque Pehuen se caracterizaron por temperaturas superiores al promedio historico y precipitaciones por debajo de lo esperado para la temporada.

Riesgo de incendio: El indice promedio fue de 48/100 (moderado), con un pico de 72/100 el dia 15, coincidiendo con una ola de calor que elevo la temperatura maxima a 31°C con humedad relativa de 22%. Se recomienda mantener la vigilancia activa durante episodios similares.

Monitoreo de fauna: Las 5 camaras trampa registraron un total de 978 detecciones de 8 especies. Destaca un aumento del 23% en registros de jabali respecto al mes anterior, concentrado en las estaciones Rio Turbio y Araucaria Norte. Se registro 1 evento de Puma en Laguna Sur, consistente con su rango de movimiento estacional.

Especies prioritarias: Guina fue detectada en 23 ocasiones en Sendero Pehuen, lo que representa una frecuencia estable respecto a meses anteriores. No se detectaron nuevas especies invasoras.

Nota: Este borrador fue generado automaticamente a partir de los datos del repositorio central. Requiere revision y edicion del equipo antes de su publicacion.`;

  const handleGenerate = () => {
    setGenerating(true);
    setDraft("");
    let i = 0;
    const interval = setInterval(() => {
      setDraft(sampleDraft.slice(0, i));
      i += 3;
      if (i > sampleDraft.length) {
        clearInterval(interval);
        setGenerating(false);
        setDraft(sampleDraft);
      }
    }, 8);
  };

  return (
    <div style={{ padding: 16, height: "calc(100vh - 56px)", overflowY: "auto" }}>
      <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: 16 }}>
        {/* Config panel */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <Card>
            <SectionLabel>Configuracion</SectionLabel>
            <div style={{ fontFamily: "'Georgia', serif", fontSize: 15, fontWeight: 700, color: C.text, marginBottom: 12 }}>Generar Reporte</div>
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11, color: C.muted, marginBottom: 4 }}>Periodo</div>
              <select value={period} onChange={e => setPeriod(e.target.value)}
                style={{ width: "100%", padding: "8px 10px", border: `1px solid ${C.mint}`, borderRadius: 6, fontSize: 13, background: C.white, color: C.text }}>
                <option>Febrero 2026</option>
                <option>Enero 2026</option>
                <option>Diciembre 2025</option>
              </select>
            </div>
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11, color: C.muted, marginBottom: 4 }}>Secciones a incluir</div>
              {["Riesgo de incendio", "Meteorologia", "Fauna y camaras trampa", "Especies prioritarias"].map((s, i) => (
                <label key={i} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: C.text, marginBottom: 4, cursor: "pointer" }}>
                  <input type="checkbox" defaultChecked /> {s}
                </label>
              ))}
            </div>
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11, color: C.muted, marginBottom: 4 }}>Audiencia</div>
              <select style={{ width: "100%", padding: "8px 10px", border: `1px solid ${C.mint}`, borderRadius: 6, fontSize: 13, background: C.white, color: C.text }}>
                <option>Equipo FMA (interna)</option>
                <option>Socios de conservacion</option>
                <option>Publico general</option>
              </select>
            </div>
            <button onClick={handleGenerate} disabled={generating}
              style={{
                width: "100%", background: generating ? C.muted : C.deepGreen, color: C.white,
                border: "none", borderRadius: 8, padding: "12px 0", cursor: generating ? "wait" : "pointer",
                fontSize: 13, fontWeight: 700, transition: "background 0.2s",
              }}>
              {generating ? "Generando..." : "Generar borrador"}
            </button>
          </Card>
          <Card>
            <SectionLabel>Importante</SectionLabel>
            <div style={{ fontSize: 12, color: C.muted, lineHeight: 1.6, marginTop: 6 }}>
              La IA redacta un borrador a partir de los datos del periodo seleccionado. El equipo humano revisa, edita y decide si publicar. Este modulo es un acelerador de escritura, no un reemplazo.
            </div>
          </Card>
          <Card>
            <SectionLabel>Exportar</SectionLabel>
            <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
              <button style={{ flex: 1, padding: "8px 0", border: `1px solid ${C.mint}`, borderRadius: 6, background: C.white, color: C.text, fontSize: 11, cursor: "pointer" }}>
                Word .docx
              </button>
              <button style={{ flex: 1, padding: "8px 0", border: `1px solid ${C.mint}`, borderRadius: 6, background: C.white, color: C.text, fontSize: 11, cursor: "pointer" }}>
                Copiar texto
              </button>
            </div>
          </Card>
        </div>

        {/* Draft area */}
        <Card style={{ display: "flex", flexDirection: "column" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <div>
              <SectionLabel>Borrador</SectionLabel>
              <div style={{ fontFamily: "'Georgia', serif", fontSize: 15, fontWeight: 700, color: C.text }}>
                Reporte mensual — {period}
              </div>
            </div>
            {draft && <div style={{ fontSize: 10, color: C.lightMuted, fontFamily: "monospace" }}>BORRADOR — REQUIERE REVISION</div>}
          </div>
          {draft ? (
            <textarea value={draft} onChange={e => setDraft(e.target.value)}
              style={{
                flex: 1, minHeight: 400, border: `1px solid ${C.mint}`, borderRadius: 8, padding: 16,
                fontSize: 13, lineHeight: 1.8, color: C.text, fontFamily: "'Georgia', serif",
                resize: "none", outline: "none", background: C.bg,
              }}
            />
          ) : (
            <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", minHeight: 400, color: C.lightMuted, fontSize: 14 }}>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: 40, marginBottom: 12, opacity: 0.3 }}>&#9998;</div>
                <div>Selecciona el periodo y haz clic en "Generar borrador"</div>
                <div style={{ fontSize: 12, marginTop: 6 }}>El sistema consultara los datos del mes y redactara un resumen</div>
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}

// ── MAIN APP ──
export default function App() {
  const [page, setPage] = useState("observatorio");

  return (
    <div style={{ background: C.bg, minHeight: "100vh", fontFamily: "'Trebuchet MS', 'Segoe UI', sans-serif" }}>
      <NavBar page={page} setPage={setPage} />
      <ErrorBoundary key={page}>
        {page === "observatorio" && <Observatorio />}
        {page === "dashboard" && <Dashboard />}
        {page === "asistente" && <Asistente />}
        {page === "reportes" && <Reportes />}
      </ErrorBoundary>
    </div>
  );
}
