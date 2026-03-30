import { useState, useEffect, useRef, useMemo, Component } from "react";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer, RadialBarChart, RadialBar, PieChart, Pie, Cell } from "recharts";
import { MapContainer, TileLayer, GeoJSON, CircleMarker, Marker, Popup, useMap } from "react-leaflet";
import L from "leaflet";
import {
  getWeatherCurrent,
  getFireRiskCurrent, getFireRiskForecast, getFireRiskHistory, getSpeciesSummary,
  transformRiskForecast, transformSpeciesSummary,
} from "./api.js";

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

// ── Shared data-fetching hook ──
function useAPI(fetchFn, transformFn, deps = []) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchFn()
      .then(raw => { if (!cancelled) setData(transformFn ? transformFn(raw) : raw); })
      .catch(err => { if (!cancelled) setError(err.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, deps);
  return { data, loading, error };
}

// ── Mock Data (partially replaced by API) ──

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

function RiskGauge({ value, color: colorProp = null, compact = false }) {
  const getColor = (v) =>
    v >= 90 ? "#b71c1c" : v >= 80 ? "#e53935" : v >= 60 ? "#fb8c00" :
    v >= 40 ? "#fbc02d" : v >= 20 ? "#c0ca33" : "#2e7d32";
  const getLabel = (v) =>
    v >= 90 ? "EXTREMO" : v >= 80 ? "MUY ALTO" : v >= 60 ? "ALTO" :
    v >= 40 ? "MODERADO" : v >= 20 ? "MOD-BAJO" : "BAJO";
  const color = colorProp || getColor(value);
  const angle = (value / 100) * 180;
  const w = compact ? 120 : 180;
  const h = compact ? 67 : 100;
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: compact ? "6px 0" : "10px 0" }}>
      <svg width={w} height={h} viewBox="0 0 180 100">
        <path d="M 10 90 A 80 80 0 0 1 170 90" fill="none" stroke={C.paleMint} strokeWidth="12" strokeLinecap="round" />
        <path d="M 10 90 A 80 80 0 0 1 170 90" fill="none" stroke={color} strokeWidth="12" strokeLinecap="round"
          strokeDasharray={`${angle / 180 * 251.3} 251.3`} />
        <text x="90" y="75" textAnchor="middle" fontSize={compact ? "24" : "28"} fontWeight="700" fontFamily="Georgia" fill={color}>{value}</text>
        <text x="90" y="92" textAnchor="middle" fontSize="10" fill={C.muted}>/100</text>
      </svg>
      <div style={{ fontSize: compact ? 10 : 13, fontWeight: 700, color, letterSpacing: 2, marginTop: 4 }}>{getLabel(value)}</div>
    </div>
  );
}

// ── POLAR CONTRIBUTION CHART ──
function PolarContrib({ components, size = 180 }) {
  const axes = [
    { key: "temp_score", label: "Temp", max: 25, angle: -90 },
    { key: "wind_score", label: "Viento", max: 15, angle: 0 },
    { key: "days_score", label: "Días s/lluvia", max: 35, angle: 90 },
    { key: "rh_score", label: "Humedad", max: 25, angle: 180 },
  ];
  const cx = 90, cy = 90, maxR = 58;
  const toRad = d => d * Math.PI / 180;
  const pts = axes.map(a => {
    const val = components?.[a.key] ?? 0;
    const r = Math.min(val / a.max, 1) * maxR;
    return [cx + r * Math.cos(toRad(a.angle)), cy + r * Math.sin(toRad(a.angle))];
  });
  return (
    <svg width={size} height={size} viewBox="0 0 180 180" overflow="visible">
      {[0.25, 0.5, 0.75, 1].map(l => (
        <circle key={l} cx={cx} cy={cy} r={maxR * l} fill="none" stroke={C.paleMint} strokeWidth="1" />
      ))}
      {axes.map(a => {
        const rad = toRad(a.angle);
        return <line key={a.key} x1={cx} y1={cy} x2={cx + maxR * Math.cos(rad)} y2={cy + maxR * Math.sin(rad)} stroke={C.mint} strokeWidth="1" />;
      })}
      <polygon points={pts.map(p => p.join(",")).join(" ")} fill={`${C.amber}35`} stroke={C.amber} strokeWidth="2" strokeLinejoin="round" />
      {pts.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={3} fill={C.amber} />)}
      {axes.map(a => {
        const rad = toRad(a.angle);
        const lx = cx + (maxR + 20) * Math.cos(rad);
        const ly = cy + (maxR + 20) * Math.sin(rad);
        const val = (components?.[a.key] ?? 0).toFixed(1);
        return (
          <g key={a.key}>
            <text x={lx} y={ly - 5} textAnchor="middle" fontSize="9" fontWeight="700" fill={C.text}>{a.label}</text>
            <text x={lx} y={ly + 6} textAnchor="middle" fontSize="8" fill={C.muted}>{val}/{a.max}</text>
          </g>
        );
      })}
    </svg>
  );
}

// ── WIND COMPASS ──
function WindCompass({ direction, speed }) {
  const cx = 55, cy = 55, r = 44;
  const toRad = d => (d - 90) * Math.PI / 180;
  const dirs = ["N", "E", "S", "O"];
  const dirAngles = [0, 90, 180, 270];
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      <svg width="110" height="110" viewBox="0 0 110 110">
        <defs>
          <marker id="windArrow" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto">
            <path d="M 0 0 L 5 2.5 L 0 5 Z" fill={C.deepGreen} />
          </marker>
        </defs>
        <circle cx={cx} cy={cy} r={r} fill={C.paleMint} stroke={C.mint} strokeWidth="1.5" />
        {Array.from({ length: 16 }, (_, i) => {
          const rad = toRad(i * 22.5);
          const rInner = i % 4 === 0 ? r - 9 : r - 5;
          return <line key={i} x1={cx + r * Math.cos(rad)} y1={cy + r * Math.sin(rad)}
            x2={cx + rInner * Math.cos(rad)} y2={cy + rInner * Math.sin(rad)}
            stroke={C.lightMuted} strokeWidth={i % 4 === 0 ? 1.5 : 1} />;
        })}
        {dirs.map((label, i) => {
          const rad = toRad(dirAngles[i]);
          const dist = r - 16;
          return <text key={label} x={cx + dist * Math.cos(rad)} y={cy + dist * Math.sin(rad)}
            textAnchor="middle" dominantBaseline="middle" fontSize="9"
            fontWeight={label === "N" ? "700" : "400"} fill={label === "N" ? C.red : C.text}>{label}</text>;
        })}
        {direction != null && (() => {
          const rad = toRad(direction);
          return <line x1={cx - 16 * Math.cos(rad)} y1={cy - 16 * Math.sin(rad)}
            x2={cx + 28 * Math.cos(rad)} y2={cy + 28 * Math.sin(rad)}
            stroke={C.deepGreen} strokeWidth="2.5" strokeLinecap="round" markerEnd="url(#windArrow)" />;
        })()}
        <circle cx={cx} cy={cy} r={3} fill={C.deepGreen} />
      </svg>
      <div style={{ fontSize: 12, color: C.text, fontWeight: 600, marginTop: 2 }}>
        {speed != null ? `${Number(speed).toFixed(1)} km/h` : "—"}
      </div>
      {direction != null && (
        <div style={{ fontSize: 10, color: C.muted }}>{direction}°</div>
      )}
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
function WindRose({ data, size = 230 }) {
  if (!data || data.length === 0) return null;
  const s = size / 230;
  const cx = 115 * s, cy = 115 * s, maxR = 90 * s;
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
    { label: "N", dx: 0,              dy: -(maxR + 12 * s) },
    { label: "E", dx: maxR + 14 * s,  dy: 4 * s },
    { label: "S", dx: 0,              dy: maxR + 16 * s },
    { label: "O", dx: -(maxR + 14 * s), dy: 4 * s },
  ];

  return (
    <svg width={size} height={size} style={{ display: "block", margin: "0 auto" }}>
      {[0.25, 0.5, 0.75, 1].map(f => (
        <circle key={f} cx={cx} cy={cy} r={maxR * f} fill="none" stroke={C.paleMint} strokeWidth={0.8} strokeDasharray="3 3" />
      ))}
      <line x1={cx} y1={cy - maxR} x2={cx} y2={cy + maxR} stroke={C.paleMint} strokeWidth={0.5} />
      <line x1={cx - maxR} y1={cy} x2={cx + maxR} y2={cy} stroke={C.paleMint} strokeWidth={0.5} />
      {sectors}
      {cardinal.map(c => (
        <text key={c.label} x={cx + c.dx} y={cy + c.dy} textAnchor="middle" fontSize={Math.round(10 * s)} fontWeight={700} fill={C.muted}>{c.label}</text>
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

  // Comparison mode
  const [compareMode, setCompareMode] = useState(false);
  const [startDate2, setStartDate2] = useState(yearAgo);
  const [endDate2, setEndDate2] = useState(today);
  const [appliedDates2, setAppliedDates2] = useState({ start: yearAgo, end: today });
  const [histData2, setHistData2] = useState([]);
  const [windRose2, setWindRose2] = useState(null);
  const [stats2, setStats2] = useState({});
  const [loading2, setLoading2] = useState(false);

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

  // Period 2 fetch (comparison mode)
  useEffect(() => {
    if (!compareMode || !appliedDates2.start || !appliedDates2.end) return;
    setLoading2(true);
    const varList = [
      ...selectedVars,
      ...(showWind ? ["wind_speed", "wind_direction"] : []),
    ].join(",");
    const p = new URLSearchParams({ start: appliedDates2.start, end: appliedDates2.end, resolution, variables: varList });
    fetch(`/api/weather/history?${p}`)
      .then(r => r.json())
      .then(json => {
        setHistData2(json.data || []);
        setWindRose2(json.wind_rose || null);
        setStats2(json.stats || {});
        setLoading2(false);
      })
      .catch(() => setLoading2(false));
  }, [compareMode, selectedVars, showWind, resolution, appliedDates2]);

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

  const renderVarChart = (varId, data, colorOverride) => {
    const conf = varConf(varId);
    const color = colorOverride || conf.color;
    return conf.type === "bar" ? (
      <BarChart data={data} {...commonChartProps}>
        <CartesianGrid strokeDasharray="3 3" stroke={C.paleMint} />
        {commonXAxis}
        <YAxis tick={{ fontSize: 9, fill: C.muted }} axisLine={{ stroke: C.mint }} tickLine={false} unit={` ${conf.unit}`} width={52} />
        {commonTooltip(conf.label, conf.unit)}
        <Bar dataKey={varId} fill={color} radius={[2, 2, 0, 0]} />
      </BarChart>
    ) : (
      <LineChart data={data} {...commonChartProps}>
        <CartesianGrid strokeDasharray="3 3" stroke={C.paleMint} />
        {commonXAxis}
        <YAxis tick={{ fontSize: 9, fill: C.muted }} axisLine={{ stroke: C.mint }} tickLine={false} unit={` ${conf.unit}`} width={52} />
        {commonTooltip(conf.label, conf.unit)}
        <Line type="monotone" dataKey={varId} stroke={color} strokeWidth={1.5} dot={false} connectNulls={false} />
      </LineChart>
    );
  };

  return (
    <div style={{ padding: 16, overflowY: "auto", height: "100%" }}>
      {/* Current conditions */}
      <Card style={{ marginBottom: 12 }}>
        <SectionLabel>
          Última medición
          {current?.timestamp ? ` — ${new Date(current.timestamp).toLocaleDateString("es-CL", { day: "numeric", month: "long", year: "numeric" })}` : ""}
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

            {/* Comparison mode toggle + Period 2 date range */}
            <div style={{ marginTop: 10, borderTop: `1px solid ${C.paleMint}`, paddingTop: 10, display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
              <label style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 12, color: C.text, cursor: "pointer", fontWeight: 600 }}>
                <input type="checkbox" checked={compareMode} onChange={() => setCompareMode(v => !v)} />
                Comparar períodos
              </label>
              {compareMode && (
                <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  <span style={{ fontSize: 11, color: C.amber, fontWeight: 600 }}>Período 2:</span>
                  <span style={{ fontSize: 11, color: C.muted }}>Desde:</span>
                  <input type="date" value={startDate2} onChange={e => setStartDate2(e.target.value)}
                    style={{ border: `1px solid ${C.amber}`, borderRadius: 4, padding: "4px 8px", fontSize: 12, color: C.text }} />
                  <span style={{ fontSize: 11, color: C.muted }}>Hasta:</span>
                  <input type="date" value={endDate2} onChange={e => setEndDate2(e.target.value)}
                    style={{ border: `1px solid ${C.amber}`, borderRadius: 4, padding: "4px 8px", fontSize: 12, color: C.text }} />
                  <button onClick={() => setAppliedDates2({ start: startDate2, end: endDate2 })}
                    style={{ padding: "5px 14px", background: C.amber, color: C.white, border: "none", borderRadius: 4, fontSize: 12, cursor: "pointer" }}>
                    Aplicar
                  </button>
                </div>
              )}
            </div>
          </Card>

          {(loading || (compareMode && loading2)) && (
            <div style={{ textAlign: "center", padding: 24, color: C.muted, fontSize: 13 }}>Cargando datos...</div>
          )}

          {/* Per-variable charts */}
          {!loading && selectedVars.map(varId => {
            const conf = varConf(varId);
            const chartData = histData.filter(d => d[varId] != null);
            const chartData2 = compareMode ? histData2.filter(d => d[varId] != null) : [];
            if (chartData.length === 0 && chartData2.length === 0) return null;
            return (
              <Card key={varId}>
                {compareMode && <div style={{ fontSize: 10, color: C.deepGreen, fontWeight: 600, marginBottom: 2 }}>Período 1</div>}
                <SectionLabel>{conf.label} ({conf.unit})</SectionLabel>
                {chartData.length > 0 && (
                  <ResponsiveContainer width="100%" height={150}>
                    {renderVarChart(varId, chartData)}
                  </ResponsiveContainer>
                )}
                {compareMode && (
                  <>
                    <div style={{ fontSize: 10, color: C.amber, fontWeight: 600, marginTop: 12, marginBottom: 2 }}>Período 2</div>
                    {chartData2.length > 0 ? (
                      <ResponsiveContainer width="100%" height={150}>
                        {renderVarChart(varId, chartData2, C.amber)}
                      </ResponsiveContainer>
                    ) : !loading2 && (
                      <div style={{ fontSize: 11, color: C.muted, padding: 8 }}>Sin datos para este período</div>
                    )}
                  </>
                )}
              </Card>
            );
          })}

          {/* Wind speed chart (separate from variable selector) */}
          {!loading && showWind && (histData.some(d => d.wind_speed != null) || (compareMode && histData2.some(d => d.wind_speed != null))) && (
            <Card>
              {compareMode && <div style={{ fontSize: 10, color: C.deepGreen, fontWeight: 600, marginBottom: 2 }}>Período 1</div>}
              <SectionLabel>Velocidad del viento (km/h)</SectionLabel>
              {histData.some(d => d.wind_speed != null) && (
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
              )}
              {compareMode && (
                <>
                  <div style={{ fontSize: 10, color: C.amber, fontWeight: 600, marginTop: 12, marginBottom: 2 }}>Período 2</div>
                  {histData2.some(d => d.wind_speed != null) ? (
                    <ResponsiveContainer width="100%" height={150}>
                      <BarChart data={histData2.filter(d => d.wind_speed != null)} {...commonChartProps}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.paleMint} />
                        {commonXAxis}
                        <YAxis tick={{ fontSize: 9, fill: C.muted }} axisLine={{ stroke: C.mint }} tickLine={false} unit=" km/h" width={52} />
                        <Tooltip contentStyle={{ borderRadius: 6, fontSize: 11, border: `1px solid ${C.mint}` }}
                          labelFormatter={tickFmt} formatter={v => [v != null ? v.toFixed(1) : "—", "Viento"]} />
                        <Bar dataKey="wind_speed" fill={C.amber} radius={[2, 2, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : !loading2 && (
                    <div style={{ fontSize: 11, color: C.muted, padding: 8 }}>Sin datos de viento para este período</div>
                  )}
                </>
              )}
            </Card>
          )}

          {showWind && (windRose || (compareMode && windRose2)) && (
            <Card>
              <SectionLabel>Rosa de vientos</SectionLabel>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 24, justifyContent: "center", marginTop: 8 }}>
                {windRose && (
                  <div style={{ textAlign: "center" }}>
                    {compareMode && <div style={{ fontSize: 10, color: C.deepGreen, fontWeight: 600, marginBottom: 4 }}>Período 1</div>}
                    <WindRose data={windRose} size={compareMode ? 280 : 350} />
                  </div>
                )}
                {compareMode && windRose2 && (
                  <div style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 10, color: C.amber, fontWeight: 600, marginBottom: 4 }}>Período 2</div>
                    <WindRose data={windRose2} size={280} />
                  </div>
                )}
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "3px 8px", marginTop: 8, justifyContent: "center" }}>
                {["0–3","3–6","6–9","9–12","12–15","≥15"].map((r, i) => (
                  <div key={r} style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 10, color: C.muted }}>
                    <div style={{ width: 9, height: 9, borderRadius: 2, background: WIND_SPEED_COLORS[i], flexShrink: 0 }} />
                    {r} km/h
                  </div>
                ))}
              </div>
            </Card>
          )}

          {!loading && (Object.keys(stats).length > 0 || (compareMode && !loading2 && Object.keys(stats2).length > 0)) && (
            <Card>
              <SectionLabel>Estadísticas del período</SectionLabel>
              <table style={{ width: "100%", fontSize: 11, borderCollapse: "collapse", marginTop: 8 }}>
                <thead>
                  {compareMode ? (
                    <>
                      <tr>
                        <th style={{ textAlign: "left", color: C.muted, fontWeight: 600, paddingBottom: 2, fontSize: 10 }} rowSpan={2}>Variable</th>
                        <th style={{ textAlign: "center", color: C.deepGreen, fontWeight: 600, paddingBottom: 2, fontSize: 10, borderBottom: `1px solid ${C.paleMint}` }} colSpan={3}>Período 1</th>
                        <th style={{ textAlign: "center", color: C.amber, fontWeight: 600, paddingBottom: 2, fontSize: 10, borderBottom: `1px solid ${C.paleMint}` }} colSpan={3}>Período 2</th>
                      </tr>
                      <tr>
                        {["Media", "Mín", "Máx", "Media", "Mín", "Máx"].map((h, i) => (
                          <th key={`${h}-${i}`} style={{ textAlign: "right", color: C.muted, fontWeight: 600, paddingBottom: 4, fontSize: 10 }}>{h}</th>
                        ))}
                      </tr>
                    </>
                  ) : (
                    <tr>
                      {["Variable", "Media", "Mín", "Máx"].map(h => (
                        <th key={h} style={{ textAlign: h === "Variable" ? "left" : "right", color: C.muted, fontWeight: 600, paddingBottom: 4, fontSize: 10 }}>{h}</th>
                      ))}
                    </tr>
                  )}
                </thead>
                <tbody>
                  {[...new Set([...Object.keys(stats), ...(compareMode ? Object.keys(stats2) : [])])].map(key => {
                    const s1 = stats[key];
                    const s2 = compareMode ? stats2[key] : null;
                    return (
                      <tr key={key} style={{ borderTop: `1px solid ${C.paleMint}` }}>
                        <td style={{ color: C.text, padding: "4px 0" }}>{VAR_FRIENDLY[key] || key}</td>
                        <td style={{ textAlign: "right", fontFamily: "monospace", color: C.text }}>{s1?.mean?.toFixed(1) ?? "—"}</td>
                        <td style={{ textAlign: "right", fontFamily: "monospace", color: C.muted }}>{s1?.min?.toFixed(1) ?? "—"}</td>
                        <td style={{ textAlign: "right", fontFamily: "monospace", color: C.muted }}>{s1?.max?.toFixed(1) ?? "—"}</td>
                        {compareMode && (
                          <>
                            <td style={{ textAlign: "right", fontFamily: "monospace", color: C.text }}>{s2?.mean?.toFixed(1) ?? "—"}</td>
                            <td style={{ textAlign: "right", fontFamily: "monospace", color: C.muted }}>{s2?.min?.toFixed(1) ?? "—"}</td>
                            <td style={{ textAlign: "right", fontFamily: "monospace", color: C.muted }}>{s2?.max?.toFixed(1) ?? "—"}</td>
                          </>
                        )}
                      </tr>
                    );
                  })}
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
  const { data: riskData } = useAPI(getFireRiskCurrent, null, []);

  const riskTotal = riskData?.rule_based?.total ? Math.round(riskData.rule_based.total) : null;
  const wx = riskData?.weather || {};

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
          <RiskGauge value={riskTotal ?? 0} />
          {riskData?.rule_based?.label && (
            <div style={{ fontSize: 11, color: C.muted, textAlign: "center", marginTop: 4 }}>
              {riskData.rule_based.label}
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

  const { data: riskCurrent } = useAPI(getFireRiskCurrent, null, []);
  const { data: riskForecast } = useAPI(getFireRiskForecast, transformRiskForecast, []);
  const { data: riskHistory } = useAPI(getFireRiskHistory, transformRiskForecast, []);
  const { data: speciesApiData } = useAPI(getSpeciesSummary, transformSpeciesSummary, []);
  const { data: weatherCurrent } = useAPI(getWeatherCurrent, null, []);

  const riskTotal = riskCurrent?.rule_based?.total ? Math.round(riskCurrent.rule_based.total) : 0;
  const mlVal = riskCurrent?.ml_probability != null ? Math.round(riskCurrent.ml_probability * 100) : null;
  const wx = riskCurrent?.weather || {};
  const speciesChartData = speciesApiData || speciesData;

  // ── Bar chart: fixed Mon–Sun weeks, combined history + forecast ──
  const [weekOffset, setWeekOffset] = useState(0); // 0 = prev+current+next week
  const todayStr = useMemo(() => new Date().toISOString().split("T")[0], []);

  // Returns the ISO date of the Monday of the week containing dateStr
  function getMondayOf(dateStr) {
    const d = new Date(dateStr + "T12:00:00");
    const day = d.getDay(); // 0=Sun
    d.setDate(d.getDate() + (day === 0 ? -6 : 1 - day));
    return d.toISOString().split("T")[0];
  }

  const allRiskData = useMemo(() => {
    const hist = riskHistory || [];
    const fore = riskForecast || [];
    // Merge: forecast takes precedence on same date (today may be in both)
    const map = new Map();
    hist.forEach(d => map.set(d.date, d));
    fore.forEach(d => map.set(d.date, d));
    return Array.from(map.values()).sort((a, b) => a.date.localeCompare(b.date));
  }, [riskHistory, riskForecast]);

  // Build exactly 21 slots (Mon×3 weeks), filled from allRiskData where available
  const DAY_NAMES = ["Dom", "Lun", "Mar", "Mie", "Jue", "Vie", "Sab"];
  const windowData = useMemo(() => {
    const windowStart = new Date(getMondayOf(todayStr) + "T12:00:00");
    windowStart.setDate(windowStart.getDate() + (weekOffset - 1) * 7);
    const lookup = new Map(allRiskData.map(d => [d.date, d]));
    return Array.from({ length: 21 }, (_, i) => {
      const d = new Date(windowStart);
      d.setDate(d.getDate() + i);
      const dateStr = d.toISOString().split("T")[0];
      const found = lookup.get(dateStr);
      const dayNum = d.getDate();
      const dayName = DAY_NAMES[d.getDay()];
      return found
        ? { ...found, diaLabel: `${dayName} ${dayNum}` }
        : { date: dateStr, riesgo: null, color: null, isHistorical: null, diaLabel: `${dayName} ${dayNum}` };
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [allRiskData, weekOffset, todayStr]);

  const todayDiaLabel = useMemo(() => {
    const d = new Date(todayStr + "T12:00:00");
    return `${DAY_NAMES[d.getDay()]} ${d.getDate()}`;
  }, [todayStr]);

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
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* Top row: polar plot (main) | gauge + compass (right) */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 260px", gap: 16, alignItems: "stretch" }}>
            {/* Polar contribution — principal visualization */}
            <Card style={{ display: "flex", flexDirection: "column" }}>
              <SectionLabel>Contribución por factor al riesgo</SectionLabel>
              <div style={{ flex: 1, display: "flex", justifyContent: "center", alignItems: "center", padding: "16px 0" }}>
                <PolarContrib components={riskCurrent?.rule_based} size={430} />
              </div>
            </Card>

            {/* Right column: gauge (top) + compass (bottom), same total height */}
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <Card>
                <SectionLabel>Índice de riesgo actual</SectionLabel>
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                  <div style={{ fontSize: 10, color: C.muted, marginBottom: 2 }}>Reglas (FRI)</div>
                  <RiskGauge value={riskTotal} color={riskCurrent?.rule_based?.color} />
                  {mlVal != null && (
                    <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>
                      ML: <span style={{ fontWeight: 700, color: C.text }}>{mlVal}%</span>
                    </div>
                  )}
                  {mlVal != null && (() => {
                    const diff = Math.abs(riskTotal - mlVal);
                    const agree = diff < 20;
                    return (
                      <div style={{ marginTop: 8, padding: "4px 8px", borderRadius: 6, width: "100%",
                        background: agree ? `${C.medGreen}18` : `${C.amber}20`,
                        borderLeft: `3px solid ${agree ? C.medGreen : C.amber}`, fontSize: 10 }}>
                        <span style={{ fontWeight: 700, color: agree ? C.medGreen : C.amber }}>
                          {agree ? "Métodos alineados" : "Discrepancia"}
                        </span>
                        <span style={{ color: C.muted, marginLeft: 6 }}>Δ {diff} pts</span>
                      </div>
                    );
                  })()}
                </div>
              </Card>
              <Card style={{ flex: 1, display: "flex", flexDirection: "column" }}>
                <SectionLabel>Dirección del viento</SectionLabel>
                <div style={{ flex: 1, display: "flex", justifyContent: "center", alignItems: "center", padding: "8px 0" }}>
                  <WindCompass
                    direction={weatherCurrent?.wind_direction}
                    speed={wx.wind_speed_kmh}
                  />
                </div>
              </Card>
            </div>
          </div>

          {/* Bottom row: history + forecast bar chart — full width */}
          <Card>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <div>
                <SectionLabel>Índice FRI — historial y pronóstico</SectionLabel>
                <div style={{ fontFamily: "'Georgia', serif", fontSize: 14, fontWeight: 700, color: C.text }}>
                  {windowData[0]?.date} → {windowData[20]?.date}
                </div>
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <button onClick={() => setWeekOffset(o => o - 1)}
                  style={{ padding: "5px 12px", borderRadius: 5, border: `1px solid ${C.mint}`,
                    background: C.paleMint, color: C.text, cursor: "pointer", fontSize: 11 }}>
                  ← Sem ant
                </button>
                <button onClick={() => setWeekOffset(0)}
                  style={{ padding: "5px 12px", borderRadius: 5, border: `1px solid ${C.mint}`,
                    background: weekOffset === 0 ? C.mint : C.paleMint, color: C.text,
                    cursor: "pointer", fontSize: 11, fontWeight: weekOffset === 0 ? 700 : 400 }}>
                  Hoy
                </button>
                <button onClick={() => setWeekOffset(o => o + 1)}
                  style={{ padding: "5px 12px", borderRadius: 5, border: `1px solid ${C.mint}`,
                    background: C.paleMint, color: C.text, cursor: "pointer", fontSize: 11 }}>
                  Sem sig →
                </button>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={windowData} margin={{ top: 16, right: 12, left: 0, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={C.paleMint} vertical={false} />
                <XAxis dataKey="diaLabel" tick={{ fontSize: 9, fill: C.muted }} axisLine={{ stroke: C.mint }} interval={0} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: C.muted }} axisLine={{ stroke: C.mint }} />
                <Tooltip
                  contentStyle={{ borderRadius: 6, border: `1px solid ${C.mint}`, fontSize: 11 }}
                  formatter={(v, name) => [v ?? "—", "FRI"]}
                  labelFormatter={(label, payload) => {
                    const d = payload?.[0]?.payload;
                    const suffix = d?.label ? ` — ${d.label}` : "";
                    const hist = d?.isHistorical ? " (hist.)" : d?.isHistorical === false ? " (pron.)" : "";
                    return `${label}${suffix}${hist}`;
                  }}
                />
                <ReferenceLine
                  x={todayDiaLabel}
                  stroke={C.red}
                  strokeDasharray="4 3"
                  strokeWidth={1.5}
                  label={{ value: "hoy", position: "top", fontSize: 9, fill: C.red, fontWeight: 700 }}
                />
                <Bar dataKey="riesgo" radius={[3, 3, 0, 0]} name="riesgo" maxBarSize={36}>
                  {windowData.map((entry, i) => (
                    <Cell
                      key={i}
                      fill={entry.color || (entry.riesgo != null ? C.medGreen : "transparent")}
                      fillOpacity={entry.isHistorical ? 0.55 : entry.isHistorical === false ? 0.9 : 0}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 10 }}>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "4px 14px" }}>
                {[
                  { label: "Bajo", color: "#2e7d32" },
                  { label: "Mod-Bajo", color: "#c0ca33" },
                  { label: "Moderado", color: "#fbc02d" },
                  { label: "Alto", color: "#fb8c00" },
                  { label: "Muy Alto", color: "#e53935" },
                  { label: "Extremo", color: "#b71c1c" },
                ].map(l => (
                  <div key={l.label} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 10, color: C.muted }}>
                    <div style={{ width: 10, height: 10, borderRadius: 2, background: l.color }} />
                    {l.label}
                  </div>
                ))}
              </div>
              <div style={{ fontSize: 10, color: C.lightMuted }}>
                Barras semitransparentes = datos históricos · Sólidas = pronóstico
              </div>
            </div>
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
              <BarChart data={speciesChartData} layout="vertical" margin={{ left: 100 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={C.paleMint} horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 10, fill: C.muted }} axisLine={{ stroke: C.mint }} />
                <YAxis type="category" dataKey="nombre" tick={{ fontSize: 11, fill: C.text }} axisLine={{ stroke: C.mint }} width={100} />
                <Tooltip contentStyle={{ borderRadius: 6, fontSize: 11 }} />
                <Bar dataKey="detecciones" radius={[0, 4, 4, 0]} name="Detecciones">
                  {speciesChartData.map((entry, i) => {
                    const invasive = /sus scrofa|lepus|jabali|liebre/i.test(entry.nombre);
                    const priority = /puma|leopardus|guiña|guina/i.test(entry.nombre);
                    return <Cell key={i} fill={priority ? C.amber : invasive ? C.red : C.deepGreen} />;
                  })}
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
                <StatBlock value={speciesApiData ? speciesApiData.reduce((s, d) => s + d.detecciones, 0) : "—"} label="Total detecciones" />
                <StatBlock value={speciesApiData ? speciesApiData.length : "—"} label="Especies" />
                <StatBlock value={speciesApiData ? "—" : "—"} label="Cámaras activas" />
                <StatBlock value="—" label="Días muestreados" />
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
