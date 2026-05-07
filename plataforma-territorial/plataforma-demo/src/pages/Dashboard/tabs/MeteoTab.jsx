import { useState, useEffect } from "react";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { getWeatherCurrent, getWeatherHistoryRange } from "../../../api.js";
import { C } from "../../../constants/colors.js";
import { WEATHER_VARS, WIND_SPEED_COLORS, RESOLUTIONS, VAR_FRIENDLY } from "../../../constants/weather_vars.js";
import {
  CHART_TICK_SM,
  CHART_AXIS_LINE,
  CHART_GRID,
  CHART_TOOLTIP_BORDERED,
} from "../../../styles/chart.js";
import { Card } from "../../../components/Card.jsx";
import { SectionLabel } from "../../../components/SectionLabel.jsx";
import { WindRose } from "../../../components/WindRose.jsx";

const MONTHS = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"];

// ── METEO TAB ──
export function MeteoTab() {
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
    getWeatherCurrent().then(setCurrent).catch(() => {});
  }, []);

  useEffect(() => {
    if (!appliedDates.start || !appliedDates.end) return;
    setLoading(true);
    const windExtras = showWind
      ? ["wind_speed", "wind_direction"].filter(v => !selectedVars.includes(v))
      : [];
    const varList = [...selectedVars, ...windExtras].join(",");
    const p = new URLSearchParams({ start: appliedDates.start, end: appliedDates.end, resolution, variables: varList });
    getWeatherHistoryRange(p)
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
    const windExtras = showWind
      ? ["wind_speed", "wind_direction"].filter(v => !selectedVars.includes(v))
      : [];
    const varList = [...selectedVars, ...windExtras].join(",");
    const p = new URLSearchParams({ start: appliedDates2.start, end: appliedDates2.end, resolution, variables: varList });
    getWeatherHistoryRange(p)
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
      tick={CHART_TICK_SM} interval="preserveStartEnd"
      axisLine={CHART_AXIS_LINE} tickLine={false} />
  );
  const commonTooltip = (label, unit) => (
    <Tooltip contentStyle={CHART_TOOLTIP_BORDERED}
      labelFormatter={tickFmt}
      formatter={v => [v != null ? v.toFixed(2) : "—", label]} />
  );

  const renderVarChart = (varId, data, colorOverride) => {
    const conf = varConf(varId);
    const color = colorOverride || conf.color;
    return conf.type === "bar" ? (
      <BarChart data={data} {...commonChartProps}>
        <CartesianGrid {...CHART_GRID} />
        {commonXAxis}
        <YAxis tick={CHART_TICK_SM} axisLine={CHART_AXIS_LINE} tickLine={false} unit={` ${conf.unit}`} width={52} />
        {commonTooltip(conf.label, conf.unit)}
        <Bar dataKey={varId} fill={color} radius={[2, 2, 0, 0]} />
      </BarChart>
    ) : (
      <LineChart data={data} {...commonChartProps}>
        <CartesianGrid {...CHART_GRID} />
        {commonXAxis}
        <YAxis tick={CHART_TICK_SM} axisLine={CHART_AXIS_LINE} tickLine={false} unit={` ${conf.unit}`} width={52} />
        {commonTooltip(conf.label, conf.unit)}
        <Line type="linear" dataKey={varId} stroke={color} strokeWidth={1.5} dot={false} connectNulls={false} />
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
                Rosa de vientos
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
