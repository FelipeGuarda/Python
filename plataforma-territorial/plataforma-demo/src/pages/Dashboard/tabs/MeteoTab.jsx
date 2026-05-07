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
import styles from "./MeteoTab.module.css";

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
    <div className={styles.container}>
      {/* Current conditions */}
      <Card className={styles.cardSpaced}>
        <SectionLabel>
          Última medición
          {current?.timestamp ? ` — ${new Date(current.timestamp).toLocaleDateString("es-CL", { day: "numeric", month: "long", year: "numeric" })}` : ""}
        </SectionLabel>
        {current ? (
          <div className={styles.currentRow}>
            {[
              { v: current.temperature_air,     decimals: 1, u: "°C",   l: "Temperatura" },
              { v: current.relative_humidity,   decimals: 0, u: "%",    l: "Humedad" },
              { v: current.wind_speed_kmh,      decimals: 1, u: "km/h", l: "Viento" },
              { v: current.BP_mbar_Avg,         decimals: 0, u: " hPa", l: "Presión" },
              { v: current.precipitation,       decimals: 1, u: " mm",  l: "Precip. 15min" },
              { v: current.solar_radiation,     decimals: 0, u: " W/m²",l: "Radiación solar" },
            ].map(item => item.v != null && (
              <div key={item.l} className={styles.currentTile}>
                <div className={styles.currentValue}>
                  {Number(item.v).toFixed(item.decimals)}<span className={styles.currentUnit}>{item.u}</span>
                </div>
                <div className={styles.currentLabel}>{item.l}</div>
              </div>
            ))}
          </div>
        ) : (
          <div className={styles.connecting}>Conectando con la API...</div>
        )}
      </Card>

      <div className={styles.stack}>
          {/* Controls */}
          <Card>
            <SectionLabel>Variables</SectionLabel>
            <div className={styles.varRow}>
              {WEATHER_VARS.map(v => (
                <label key={v.id} className={styles.varCheck}>
                  <input type="checkbox" checked={selectedVars.includes(v.id)} onChange={() => toggleVar(v.id)} />
                  {v.label}
                </label>
              ))}
              <label className={styles.varCheck}>
                <input type="checkbox" checked={showWind} onChange={() => setShowWind(v => !v)} />
                Rosa de vientos
              </label>
            </div>
            <div className={styles.controlsRow}>
              <div className={styles.dateGroup}>
                <span className={styles.dateLabel}>Desde:</span>
                <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)}
                  className={styles.dateInput} />
                <span className={styles.dateLabel}>Hasta:</span>
                <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)}
                  className={styles.dateInput} />
                <button onClick={() => setAppliedDates({ start: startDate, end: endDate })}
                  className={styles.applyBtn}>
                  Aplicar
                </button>
              </div>
              <div className={styles.resGroup}>
                {RESOLUTIONS.map(r => (
                  <label key={r.id} className={styles.resLabel}>
                    <input type="radio" name="resolution" value={r.id} checked={resolution === r.id} onChange={() => setResolution(r.id)} />
                    {r.label}
                  </label>
                ))}
              </div>
            </div>

            {/* Comparison mode toggle + Period 2 date range */}
            <div className={styles.compareRow}>
              <label className={styles.compareCheck}>
                <input type="checkbox" checked={compareMode} onChange={() => setCompareMode(v => !v)} />
                Comparar períodos
              </label>
              {compareMode && (
                <div className={styles.dateGroup}>
                  <span className={styles.amberHeader}>Período 2:</span>
                  <span className={styles.dateLabel}>Desde:</span>
                  <input type="date" value={startDate2} onChange={e => setStartDate2(e.target.value)}
                    className={styles.dateInputAmber} />
                  <span className={styles.dateLabel}>Hasta:</span>
                  <input type="date" value={endDate2} onChange={e => setEndDate2(e.target.value)}
                    className={styles.dateInputAmber} />
                  <button onClick={() => setAppliedDates2({ start: startDate2, end: endDate2 })}
                    className={styles.applyBtnAmber}>
                    Aplicar
                  </button>
                </div>
              )}
            </div>
          </Card>

          {(loading || (compareMode && loading2)) && (
            <div className={styles.loading}>Cargando datos...</div>
          )}

          {/* Per-variable charts */}
          {!loading && selectedVars.map(varId => {
            const conf = varConf(varId);
            const chartData = histData.filter(d => d[varId] != null);
            const chartData2 = compareMode ? histData2.filter(d => d[varId] != null) : [];
            if (chartData.length === 0 && chartData2.length === 0) return null;
            return (
              <Card key={varId}>
                {compareMode && <div className={styles.periodLabelGreen}>Período 1</div>}
                <SectionLabel>{conf.label} ({conf.unit})</SectionLabel>
                {chartData.length > 0 && (
                  <ResponsiveContainer width="100%" height={150}>
                    {renderVarChart(varId, chartData)}
                  </ResponsiveContainer>
                )}
                {compareMode && (
                  <>
                    <div className={styles.periodLabelAmberInChart}>Período 2</div>
                    {chartData2.length > 0 ? (
                      <ResponsiveContainer width="100%" height={150}>
                        {renderVarChart(varId, chartData2, C.amber)}
                      </ResponsiveContainer>
                    ) : !loading2 && (
                      <div className={styles.noData}>Sin datos para este período</div>
                    )}
                  </>
                )}
              </Card>
            );
          })}

          {showWind && (windRose || (compareMode && windRose2)) && (
            <Card>
              <SectionLabel>Rosa de vientos</SectionLabel>
              <div className={styles.windRoseRow}>
                {windRose && (
                  <div className={styles.windRoseItem}>
                    {compareMode && <div className={styles.windRoseLabelGreen}>Período 1</div>}
                    <WindRose data={windRose} size={compareMode ? 280 : 350} />
                  </div>
                )}
                {compareMode && windRose2 && (
                  <div className={styles.windRoseItem}>
                    <div className={styles.windRoseLabelAmber}>Período 2</div>
                    <WindRose data={windRose2} size={280} />
                  </div>
                )}
              </div>
              <div className={styles.windLegend}>
                {["0–3","3–6","6–9","9–12","12–15","≥15"].map((r, i) => (
                  <div key={r} className={styles.windLegendItem}>
                    <div className={styles.windLegendSwatch} style={{ background: WIND_SPEED_COLORS[i] }} />
                    {r} km/h
                  </div>
                ))}
              </div>
            </Card>
          )}

          {!loading && (Object.keys(stats).length > 0 || (compareMode && !loading2 && Object.keys(stats2).length > 0)) && (
            <Card>
              <SectionLabel>Estadísticas del período</SectionLabel>
              <table className={styles.statsTable}>
                <thead>
                  {compareMode ? (
                    <>
                      <tr>
                        <th className={styles.thVarRowSpan} rowSpan={2}>Variable</th>
                        <th className={styles.thPeriodGreen} colSpan={3}>Período 1</th>
                        <th className={styles.thPeriodAmber} colSpan={3}>Período 2</th>
                      </tr>
                      <tr>
                        {["Media", "Mín", "Máx", "Media", "Mín", "Máx"].map((h, i) => (
                          <th key={`${h}-${i}`} className={styles.thRight}>{h}</th>
                        ))}
                      </tr>
                    </>
                  ) : (
                    <tr>
                      {["Variable", "Media", "Mín", "Máx"].map(h => (
                        <th key={h} className={h === "Variable" ? styles.thLeft : styles.thRight}>{h}</th>
                      ))}
                    </tr>
                  )}
                </thead>
                <tbody>
                  {[...new Set([...Object.keys(stats), ...(compareMode ? Object.keys(stats2) : [])])].map(key => {
                    const s1 = stats[key];
                    const s2 = compareMode ? stats2[key] : null;
                    return (
                      <tr key={key} className={styles.statRow}>
                        <td className={styles.statName}>{VAR_FRIENDLY[key] || key}</td>
                        <td className={styles.statNum}>{s1?.mean?.toFixed(1) ?? "—"}</td>
                        <td className={styles.statNumMuted}>{s1?.min?.toFixed(1) ?? "—"}</td>
                        <td className={styles.statNumMuted}>{s1?.max?.toFixed(1) ?? "—"}</td>
                        {compareMode && (
                          <>
                            <td className={styles.statNum}>{s2?.mean?.toFixed(1) ?? "—"}</td>
                            <td className={styles.statNumMuted}>{s2?.min?.toFixed(1) ?? "—"}</td>
                            <td className={styles.statNumMuted}>{s2?.max?.toFixed(1) ?? "—"}</td>
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
              <div className={styles.empty}>
                Selecciona variables,<br />define el período<br />y haz clic en Aplicar.
              </div>
            </Card>
          )}
      </div>
    </div>
  );
}
