import { BarChart, Bar, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { C, SP_COLORS } from "../../../constants/colors.js";
import {
  CHART_TICK_SM,
  CHART_TICK_MD,
  CHART_TICK_CAT,
  CHART_AXIS_LINE,
  CHART_GRID,
  CHART_TOOLTIP,
} from "../../../styles/chart.js";
import { Card } from "../../../components/Card.jsx";
import { SectionLabel } from "../../../components/SectionLabel.jsx";
import { StatBlock } from "../../../components/StatBlock.jsx";
import { SpeciesMap } from "../../../components/SpeciesMap.jsx";
import styles from "./CamarasTab.module.css";

export function CamarasTab({
  dielData, ctStats, speciesList,
  sp1Sel, setSp1Sel, sp2Sel, setSp2Sel,
  applyOverlap, overlapLoading, overlapData,
  totalStations, camBoundary, geo,
  isInvasive, isPriority,
}) {
  const applyDisabled = !sp1Sel || !sp2Sel || sp1Sel === sp2Sel;
  return (
    <div className={styles.container}>

      {/* Row 1: diel activity (all species) + summary stats */}
      <div className={styles.row2col}>
        <Card>
          <SectionLabel>Actividad por hora del día</SectionLabel>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={dielData || []} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid {...CHART_GRID} />
              <XAxis dataKey="hora" tick={CHART_TICK_SM} axisLine={CHART_AXIS_LINE} interval={2} />
              <YAxis tick={CHART_TICK_MD} axisLine={CHART_AXIS_LINE} />
              <Tooltip contentStyle={CHART_TOOLTIP} formatter={(v) => [v, "Detecciones"]} />
              <Bar dataKey="actividad" fill={C.deepGreen} radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <div className={styles.captionItalic}>
            Todas las especies · {ctStats ? `${ctStats.total_detections} detecciones totales` : "cargando…"}
          </div>
        </Card>
        <Card>
          <SectionLabel>Resumen del monitoreo</SectionLabel>
          <div className={styles.statsGrid}>
            <StatBlock value={ctStats?.total_detections ?? "…"} label="Detecciones" />
            <StatBlock value={ctStats?.unique_species ?? "…"} label="Especies" />
            <StatBlock value={ctStats?.active_stations ?? "…"} label="Estaciones" />
            <StatBlock value={ctStats?.campaign_count ?? "…"} label="Campañas" />
          </div>
          {ctStats?.date_range_start && (
            <div className={styles.periodInfo}>
              <div>Período: {ctStats.date_range_start.slice(0, 10)} — {ctStats.date_range_end.slice(0, 10)}</div>
              <div className={styles.periodInfoSub}>{ctStats.days_sampled} días con registros</div>
              <div className={styles.periodInfoSub}>{ctStats.campaigns?.join(" · ")}</div>
            </div>
          )}
        </Card>
      </div>

      {/* Species selector */}
      <Card>
        <SectionLabel>Comparador de Especies</SectionLabel>
        <div className={styles.selectorRow}>
          <div className={styles.selectorCol}>
            <span className={styles.spLabel} style={{ color: SP_COLORS[0] }}>Especie A</span>
            <select value={sp1Sel} onChange={e => setSp1Sel(e.target.value)}
              className={styles.spSelect} style={{ borderColor: SP_COLORS[0] }}>
              {speciesList.map(s => (
                <option key={s.scientific_name} value={s.scientific_name}>{s.common_name}</option>
              ))}
            </select>
          </div>
          <div className={styles.vsLabel}>vs.</div>
          <div className={styles.selectorCol}>
            <span className={styles.spLabel} style={{ color: SP_COLORS[1] }}>Especie B</span>
            <select value={sp2Sel} onChange={e => setSp2Sel(e.target.value)}
              className={styles.spSelect} style={{ borderColor: SP_COLORS[1] }}>
              {speciesList.map(s => (
                <option key={s.scientific_name} value={s.scientific_name}>{s.common_name}</option>
              ))}
            </select>
          </div>
          <button onClick={applyOverlap}
            disabled={applyDisabled || overlapLoading}
            className={styles.applyBtn}
            style={{
              background: applyDisabled ? C.lightMuted : C.deepGreen,
              cursor: (applyDisabled || overlapLoading) ? "not-allowed" : "pointer",
            }}>
            {overlapLoading ? "Calculando…" : "Aplicar"}
          </button>
          {sp1Sel && sp2Sel && sp1Sel === sp2Sel && (
            <span className={styles.errMsg}>Selecciona dos especies distintas</span>
          )}
        </div>
      </Card>

      {/* Loading placeholder */}
      {overlapLoading && (
        <Card>
          <div className={styles.loadingMsg}>
            Calculando solapamiento…
          </div>
        </Card>
      )}

      {/* Overlap activity chart — full width */}
      {overlapData && !overlapLoading && (
        <Card>
          <div className={styles.overlapHeader}>
            <SectionLabel>Actividad diaria — superposición</SectionLabel>
            <div className={styles.legendGroup}>
              {[overlapData.sp1_name, overlapData.sp2_name].map((name, i) => (
                <div key={i} className={styles.legendItem}>
                  <div className={styles.legendDash} style={{ background: SP_COLORS[i] }} />
                  <span className={styles.spName}>{name}</span>
                </div>
              ))}
              <div className={styles.chipPaleMint}>
                Solapamiento: <strong>{Math.round(overlapData.overlap_coeff * 100)}%</strong>
              </div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={230}>
            <AreaChart data={overlapData.chart} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="gradSp1" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={SP_COLORS[0]} stopOpacity={0.4} />
                  <stop offset="95%" stopColor={SP_COLORS[0]} stopOpacity={0.02} />
                </linearGradient>
                <linearGradient id="gradSp2" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={SP_COLORS[1]} stopOpacity={0.4} />
                  <stop offset="95%" stopColor={SP_COLORS[1]} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid {...CHART_GRID} />
              <XAxis dataKey="hour" tick={CHART_TICK_SM} axisLine={CHART_AXIS_LINE}
                tickFormatter={h => `${String(h).padStart(2, "0")}h`} interval={2} />
              <YAxis tick={CHART_TICK_MD} axisLine={CHART_AXIS_LINE} />
              <Tooltip
                contentStyle={CHART_TOOLTIP}
                labelFormatter={h => `${String(h).padStart(2, "0")}:00 h`}
                formatter={(v, key) => [v, key === "sp1" ? overlapData.sp1_name : overlapData.sp2_name]}
              />
              <Area type="monotone" dataKey="sp1" stroke={SP_COLORS[0]} strokeWidth={2}
                fill="url(#gradSp1)" dot={false} />
              <Area type="monotone" dataKey="sp2" stroke={SP_COLORS[1]} strokeWidth={2}
                fill="url(#gradSp2)" dot={false} />
            </AreaChart>
          </ResponsiveContainer>
          <div className={styles.captionItalicSm}>
            {overlapData.total_sp1} detecciones de {overlapData.sp1_name} · {overlapData.total_sp2} de {overlapData.sp2_name}
          </div>
        </Card>
      )}

      {/* Detection maps — side by side */}
      {overlapData && !overlapLoading && (
        <div className={styles.row2col}>
          {[
            { name: overlapData.sp1_name, stations: overlapData.stations_sp1, occ: overlapData.occupancy_sp1, ci: 0 },
            { name: overlapData.sp2_name, stations: overlapData.stations_sp2, occ: overlapData.occupancy_sp2, ci: 1 },
          ].map(({ name, stations, occ, ci }) => (
            <Card key={ci} style={{ padding: 12 }}>
              <div className={styles.mapHeader}>
                <SectionLabel style={{ margin: 0 }}>{name}</SectionLabel>
                <div className={styles.countText}>
                  <span className={styles.countNum} style={{ color: SP_COLORS[ci] }}>{occ}</span>
                  <span className={styles.countDenom}> / {totalStations ?? "…"} est.</span>
                  {totalStations && (
                    <span className={styles.countChip}
                      style={{ background: `${SP_COLORS[ci]}20`, color: SP_COLORS[ci] }}>
                      {Math.round(occ / totalStations * 100)}%
                    </span>
                  )}
                </div>
              </div>
              <div className={styles.mapBox}>
                <SpeciesMap boundary={camBoundary} stations={stations} colorIdx={ci} center={geo?.reserve?.center} />
              </div>
              <div className={styles.captionLightMuted}>
                × = sin detección · Tamaño de burbuja proporcional al número de registros
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Naive occupancy — all species */}
      {speciesList.length > 0 && (
        <Card>
          <SectionLabel>Ocupación por especie — % de estaciones con detecciones</SectionLabel>
          <ResponsiveContainer width="100%" height={Math.max(180, speciesList.length * 22)}>
            <BarChart data={speciesList} layout="vertical"
              margin={{ left: 150, right: 50, top: 4, bottom: 4 }}>
              <CartesianGrid {...CHART_GRID} horizontal={false} />
              <XAxis type="number" domain={[0, 100]} tick={CHART_TICK_MD}
                axisLine={CHART_AXIS_LINE} tickFormatter={v => `${v}%`} />
              <YAxis type="category" dataKey="common_name" tick={CHART_TICK_CAT}
                axisLine={CHART_AXIS_LINE} width={150} />
              <Tooltip contentStyle={CHART_TOOLTIP}
                formatter={(v, _n, p) => [`${v}% (${p.payload.n_stations}/${totalStations ?? "…"} estaciones)`, "Ocupación"]} />
              <Bar dataKey="occupancy_pct" radius={[0, 4, 4, 0]}
                label={{ position: "right", fontSize: 10, fill: C.muted,
                  formatter: (v) => `${v}%` }}>
                {speciesList.map((entry, i) => {
                  const fill = isPriority(entry.scientific_name) ? C.amber
                             : isInvasive(entry.scientific_name) ? C.red
                             : C.deepGreen;
                  return <Cell key={i} fill={fill} />;
                })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className={styles.occLegendRow}>
            {[["Nativa", C.deepGreen], ["Prioritaria", C.amber], ["Invasora / introducida", C.red]].map(([label, color]) => (
              <div key={label} className={styles.occLegendItem}>
                <div className={styles.swatchSm} style={{ background: color }} /> {label}
              </div>
            ))}
          </div>
        </Card>
      )}

    </div>
  );
}
