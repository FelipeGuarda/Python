import { BarChart, Bar, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { C, SP_COLORS } from "../../../constants/colors.js";
import { Card } from "../../../components/Card.jsx";
import { SectionLabel } from "../../../components/SectionLabel.jsx";
import { StatBlock } from "../../../components/StatBlock.jsx";
import { SpeciesMap } from "../../../components/SpeciesMap.jsx";

export function CamarasTab({
  dielData, ctStats, speciesList,
  sp1Sel, setSp1Sel, sp2Sel, setSp2Sel,
  applyOverlap, overlapLoading, overlapData,
  totalStations, camBoundary, geo,
  isInvasive, isPriority,
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

      {/* Row 1: diel activity (all species) + summary stats */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <Card>
          <SectionLabel>Actividad por hora del día</SectionLabel>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={dielData || []} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.paleMint} />
              <XAxis dataKey="hora" tick={{ fontSize: 9, fill: C.muted }} axisLine={{ stroke: C.mint }} interval={2} />
              <YAxis tick={{ fontSize: 10, fill: C.muted }} axisLine={{ stroke: C.mint }} />
              <Tooltip contentStyle={{ borderRadius: 6, fontSize: 11 }} formatter={(v) => [v, "Detecciones"]} />
              <Bar dataKey="actividad" fill={C.deepGreen} radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <div style={{ fontSize: 10, color: C.muted, marginTop: 6, fontStyle: "italic" }}>
            Todas las especies · {ctStats ? `${ctStats.total_detections} detecciones totales` : "cargando…"}
          </div>
        </Card>
        <Card>
          <SectionLabel>Resumen del monitoreo</SectionLabel>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 10 }}>
            <StatBlock value={ctStats?.total_detections ?? "…"} label="Detecciones" />
            <StatBlock value={ctStats?.unique_species ?? "…"} label="Especies" />
            <StatBlock value={ctStats?.active_stations ?? "…"} label="Estaciones" />
            <StatBlock value={ctStats?.campaign_count ?? "…"} label="Campañas" />
          </div>
          {ctStats?.date_range_start && (
            <div style={{ fontSize: 10, color: C.muted, marginTop: 10, borderTop: `1px solid ${C.paleMint}`, paddingTop: 8 }}>
              <div>Período: {ctStats.date_range_start.slice(0, 10)} — {ctStats.date_range_end.slice(0, 10)}</div>
              <div style={{ marginTop: 2 }}>{ctStats.days_sampled} días con registros</div>
              <div style={{ marginTop: 2 }}>{ctStats.campaigns?.join(" · ")}</div>
            </div>
          )}
        </Card>
      </div>

      {/* Species selector */}
      <Card>
        <SectionLabel>Comparador de Especies</SectionLabel>
        <div style={{ display: "flex", gap: 16, alignItems: "flex-end", marginTop: 10, flexWrap: "wrap" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontSize: 10, color: SP_COLORS[0], fontWeight: 700 }}>Especie A</span>
            <select value={sp1Sel} onChange={e => setSp1Sel(e.target.value)} style={{
              padding: "6px 10px", borderRadius: 6, border: `2px solid ${SP_COLORS[0]}`,
              fontSize: 12, color: C.text, background: C.white, minWidth: 200, cursor: "pointer",
            }}>
              {speciesList.map(s => (
                <option key={s.scientific_name} value={s.scientific_name}>{s.common_name}</option>
              ))}
            </select>
          </div>
          <div style={{ fontSize: 13, color: C.lightMuted, paddingBottom: 6 }}>vs.</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontSize: 10, color: SP_COLORS[1], fontWeight: 700 }}>Especie B</span>
            <select value={sp2Sel} onChange={e => setSp2Sel(e.target.value)} style={{
              padding: "6px 10px", borderRadius: 6, border: `2px solid ${SP_COLORS[1]}`,
              fontSize: 12, color: C.text, background: C.white, minWidth: 200, cursor: "pointer",
            }}>
              {speciesList.map(s => (
                <option key={s.scientific_name} value={s.scientific_name}>{s.common_name}</option>
              ))}
            </select>
          </div>
          <button onClick={applyOverlap}
            disabled={!sp1Sel || !sp2Sel || sp1Sel === sp2Sel || overlapLoading}
            style={{
              padding: "7px 22px",
              background: (!sp1Sel || !sp2Sel || sp1Sel === sp2Sel) ? C.lightMuted : C.deepGreen,
              color: C.white, border: "none", borderRadius: 6,
              cursor: (!sp1Sel || !sp2Sel || sp1Sel === sp2Sel || overlapLoading) ? "not-allowed" : "pointer",
              fontSize: 13, fontWeight: 600, fontFamily: "'Trebuchet MS', sans-serif",
              paddingBottom: 7,
            }}>
            {overlapLoading ? "Calculando…" : "Aplicar"}
          </button>
          {sp1Sel && sp2Sel && sp1Sel === sp2Sel && (
            <span style={{ fontSize: 11, color: C.red, paddingBottom: 6 }}>Selecciona dos especies distintas</span>
          )}
        </div>
      </Card>

      {/* Loading placeholder */}
      {overlapLoading && (
        <Card>
          <div style={{ fontSize: 13, color: C.muted, textAlign: "center", padding: "36px 0" }}>
            Calculando solapamiento…
          </div>
        </Card>
      )}

      {/* Overlap activity chart — full width */}
      {overlapData && !overlapLoading && (
        <Card>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
            <SectionLabel>Actividad diaria — superposición</SectionLabel>
            <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
              {[overlapData.sp1_name, overlapData.sp2_name].map((name, i) => (
                <div key={i} style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  <div style={{ width: 16, height: 3, background: SP_COLORS[i], borderRadius: 2 }} />
                  <span style={{ fontSize: 11, color: C.muted }}>{name}</span>
                </div>
              ))}
              <div style={{
                padding: "3px 12px", borderRadius: 20,
                background: C.paleMint, fontSize: 11, color: C.text,
              }}>
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
              <CartesianGrid strokeDasharray="3 3" stroke={C.paleMint} />
              <XAxis dataKey="hour" tick={{ fontSize: 9, fill: C.muted }} axisLine={{ stroke: C.mint }}
                tickFormatter={h => `${String(h).padStart(2, "0")}h`} interval={2} />
              <YAxis tick={{ fontSize: 10, fill: C.muted }} axisLine={{ stroke: C.mint }} />
              <Tooltip
                contentStyle={{ borderRadius: 6, fontSize: 11 }}
                labelFormatter={h => `${String(h).padStart(2, "0")}:00 h`}
                formatter={(v, key) => [v, key === "sp1" ? overlapData.sp1_name : overlapData.sp2_name]}
              />
              <Area type="monotone" dataKey="sp1" stroke={SP_COLORS[0]} strokeWidth={2}
                fill="url(#gradSp1)" dot={false} />
              <Area type="monotone" dataKey="sp2" stroke={SP_COLORS[1]} strokeWidth={2}
                fill="url(#gradSp2)" dot={false} />
            </AreaChart>
          </ResponsiveContainer>
          <div style={{ fontSize: 10, color: C.muted, marginTop: 4, fontStyle: "italic" }}>
            {overlapData.total_sp1} detecciones de {overlapData.sp1_name} · {overlapData.total_sp2} de {overlapData.sp2_name}
          </div>
        </Card>
      )}

      {/* Detection maps — side by side */}
      {overlapData && !overlapLoading && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          {[
            { name: overlapData.sp1_name, stations: overlapData.stations_sp1, occ: overlapData.occupancy_sp1, ci: 0 },
            { name: overlapData.sp2_name, stations: overlapData.stations_sp2, occ: overlapData.occupancy_sp2, ci: 1 },
          ].map(({ name, stations, occ, ci }) => (
            <Card key={ci} style={{ padding: 12 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                <SectionLabel style={{ margin: 0 }}>{name}</SectionLabel>
                <div style={{ fontSize: 11, color: C.muted, textAlign: "right" }}>
                  <span style={{ color: SP_COLORS[ci], fontWeight: 700 }}>{occ}</span>
                  <span style={{ color: C.lightMuted }}> / {totalStations ?? "…"} est.</span>
                  {totalStations && (
                    <span style={{ marginLeft: 8, padding: "2px 8px", borderRadius: 10,
                      background: `${SP_COLORS[ci]}20`, color: SP_COLORS[ci], fontWeight: 600, fontSize: 10 }}>
                      {Math.round(occ / totalStations * 100)}%
                    </span>
                  )}
                </div>
              </div>
              <div style={{ width: "100%", aspectRatio: "1 / 1" }}>
                <SpeciesMap boundary={camBoundary} stations={stations} colorIdx={ci} center={geo?.reserve?.center} />
              </div>
              <div style={{ fontSize: 10, color: C.lightMuted, marginTop: 6, fontStyle: "italic" }}>
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
              <CartesianGrid strokeDasharray="3 3" stroke={C.paleMint} horizontal={false} />
              <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 10, fill: C.muted }}
                axisLine={{ stroke: C.mint }} tickFormatter={v => `${v}%`} />
              <YAxis type="category" dataKey="common_name" tick={{ fontSize: 11, fill: C.text }}
                axisLine={{ stroke: C.mint }} width={150} />
              <Tooltip contentStyle={{ borderRadius: 6, fontSize: 11 }}
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
          <div style={{ display: "flex", gap: 16, marginTop: 8 }}>
            {[["Nativa", C.deepGreen], ["Prioritaria", C.amber], ["Invasora / introducida", C.red]].map(([label, color]) => (
              <div key={label} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 10, color: C.muted }}>
                <div style={{ width: 10, height: 10, borderRadius: 2, background: color }} /> {label}
              </div>
            ))}
          </div>
        </Card>
      )}

    </div>
  );
}
