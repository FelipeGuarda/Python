import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer, Cell } from "recharts";
import { C } from "../../../constants/colors.js";
import {
  CHART_TICK_SM,
  CHART_TICK_MD,
  CHART_AXIS_LINE,
  CHART_GRID,
  CHART_TOOLTIP_BORDERED,
} from "../../../styles/chart.js";
import { Card } from "../../../components/Card.jsx";
import { SectionLabel } from "../../../components/SectionLabel.jsx";
import { PolarContrib } from "../../../components/PolarContrib.jsx";
import { RiskGauge } from "../../../components/RiskGauge.jsx";
import { WindCompass } from "../../../components/WindCompass.jsx";

export function RiesgoTab({ riskCurrent, riskTotal, mlVal, wx, windowData, todayDiaLabel }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Top row: polar plot (main) | gauge + compass (right) */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 260px", gap: 16, alignItems: "stretch" }}>
        {/* Polar contribution — principal visualization */}
        <Card style={{ display: "flex", flexDirection: "column" }}>
          <SectionLabel>Contribución por factor al riesgo</SectionLabel>
          <div style={{ flex: 1, display: "flex", justifyContent: "center", alignItems: "center", padding: "16px 0" }}>
            <PolarContrib components={riskCurrent?.rule_based} weather={riskCurrent?.weather} size={430} color={riskCurrent?.rule_based?.color || C.amber} />
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
              {riskCurrent?.timestamp && (
                <div style={{ fontSize: 10, color: C.lightMuted, marginTop: 8 }}>
                  Datos del{" "}
                  {new Date(riskCurrent.timestamp).toLocaleDateString("es-CL", { day: "numeric", month: "short" })}{" "}
                  {new Date(riskCurrent.timestamp).toLocaleTimeString("es-CL", { hour: "2-digit", minute: "2-digit" })}
                </div>
              )}
            </div>
          </Card>
          <Card style={{ flex: 1, display: "flex", flexDirection: "column" }}>
            <SectionLabel>Dirección del viento</SectionLabel>
            <div style={{ flex: 1, display: "flex", justifyContent: "center", alignItems: "center", padding: "8px 0" }}>
              <WindCompass
                direction={riskCurrent?.weather?.wind_direction}
                speed={wx.wind_speed_kmh}
              />
            </div>
            {riskCurrent?.timestamp && (
              <div style={{ fontSize: 10, color: C.lightMuted, textAlign: "center", marginTop: 4 }}>
                Datos del{" "}
                {new Date(riskCurrent.timestamp).toLocaleDateString("es-CL", { day: "numeric", month: "short" })}{" "}
                {new Date(riskCurrent.timestamp).toLocaleTimeString("es-CL", { hour: "2-digit", minute: "2-digit" })}
              </div>
            )}
          </Card>
        </div>
      </div>

      {/* Bottom row: history + forecast bar chart — full width */}
      <Card>
        <div style={{ marginBottom: 8 }}>
          <SectionLabel>Índice FRI — historial y pronóstico</SectionLabel>
          <div style={{ fontFamily: "'Georgia', serif", fontSize: 14, fontWeight: 700, color: C.text }}>
            {windowData[0]?.date} → {windowData[20]?.date}
          </div>
        </div>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={windowData} margin={{ top: 16, right: 12, left: 0, bottom: 4 }}>
            <CartesianGrid {...CHART_GRID} vertical={false} />
            <XAxis dataKey="diaLabel" tick={CHART_TICK_SM} axisLine={CHART_AXIS_LINE} interval={0} />
            <YAxis domain={[0, 100]} tick={CHART_TICK_MD} axisLine={CHART_AXIS_LINE} />
            <Tooltip
              contentStyle={CHART_TOOLTIP_BORDERED}
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
  );
}
