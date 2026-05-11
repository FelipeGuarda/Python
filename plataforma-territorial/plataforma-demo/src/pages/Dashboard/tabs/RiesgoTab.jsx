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
import styles from "./RiesgoTab.module.css";

const POLAR_CARD_STYLE = { display: "flex", flexDirection: "column" };
const COMPASS_CARD_STYLE = { flex: 1, display: "flex", flexDirection: "column" };

export function RiesgoTab({ riskCurrent, riskTotal, mlVal, wx, windowData, todayDiaLabel }) {
  return (
    <div className={styles.container}>
      {/* Top row: polar plot (main) | gauge + compass (right) */}
      <div className={styles.topGrid}>
        {/* Polar contribution — principal visualization */}
        <Card style={POLAR_CARD_STYLE}>
          <SectionLabel>Contribución por factor al riesgo</SectionLabel>
          <div className={styles.polarBox}>
            <PolarContrib components={riskCurrent?.rule_based} weather={riskCurrent?.weather} size={430} color={riskCurrent?.rule_based?.color || C.amber} />
          </div>
        </Card>

        {/* Right column: gauge (top) + compass (bottom), same total height */}
        <div className={styles.rightCol}>
          <Card>
            <SectionLabel>Índice de riesgo actual</SectionLabel>
            <div className={styles.gaugeCol}>
              <div className={styles.gaugeMethod}>Reglas (FRI)</div>
              <RiskGauge
                value={riskTotal}
                color={riskCurrent?.rule_based?.color}
                label={riskCurrent?.rule_based?.label}
              />
              {mlVal != null && (
                <div className={styles.gaugeMlLine}>
                  ML: <span className={styles.gaugeMlVal}>{mlVal}%</span>
                </div>
              )}
              {mlVal != null && (() => {
                const diff = Math.abs(riskTotal - mlVal);
                const agree = diff < 20;
                return (
                  <div className={styles.agreementBox}
                    style={{
                      background: agree ? `${C.medGreen}18` : `${C.amber}20`,
                      borderLeftColor: agree ? C.medGreen : C.amber,
                    }}>
                    <span className={styles.agreementText} style={{ color: agree ? C.medGreen : C.amber }}>
                      {agree ? "Métodos alineados" : "Discrepancia"}
                    </span>
                    <span className={styles.agreementDelta}>Δ {diff} pts</span>
                  </div>
                );
              })()}
              {riskCurrent?.timestamp && (
                <div className={styles.gaugeTimestamp}>
                  Datos del{" "}
                  {new Date(riskCurrent.timestamp).toLocaleDateString("es-CL", { day: "numeric", month: "short" })}{" "}
                  {new Date(riskCurrent.timestamp).toLocaleTimeString("es-CL", { hour: "2-digit", minute: "2-digit" })}
                </div>
              )}
            </div>
          </Card>
          <Card style={COMPASS_CARD_STYLE}>
            <SectionLabel>Dirección del viento</SectionLabel>
            <div className={styles.compassBox}>
              <WindCompass
                direction={riskCurrent?.weather?.wind_direction}
                speed={wx.wind_speed_kmh}
              />
            </div>
            {riskCurrent?.timestamp && (
              <div className={styles.compassTimestamp}>
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
        <div className={styles.chartHeader}>
          <SectionLabel>Índice FRI — historial y pronóstico</SectionLabel>
          <div className={styles.chartTitle}>
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
        <div className={styles.legendRow}>
          <div className={styles.legendItems}>
            {[
              { label: "Bajo", color: "#2e7d32" },
              { label: "Mod-Bajo", color: "#c0ca33" },
              { label: "Moderado", color: "#fbc02d" },
              { label: "Alto", color: "#fb8c00" },
              { label: "Muy Alto", color: "#e53935" },
              { label: "Extremo", color: "#b71c1c" },
            ].map(l => (
              <div key={l.label} className={styles.legendItem}>
                <div className={styles.legendSwatch} style={{ background: l.color }} />
                {l.label}
              </div>
            ))}
          </div>
          <div className={styles.legendNote}>
            Barras semitransparentes = datos históricos · Sólidas = pronóstico
          </div>
        </div>
      </Card>
    </div>
  );
}
