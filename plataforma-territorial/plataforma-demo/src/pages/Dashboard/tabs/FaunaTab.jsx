import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { C } from "../../../constants/colors.js";
import {
  CHART_TICK_MD,
  CHART_TICK_CAT,
  CHART_AXIS_LINE,
  CHART_GRID,
  CHART_TOOLTIP,
} from "../../../styles/chart.js";
import { Card } from "../../../components/Card.jsx";
import { SectionLabel } from "../../../components/SectionLabel.jsx";
import { StatBlock } from "../../../components/StatBlock.jsx";
import styles from "./FaunaTab.module.css";

export function FaunaTab({ speciesChartData, speciesApiData, ctStats, isInvasive, isPriority }) {
  return (
    <div className={styles.container}>
      <Card>
        <SectionLabel>Detecciones por especie (ultimo mes)</SectionLabel>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={speciesChartData} layout="vertical" margin={{ left: 100 }}>
            <CartesianGrid {...CHART_GRID} horizontal={false} />
            <XAxis type="number" tick={CHART_TICK_MD} axisLine={CHART_AXIS_LINE} />
            <YAxis type="category" dataKey="nombre" tick={CHART_TICK_CAT} axisLine={CHART_AXIS_LINE} width={100} />
            <Tooltip contentStyle={CHART_TOOLTIP} />
            <Bar dataKey="detecciones" radius={[0, 4, 4, 0]} name="Detecciones">
              {speciesChartData.map((entry, i) => {
                const fill = isPriority(entry.nombre) ? C.amber
                           : isInvasive(entry.nombre) ? C.red
                           : C.deepGreen;
                return <Cell key={i} fill={fill} />;
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className={styles.legendRow}>
          <div className={styles.legendItem}>
            <div className={styles.legendSwatch} style={{ background: C.deepGreen }} /> Nativa
          </div>
          <div className={styles.legendItem}>
            <div className={styles.legendSwatch} style={{ background: C.amber }} /> Prioritaria
          </div>
          <div className={styles.legendItem}>
            <div className={styles.legendSwatch} style={{ background: C.red }} /> Invasora
          </div>
        </div>
      </Card>
      <div className={styles.sidebarCol}>
        <Card>
          <SectionLabel>Resumen del mes</SectionLabel>
          <div className={styles.statsGrid}>
            <StatBlock value={ctStats?.total_detections ?? "—"} label="Total detecciones" />
            <StatBlock value={ctStats?.unique_species ?? "—"} label="Especies" />
            <StatBlock value={ctStats?.active_stations ?? "—"} label="Cámaras activas" />
            <StatBlock value={ctStats?.days_sampled ?? "—"} label="Días muestreados" />
          </div>
        </Card>
        <Card>
          <SectionLabel>Especies de interés</SectionLabel>
          <div className={styles.speciesList}>
            {speciesApiData
              ? (() => {
                  const priority = speciesApiData.filter(d => isPriority(d.nombre));
                  const invasive = speciesApiData.filter(d => isInvasive(d.nombre));
                  return (
                    <>
                      {priority.map(d => (
                        <div key={d.nombre} className={styles.speciesBox}
                          style={{ background: `${C.amber}15`, borderLeftColor: C.amber }}>
                          <div className={styles.speciesName}>{d.nombre}</div>
                          <div className={styles.speciesMeta}>
                            {d.detecciones} detección{d.detecciones !== 1 ? "es" : ""}
                            {d.lastSeen && <span> · última: {d.lastSeen.slice(0, 10)}</span>}
                          </div>
                        </div>
                      ))}
                      {invasive.map(d => (
                        <div key={d.nombre} className={styles.speciesBox}
                          style={{ background: `${C.red}12`, borderLeftColor: C.red }}>
                          <div className={styles.speciesName}>{d.nombre} <span className={styles.invasiveTag}>invasora</span></div>
                          <div className={styles.speciesMeta}>
                            {d.detecciones} detección{d.detecciones !== 1 ? "es" : ""}
                            {d.lastSeen && <span> · última: {d.lastSeen.slice(0, 10)}</span>}
                          </div>
                        </div>
                      ))}
                      {priority.length === 0 && invasive.length === 0 && (
                        <div className={styles.emptyMsg}>Sin registros de especies prioritarias o invasoras.</div>
                      )}
                    </>
                  );
                })()
              : <div className={styles.emptyMsg}>Cargando…</div>
            }
          </div>
        </Card>
      </div>
    </div>
  );
}
