import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { C } from "../../../constants/colors.js";
import { Card } from "../../../components/Card.jsx";
import { SectionLabel } from "../../../components/SectionLabel.jsx";
import { StatBlock } from "../../../components/StatBlock.jsx";

export function FaunaTab({ speciesChartData, speciesApiData, ctStats, isInvasive, isPriority }) {
  return (
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
                const fill = isPriority(entry.nombre) ? C.amber
                           : isInvasive(entry.nombre) ? C.red
                           : C.deepGreen;
                return <Cell key={i} fill={fill} />;
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
            <StatBlock value={ctStats?.total_detections ?? "—"} label="Total detecciones" />
            <StatBlock value={ctStats?.unique_species ?? "—"} label="Especies" />
            <StatBlock value={ctStats?.active_stations ?? "—"} label="Cámaras activas" />
            <StatBlock value={ctStats?.days_sampled ?? "—"} label="Días muestreados" />
          </div>
        </Card>
        <Card>
          <SectionLabel>Especies de interés</SectionLabel>
          <div style={{ marginTop: 8 }}>
            {speciesApiData
              ? (() => {
                  const priority = speciesApiData.filter(d => isPriority(d.nombre));
                  const invasive = speciesApiData.filter(d => isInvasive(d.nombre));
                  return (
                    <>
                      {priority.map(d => (
                        <div key={d.nombre} style={{ padding: "8px 10px", background: `${C.amber}15`, borderRadius: 6, marginBottom: 6, borderLeft: `3px solid ${C.amber}` }}>
                          <div style={{ fontSize: 12, fontWeight: 700, color: C.text }}>{d.nombre}</div>
                          <div style={{ fontSize: 11, color: C.muted }}>
                            {d.detecciones} detección{d.detecciones !== 1 ? "es" : ""}
                            {d.lastSeen && <span> · última: {d.lastSeen.slice(0, 10)}</span>}
                          </div>
                        </div>
                      ))}
                      {invasive.map(d => (
                        <div key={d.nombre} style={{ padding: "8px 10px", background: `${C.red}12`, borderRadius: 6, marginBottom: 6, borderLeft: `3px solid ${C.red}` }}>
                          <div style={{ fontSize: 12, fontWeight: 700, color: C.text }}>{d.nombre} <span style={{ fontSize: 10, color: C.red }}>invasora</span></div>
                          <div style={{ fontSize: 11, color: C.muted }}>
                            {d.detecciones} detección{d.detecciones !== 1 ? "es" : ""}
                            {d.lastSeen && <span> · última: {d.lastSeen.slice(0, 10)}</span>}
                          </div>
                        </div>
                      ))}
                      {priority.length === 0 && invasive.length === 0 && (
                        <div style={{ fontSize: 12, color: C.muted }}>Sin registros de especies prioritarias o invasoras.</div>
                      )}
                    </>
                  );
                })()
              : <div style={{ fontSize: 12, color: C.muted }}>Cargando…</div>
            }
          </div>
        </Card>
      </div>
    </div>
  );
}
