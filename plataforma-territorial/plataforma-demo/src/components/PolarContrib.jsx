import { C } from "../constants/colors.js";

// ── POLAR CONTRIBUTION CHART ──
export function PolarContrib({ components, size = 180, color = C.amber }) {
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
      <polygon points={pts.map(p => p.join(",")).join(" ")} fill={`${color}35`} stroke={color} strokeWidth="2" strokeLinejoin="round" />
      {pts.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={3} fill={color} />)}
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
