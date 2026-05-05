import { C } from "../constants/colors.js";

// ── POLAR CONTRIBUTION CHART ──
export function PolarContrib({ components, size = 180, color = C.amber, weather = null }) {
  const axes = [
    {
      key: "temp_score", label: "Temp", max: 25, angle: -90,
      actual: weather?.temperature_c != null ? `${weather.temperature_c.toFixed(1)}°C` : null,
    },
    {
      key: "wind_score", label: "Viento", max: 15, angle: 0,
      actual: weather?.wind_speed_kmh != null ? `${Math.round(weather.wind_speed_kmh)} km/h` : null,
    },
    {
      key: "days_score", label: "Días s/lluvia", max: 35, angle: 90,
      actual: weather?.days_without_rain != null ? `${weather.days_without_rain} días` : null,
    },
    {
      key: "rh_score", label: "Humedad", max: 25, angle: 180,
      actual: weather?.relative_humidity_pct != null ? `${Math.round(weather.relative_humidity_pct)}%` : null,
    },
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
            {a.actual && (
              <text x={lx} y={ly - 14} textAnchor="middle" dominantBaseline="middle"
                fontSize="11" fontWeight="700" fill={color}>{a.actual}</text>
            )}
            <text x={lx} y={a.actual ? ly - 2 : ly - 5} textAnchor="middle" dominantBaseline="middle"
              fontSize="9" fontWeight="700" fill={C.text}>{a.label}</text>
            <text x={lx} y={a.actual ? ly + 9 : ly + 6} textAnchor="middle" dominantBaseline="middle"
              fontSize="8" fill={C.muted}>{val}/{a.max}</text>
          </g>
        );
      })}
    </svg>
  );
}
