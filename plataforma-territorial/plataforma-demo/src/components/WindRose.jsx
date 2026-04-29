import { C } from "../constants/colors.js";
import { WIND_SPEED_COLORS } from "../constants/weather_vars.js";

// ── WIND ROSE SVG ──
export function WindRose({ data, size = 230 }) {
  if (!data || data.length === 0) return null;
  const s = size / 230;
  const cx = 115 * s, cy = 115 * s, maxR = 90 * s;
  const maxPct = Math.max(...data.map(d => d.total_pct), 0.1);

  function arcPath(r1, r2, a1Deg, a2Deg) {
    const toRad = d => (d * Math.PI) / 180;
    const a1 = toRad(a1Deg), a2 = toRad(a2Deg);
    const x2 = cx + r2 * Math.cos(a1), y2 = cy + r2 * Math.sin(a1);
    const x3 = cx + r2 * Math.cos(a2), y3 = cy + r2 * Math.sin(a2);
    const lg = (a2 - a1 > Math.PI) ? 1 : 0;
    if (r1 < 1) {
      return `M ${cx} ${cy} L ${x2.toFixed(1)} ${y2.toFixed(1)} A ${r2} ${r2} 0 ${lg} 1 ${x3.toFixed(1)} ${y3.toFixed(1)} Z`;
    }
    const x1 = cx + r1 * Math.cos(a1), y1 = cy + r1 * Math.sin(a1);
    const x4 = cx + r1 * Math.cos(a2), y4 = cy + r1 * Math.sin(a2);
    return `M ${x1.toFixed(1)} ${y1.toFixed(1)} L ${x2.toFixed(1)} ${y2.toFixed(1)} A ${r2} ${r2} 0 ${lg} 1 ${x3.toFixed(1)} ${y3.toFixed(1)} L ${x4.toFixed(1)} ${y4.toFixed(1)} A ${r1} ${r1} 0 ${lg} 0 ${x1.toFixed(1)} ${y1.toFixed(1)} Z`;
  }

  const sectors = [];
  data.forEach((d, i) => {
    const mid = i * 22.5 - 90; // rotate so 0° = North = up
    const a1 = mid - 11.25, a2 = mid + 11.25;
    let r0 = 0;
    d.bins.forEach((bin, j) => {
      const r = (bin.pct / maxPct) * maxR;
      if (r > 0.3) {
        sectors.push(
          <path key={`${i}-${j}`} d={arcPath(r0, r0 + r, a1, a2)}
            fill={WIND_SPEED_COLORS[j]} stroke="white" strokeWidth={0.4} opacity={0.9} />
        );
      }
      r0 += r;
    });
  });

  const cardinal = [
    { label: "N", dx: 0,              dy: -(maxR + 12 * s) },
    { label: "E", dx: maxR + 14 * s,  dy: 4 * s },
    { label: "S", dx: 0,              dy: maxR + 16 * s },
    { label: "O", dx: -(maxR + 14 * s), dy: 4 * s },
  ];

  return (
    <svg width={size} height={size} style={{ display: "block", margin: "0 auto" }}>
      {[0.25, 0.5, 0.75, 1].map(f => (
        <circle key={f} cx={cx} cy={cy} r={maxR * f} fill="none" stroke={C.paleMint} strokeWidth={0.8} strokeDasharray="3 3" />
      ))}
      <line x1={cx} y1={cy - maxR} x2={cx} y2={cy + maxR} stroke={C.paleMint} strokeWidth={0.5} />
      <line x1={cx - maxR} y1={cy} x2={cx + maxR} y2={cy} stroke={C.paleMint} strokeWidth={0.5} />
      {sectors}
      {cardinal.map(c => (
        <text key={c.label} x={cx + c.dx} y={cy + c.dy} textAnchor="middle" fontSize={Math.round(10 * s)} fontWeight={700} fill={C.muted}>{c.label}</text>
      ))}
    </svg>
  );
}
