import { C } from "../constants/colors.js";

// ── WIND COMPASS ──
export function WindCompass({ direction, speed }) {
  const cx = 55, cy = 55, r = 44;
  const toRad = d => (d - 90) * Math.PI / 180;
  const dirs = ["N", "E", "S", "O"];
  const dirAngles = [0, 90, 180, 270];
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      <svg width="110" height="110" viewBox="0 0 110 110">
        <defs>
          <marker id="windArrow" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto">
            <path d="M 0 0 L 5 2.5 L 0 5 Z" fill={C.deepGreen} />
          </marker>
        </defs>
        <circle cx={cx} cy={cy} r={r} fill={C.paleMint} stroke={C.mint} strokeWidth="1.5" />
        {Array.from({ length: 16 }, (_, i) => {
          const rad = toRad(i * 22.5);
          const rInner = i % 4 === 0 ? r - 9 : r - 5;
          return <line key={i} x1={cx + r * Math.cos(rad)} y1={cy + r * Math.sin(rad)}
            x2={cx + rInner * Math.cos(rad)} y2={cy + rInner * Math.sin(rad)}
            stroke={C.lightMuted} strokeWidth={i % 4 === 0 ? 1.5 : 1} />;
        })}
        {dirs.map((label, i) => {
          const rad = toRad(dirAngles[i]);
          const dist = r - 16;
          return <text key={label} x={cx + dist * Math.cos(rad)} y={cy + dist * Math.sin(rad)}
            textAnchor="middle" dominantBaseline="middle" fontSize="9"
            fontWeight={label === "N" ? "700" : "400"} fill={label === "N" ? C.red : C.text}>{label}</text>;
        })}
        {direction != null && (() => {
          const rad = toRad(direction);
          return <line x1={cx - 16 * Math.cos(rad)} y1={cy - 16 * Math.sin(rad)}
            x2={cx + 28 * Math.cos(rad)} y2={cy + 28 * Math.sin(rad)}
            stroke={C.deepGreen} strokeWidth="2.5" strokeLinecap="round" markerEnd="url(#windArrow)" />;
        })()}
        <circle cx={cx} cy={cy} r={3} fill={C.deepGreen} />
      </svg>
      <div style={{ fontSize: 12, color: C.text, fontWeight: 600, marginTop: 2 }}>
        {speed != null ? `${Number(speed).toFixed(1)} km/h` : "—"}
      </div>
      {direction != null && (
        <div style={{ fontSize: 10, color: C.muted }}>{direction}°</div>
      )}
    </div>
  );
}
