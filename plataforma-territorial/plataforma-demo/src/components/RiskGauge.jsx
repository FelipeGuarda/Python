import { C } from "../constants/colors.js";
import styles from "./RiskGauge.module.css";

export function RiskGauge({ value, color: colorProp = null, compact = false }) {
  const getColor = (v) =>
    v >= 90 ? "#b71c1c" : v >= 80 ? "#e53935" : v >= 60 ? "#fb8c00" :
    v >= 40 ? "#fbc02d" : v >= 20 ? "#c0ca33" : "#2e7d32";
  const getLabel = (v) =>
    v >= 90 ? "EXTREMO" : v >= 80 ? "MUY ALTO" : v >= 60 ? "ALTO" :
    v >= 40 ? "MODERADO" : v >= 20 ? "MOD-BAJO" : "BAJO";
  const color = colorProp || getColor(value);
  const angle = (value / 100) * 180;
  const w = compact ? 120 : 180;
  const h = compact ? 67 : 100;
  return (
    <div className={`${styles.box} ${compact ? styles.boxCompact : styles.boxRegular}`}>
      <svg width={w} height={h} viewBox="0 0 180 100">
        <path d="M 10 90 A 80 80 0 0 1 170 90" fill="none" stroke={C.paleMint} strokeWidth="12" strokeLinecap="round" />
        <path d="M 10 90 A 80 80 0 0 1 170 90" fill="none" stroke={color} strokeWidth="12" strokeLinecap="round"
          strokeDasharray={`${angle / 180 * 251.3} 251.3`} />
        <text x="90" y="75" textAnchor="middle" fontSize={compact ? "24" : "28"} fontWeight="700" fontFamily="Georgia" fill={color}>{value}</text>
        <text x="90" y="92" textAnchor="middle" fontSize="10" fill={C.muted}>/100</text>
      </svg>
      <div className={`${styles.label} ${compact ? styles.labelCompact : styles.labelRegular}`} style={{ color }}>
        {getLabel(value)}
      </div>
    </div>
  );
}
