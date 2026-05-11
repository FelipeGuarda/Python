import { C } from "../constants/colors.js";
import styles from "./RiskGauge.module.css";

export function RiskGauge({ value, color: colorProp = null, label: labelProp = null, compact = false }) {
  // Color + label come from the backend (fire_risk.py risk_components()), which is
  // the single source of truth for the risk scale. Fall back to muted gray + "—"
  // when no data is available (e.g. backend unreachable on first render).
  const color = colorProp ?? C.muted;
  const label = labelProp ?? "—";
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
        {label}
      </div>
    </div>
  );
}
