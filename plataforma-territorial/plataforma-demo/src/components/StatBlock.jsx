import { C } from "../constants/colors.js";
import styles from "./StatBlock.module.css";

export function StatBlock({ value, label, unit = "", color = C.text }) {
  return (
    <div className={styles.box}>
      <div className={styles.value} style={{ color }}>
        {value}<span className={styles.unit}>{unit}</span>
      </div>
      <div className={styles.label}>{label}</div>
    </div>
  );
}
