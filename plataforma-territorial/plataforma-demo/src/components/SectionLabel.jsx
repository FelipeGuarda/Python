import styles from "./SectionLabel.module.css";

export function SectionLabel({ children, className = "", style }) {
  return (
    <div className={`${styles.label} ${className}`.trim()} style={style}>
      {children}
    </div>
  );
}
