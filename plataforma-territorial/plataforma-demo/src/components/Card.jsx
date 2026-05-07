import styles from "./Card.module.css";

export function Card({ children, className = "", style }) {
  return (
    <div className={`${styles.card} ${className}`.trim()} style={style}>
      {children}
    </div>
  );
}
