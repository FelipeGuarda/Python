import styles from "./NavBar.module.css";

export function NavBar({ page, setPage }) {
  const pages = [
    { id: "observatorio", label: "Observatorio", icon: "M" },
    { id: "dashboard", label: "Dashboard", icon: "D" },
    { id: "asistente", label: "Asistente", icon: "A" },
    { id: "reportes", label: "Reportes", icon: "R" },
  ];
  return (
    <div className={styles.bar}>
      <div className={styles.brand}>
        Plataforma Territorial <span className={styles.brandSuffix}>FMA</span>
      </div>
      <div className={styles.tabs}>
        {pages.map(p => (
          <button key={p.id} onClick={() => setPage(p.id)}
            className={`${styles.tab} ${page === p.id ? styles.tabActive : ""}`.trim()}>
            {p.label}
          </button>
        ))}
      </div>
      <div className={styles.location}>
        BOSQUE PEHUEN
      </div>
    </div>
  );
}
