import { useState } from "react";
import { ErrorBoundary } from "./components/ErrorBoundary.jsx";
import { NavBar } from "./components/NavBar.jsx";
import { Observatorio } from "./pages/Observatorio.jsx";
import { Dashboard } from "./pages/Dashboard/index.jsx";
import { Asistente } from "./pages/Asistente.jsx";
import { Reportes } from "./pages/Reportes.jsx";
import styles from "./App.module.css";

// Page state lives in a single useState — no URL routing by design. The
// platform is a single-user internal tool with 4 pages and no current need
// for deep-linkable URLs or back-button page navigation. When that need
// surfaces (e.g. sharing a link to a specific Dashboard tab), introduce
// react-router-dom: wrap App in <BrowserRouter>, replace this switch with
// <Routes>, and the NavBar's setPage calls with <Link>/useNavigate.

// ── MAIN APP ──
export default function App() {
  const [page, setPage] = useState("observatorio");

  return (
    <div className={styles.shell}>
      <NavBar page={page} setPage={setPage} />
      <ErrorBoundary key={page}>
        {page === "observatorio" && <Observatorio />}
        {page === "dashboard" && <Dashboard />}
        {page === "asistente" && <Asistente />}
        {page === "reportes" && <Reportes />}
      </ErrorBoundary>
    </div>
  );
}
