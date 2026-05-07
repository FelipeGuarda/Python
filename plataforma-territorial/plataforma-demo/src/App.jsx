import { useState } from "react";
import { ErrorBoundary } from "./components/ErrorBoundary.jsx";
import { NavBar } from "./components/NavBar.jsx";
import { Observatorio } from "./pages/Observatorio.jsx";
import { Dashboard } from "./pages/Dashboard/index.jsx";
import { Asistente } from "./pages/Asistente.jsx";
import { Reportes } from "./pages/Reportes.jsx";
import styles from "./App.module.css";

// TODO(future-cleanup): The decomposition of App.jsx (April 2026) deferred two
// items from the planned `constants/` folder, both pending future architectural
// refinement. They are kept out of scope here to preserve the "pure structural
// move" guarantee of this refactor.
//   1. `constants/chart_defaults.js` — repeated Recharts axis/grid/tick style
//      objects could be extracted to named constants once we are willing to
//      rewrite call sites uniformly across MeteoTab, RiesgoTab, CamarasTab,
//      and FaunaTab.
//   2. `constants/demo_chat.js` — should hold the missing `chatMessages` seed
//      array (referenced by Asistente.jsx but undefined; runtime ReferenceError)
//      and a `sampleDraft(period)` function-form for Reportes.jsx.

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
