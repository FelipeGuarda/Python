import { C } from "../constants/colors.js";

export function NavBar({ page, setPage }) {
  const pages = [
    { id: "observatorio", label: "Observatorio", icon: "M" },
    { id: "dashboard", label: "Dashboard", icon: "D" },
    { id: "asistente", label: "Asistente", icon: "A" },
    { id: "reportes", label: "Reportes", icon: "R" },
  ];
  return (
    <div style={{ background: C.deepGreen, padding: "0 24px", display: "flex", alignItems: "center", height: 56, gap: 0 }}>
      <div style={{ fontFamily: "'Georgia', serif", color: C.white, fontSize: 15, fontWeight: 700, marginRight: 40, letterSpacing: -0.3 }}>
        Plataforma Territorial <span style={{ color: C.lightGreen, fontWeight: 400, fontSize: 12 }}>FMA</span>
      </div>
      <div style={{ display: "flex", gap: 0, flex: 1 }}>
        {pages.map(p => (
          <button key={p.id} onClick={() => setPage(p.id)} style={{
            background: page === p.id ? "rgba(255,255,255,0.12)" : "transparent",
            border: "none", color: page === p.id ? C.white : C.lightGreen,
            padding: "16px 20px", cursor: "pointer", fontSize: 13,
            fontFamily: "'Trebuchet MS', sans-serif", fontWeight: page === p.id ? 700 : 400,
            borderBottom: page === p.id ? `2px solid ${C.white}` : "2px solid transparent",
            transition: "all 0.2s",
          }}>{p.label}</button>
        ))}
      </div>
      <div style={{ color: C.lightMuted, fontSize: 11, fontFamily: "monospace", letterSpacing: 1 }}>
        BOSQUE PEHUEN
      </div>
    </div>
  );
}
