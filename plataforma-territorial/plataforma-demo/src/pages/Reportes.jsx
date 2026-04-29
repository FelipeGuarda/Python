import { useState } from "react";
import { C } from "../constants/colors.js";
import { Card } from "../components/Card.jsx";
import { SectionLabel } from "../components/SectionLabel.jsx";

// TODO(future-cleanup): `sampleDraft` is a template literal that interpolates
// the `period` state, so it cannot be hoisted to a static constants module
// without being converted to a function (e.g. `sampleDraft(period) => ...`).
// Kept inside Reportes for the App.jsx decomposition (pure structural move).
// When refining the frontend architecture, convert to a function and move to
// `constants/demo_chat.js` (or similar).

export function Reportes() {
  const [generating, setGenerating] = useState(false);
  const [draft, setDraft] = useState("");
  const [period, setPeriod] = useState("Febrero 2026");

  const sampleDraft = `Resumen mensual — Bosque Pehuen, ${period}

Durante ${period.toLowerCase()}, las condiciones en Bosque Pehuen se caracterizaron por temperaturas superiores al promedio historico y precipitaciones por debajo de lo esperado para la temporada.

Riesgo de incendio: El indice promedio fue de 48/100 (moderado), con un pico de 72/100 el dia 15, coincidiendo con una ola de calor que elevo la temperatura maxima a 31°C con humedad relativa de 22%. Se recomienda mantener la vigilancia activa durante episodios similares.

Monitoreo de fauna: Las 5 camaras trampa registraron un total de 978 detecciones de 8 especies. Destaca un aumento del 23% en registros de jabali respecto al mes anterior, concentrado en las estaciones Rio Turbio y Araucaria Norte. Se registro 1 evento de Puma en Laguna Sur, consistente con su rango de movimiento estacional.

Especies prioritarias: Guina fue detectada en 23 ocasiones en Sendero Pehuen, lo que representa una frecuencia estable respecto a meses anteriores. No se detectaron nuevas especies invasoras.

Nota: Este borrador fue generado automaticamente a partir de los datos del repositorio central. Requiere revision y edicion del equipo antes de su publicacion.`;

  const handleGenerate = () => {
    setGenerating(true);
    setDraft("");
    let i = 0;
    const interval = setInterval(() => {
      setDraft(sampleDraft.slice(0, i));
      i += 3;
      if (i > sampleDraft.length) {
        clearInterval(interval);
        setGenerating(false);
        setDraft(sampleDraft);
      }
    }, 8);
  };

  return (
    <div style={{ padding: 16, height: "calc(100vh - 56px)", overflowY: "auto" }}>
      <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: 16 }}>
        {/* Config panel */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <Card>
            <SectionLabel>Configuracion</SectionLabel>
            <div style={{ fontFamily: "'Georgia', serif", fontSize: 15, fontWeight: 700, color: C.text, marginBottom: 12 }}>Generar Reporte</div>
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11, color: C.muted, marginBottom: 4 }}>Periodo</div>
              <select value={period} onChange={e => setPeriod(e.target.value)}
                style={{ width: "100%", padding: "8px 10px", border: `1px solid ${C.mint}`, borderRadius: 6, fontSize: 13, background: C.white, color: C.text }}>
                <option>Febrero 2026</option>
                <option>Enero 2026</option>
                <option>Diciembre 2025</option>
              </select>
            </div>
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11, color: C.muted, marginBottom: 4 }}>Secciones a incluir</div>
              {["Riesgo de incendio", "Meteorologia", "Fauna y camaras trampa", "Especies prioritarias"].map((s, i) => (
                <label key={i} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: C.text, marginBottom: 4, cursor: "pointer" }}>
                  <input type="checkbox" defaultChecked /> {s}
                </label>
              ))}
            </div>
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11, color: C.muted, marginBottom: 4 }}>Audiencia</div>
              <select style={{ width: "100%", padding: "8px 10px", border: `1px solid ${C.mint}`, borderRadius: 6, fontSize: 13, background: C.white, color: C.text }}>
                <option>Equipo FMA (interna)</option>
                <option>Socios de conservacion</option>
                <option>Publico general</option>
              </select>
            </div>
            <button onClick={handleGenerate} disabled={generating}
              style={{
                width: "100%", background: generating ? C.muted : C.deepGreen, color: C.white,
                border: "none", borderRadius: 8, padding: "12px 0", cursor: generating ? "wait" : "pointer",
                fontSize: 13, fontWeight: 700, transition: "background 0.2s",
              }}>
              {generating ? "Generando..." : "Generar borrador"}
            </button>
          </Card>
          <Card>
            <SectionLabel>Importante</SectionLabel>
            <div style={{ fontSize: 12, color: C.muted, lineHeight: 1.6, marginTop: 6 }}>
              La IA redacta un borrador a partir de los datos del periodo seleccionado. El equipo humano revisa, edita y decide si publicar. Este modulo es un acelerador de escritura, no un reemplazo.
            </div>
          </Card>
          <Card>
            <SectionLabel>Exportar</SectionLabel>
            <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
              <button style={{ flex: 1, padding: "8px 0", border: `1px solid ${C.mint}`, borderRadius: 6, background: C.white, color: C.text, fontSize: 11, cursor: "pointer" }}>
                Word .docx
              </button>
              <button style={{ flex: 1, padding: "8px 0", border: `1px solid ${C.mint}`, borderRadius: 6, background: C.white, color: C.text, fontSize: 11, cursor: "pointer" }}>
                Copiar texto
              </button>
            </div>
          </Card>
        </div>

        {/* Draft area */}
        <Card style={{ display: "flex", flexDirection: "column" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <div>
              <SectionLabel>Borrador</SectionLabel>
              <div style={{ fontFamily: "'Georgia', serif", fontSize: 15, fontWeight: 700, color: C.text }}>
                Reporte mensual — {period}
              </div>
            </div>
            {draft && <div style={{ fontSize: 10, color: C.lightMuted, fontFamily: "monospace" }}>BORRADOR — REQUIERE REVISION</div>}
          </div>
          {draft ? (
            <textarea value={draft} onChange={e => setDraft(e.target.value)}
              style={{
                flex: 1, minHeight: 400, border: `1px solid ${C.mint}`, borderRadius: 8, padding: 16,
                fontSize: 13, lineHeight: 1.8, color: C.text, fontFamily: "'Georgia', serif",
                resize: "none", outline: "none", background: C.bg,
              }}
            />
          ) : (
            <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", minHeight: 400, color: C.lightMuted, fontSize: 14 }}>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: 40, marginBottom: 12, opacity: 0.3 }}>&#9998;</div>
                <div>Selecciona el periodo y haz clic en "Generar borrador"</div>
                <div style={{ fontSize: 12, marginTop: 6 }}>El sistema consultara los datos del mes y redactara un resumen</div>
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
