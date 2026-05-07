import { useState } from "react";
import { C } from "../constants/colors.js";
import { Card } from "../components/Card.jsx";
import { SectionLabel } from "../components/SectionLabel.jsx";
import styles from "./Reportes.module.css";

// TODO(future-cleanup): `sampleDraft` is a template literal that interpolates
// the `period` state, so it cannot be hoisted to a static constants module
// without being converted to a function (e.g. `sampleDraft(period) => ...`).
// Kept inside Reportes for the App.jsx decomposition (pure structural move).
// When refining the frontend architecture, convert to a function and move to
// `constants/demo_chat.js` (or similar).

const DRAFT_CARD_STYLE = { display: "flex", flexDirection: "column" };

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
    <div className={styles.container}>
      <div className={styles.layout}>
        {/* Config panel */}
        <div className={styles.column}>
          <Card>
            <SectionLabel>Configuracion</SectionLabel>
            <div className={styles.title}>Generar Reporte</div>
            <div className={styles.formGroup}>
              <div className={styles.formLabel}>Periodo</div>
              <select value={period} onChange={e => setPeriod(e.target.value)}
                className={styles.select}>
                <option>Febrero 2026</option>
                <option>Enero 2026</option>
                <option>Diciembre 2025</option>
              </select>
            </div>
            <div className={styles.formGroup}>
              <div className={styles.formLabel}>Secciones a incluir</div>
              {["Riesgo de incendio", "Meteorologia", "Fauna y camaras trampa", "Especies prioritarias"].map((s, i) => (
                <label key={i} className={styles.checkLabel}>
                  <input type="checkbox" defaultChecked /> {s}
                </label>
              ))}
            </div>
            <div className={styles.formGroup}>
              <div className={styles.formLabel}>Audiencia</div>
              <select className={styles.select}>
                <option>Equipo FMA (interna)</option>
                <option>Socios de conservacion</option>
                <option>Publico general</option>
              </select>
            </div>
            <button onClick={handleGenerate} disabled={generating}
              className={styles.generateBtn}
              style={{
                background: generating ? C.muted : C.deepGreen,
                cursor: generating ? "wait" : "pointer",
              }}>
              {generating ? "Generando..." : "Generar borrador"}
            </button>
          </Card>
          <Card>
            <SectionLabel>Importante</SectionLabel>
            <div className={styles.noticeBody}>
              La IA redacta un borrador a partir de los datos del periodo seleccionado. El equipo humano revisa, edita y decide si publicar. Este modulo es un acelerador de escritura, no un reemplazo.
            </div>
          </Card>
          <Card>
            <SectionLabel>Exportar</SectionLabel>
            <div className={styles.exportRow}>
              <button className={styles.exportBtn}>
                Word .docx
              </button>
              <button className={styles.exportBtn}>
                Copiar texto
              </button>
            </div>
          </Card>
        </div>

        {/* Draft area */}
        <Card style={DRAFT_CARD_STYLE}>
          <div className={styles.draftHeader}>
            <div>
              <SectionLabel>Borrador</SectionLabel>
              <div className={styles.draftTitle}>
                Reporte mensual — {period}
              </div>
            </div>
            {draft && <div className={styles.draftBadge}>BORRADOR — REQUIERE REVISION</div>}
          </div>
          {draft ? (
            <textarea value={draft} onChange={e => setDraft(e.target.value)}
              className={styles.draftTextarea}
            />
          ) : (
            <div className={styles.placeholder}>
              <div className={styles.placeholderInner}>
                <div className={styles.placeholderIcon}>&#9998;</div>
                <div>Selecciona el periodo y haz clic en "Generar borrador"</div>
                <div className={styles.placeholderHint}>El sistema consultara los datos del mes y redactara un resumen</div>
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
