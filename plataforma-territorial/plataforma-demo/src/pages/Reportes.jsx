import { useState } from "react";
import { C } from "../constants/colors.js";
import { sampleDraft } from "../constants/demo_report.js";
import { Card } from "../components/Card.jsx";
import { SectionLabel } from "../components/SectionLabel.jsx";
import styles from "./Reportes.module.css";

const DRAFT_CARD_STYLE = { display: "flex", flexDirection: "column" };

export function Reportes() {
  const [generating, setGenerating] = useState(false);
  const [draft, setDraft] = useState("");
  const [period, setPeriod] = useState("Febrero 2026");

  const handleGenerate = () => {
    setGenerating(true);
    setDraft("");
    const text = sampleDraft(period);
    let i = 0;
    const interval = setInterval(() => {
      setDraft(text.slice(0, i));
      i += 3;
      if (i > text.length) {
        clearInterval(interval);
        setGenerating(false);
        setDraft(text);
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
