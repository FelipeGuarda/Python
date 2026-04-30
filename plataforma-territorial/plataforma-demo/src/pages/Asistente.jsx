import { useState, useEffect, useRef } from "react";
import { C } from "../constants/colors.js";
import { Card } from "../components/Card.jsx";
import { SectionLabel } from "../components/SectionLabel.jsx";
import { chatMessages as initialMessages } from "../constants/demo_chat.js";

export function Asistente() {
  const [messages, setMessages] = useState(initialMessages);
  const [input, setInput] = useState("");
  const [typing, setTyping] = useState(false);
  const endRef = useRef(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  const handleSend = () => {
    if (!input.trim()) return;
    const userMsg = input.trim();
    setMessages(prev => [...prev, { role: "user", text: userMsg }]);
    setInput("");
    setTyping(true);
    setTimeout(() => {
      setMessages(prev => [...prev, {
        role: "assistant",
        text: "Esta es una demostracion del Asistente. En la version final, cada respuesta consultara la base de datos en tiempo real y citara la formula o metodologia utilizada para generar la informacion. Transparencia metodologica total.",
      }]);
      setTyping(false);
    }, 1500);
  };

  const suggestions = [
    "Como se calcula el indice de riesgo?",
    "Cuantas detecciones de Puma hay este ano?",
    "Cual es la tendencia de temperatura?",
    "Que camaras tienen mas actividad?",
  ];

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 280px", gap: 16, height: "calc(100vh - 56px)", padding: 16 }}>
      <Card style={{ display: "flex", flexDirection: "column", padding: 0, overflow: "hidden" }}>
        {/* Header */}
        <div style={{ padding: "14px 20px", borderBottom: `1px solid ${C.paleMint}`, background: C.white }}>
          <div style={{ fontSize: 15, fontWeight: 700, color: C.text, fontFamily: "'Georgia', serif" }}>Asistente Territorial</div>
          <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>Consulta datos y metodologias en lenguaje natural</div>
        </div>

        {/* Messages */}
        <div style={{ flex: 1, overflowY: "auto", padding: 20, display: "flex", flexDirection: "column", gap: 12, background: C.bg }}>
          {messages.map((m, i) => (
            <div key={i} style={{ display: "flex", justifyContent: m.role === "user" ? "flex-end" : "flex-start" }}>
              <div style={{
                maxWidth: "80%", padding: "10px 14px", borderRadius: 12, fontSize: 13, lineHeight: 1.6,
                background: m.role === "user" ? C.deepGreen : C.white,
                color: m.role === "user" ? C.white : C.text,
                borderBottomRightRadius: m.role === "user" ? 2 : 12,
                borderBottomLeftRadius: m.role === "user" ? 12 : 2,
                boxShadow: m.role === "assistant" ? "0 1px 3px rgba(0,0,0,0.06)" : "none",
              }}>
                {m.text}
                {m.role === "assistant" && (
                  <div style={{ fontSize: 10, color: C.lightMuted, marginTop: 6, fontStyle: "italic" }}>
                    Fuente: base de datos local + formula documentada
                  </div>
                )}
              </div>
            </div>
          ))}
          {typing && (
            <div style={{ display: "flex", justifyContent: "flex-start" }}>
              <div style={{ background: C.white, padding: "10px 14px", borderRadius: 12, fontSize: 13, color: C.muted, boxShadow: "0 1px 3px rgba(0,0,0,0.06)" }}>
                Consultando datos...
              </div>
            </div>
          )}
          <div ref={endRef} />
        </div>

        {/* Input */}
        <div style={{ padding: "12px 16px", borderTop: `1px solid ${C.paleMint}`, background: C.white, display: "flex", gap: 8 }}>
          <input value={input} onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleSend()}
            placeholder="Pregunta sobre datos o metodologias..."
            style={{ flex: 1, border: `1px solid ${C.mint}`, borderRadius: 8, padding: "10px 14px", fontSize: 13, outline: "none", fontFamily: "inherit" }}
          />
          <button onClick={handleSend} style={{
            background: C.deepGreen, color: C.white, border: "none", borderRadius: 8,
            padding: "10px 20px", cursor: "pointer", fontSize: 13, fontWeight: 600,
          }}>Enviar</button>
        </div>
      </Card>

      {/* Sidebar */}
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <Card>
          <SectionLabel>Preguntas sugeridas</SectionLabel>
          <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 8 }}>
            {suggestions.map((s, i) => (
              <button key={i} onClick={() => { setInput(s); }}
                style={{ background: C.paleMint, border: "none", borderRadius: 6, padding: "8px 10px",
                  fontSize: 11, color: C.text, textAlign: "left", cursor: "pointer", lineHeight: 1.4,
                  transition: "background 0.2s" }}
                onMouseEnter={e => e.target.style.background = C.mint}
                onMouseLeave={e => e.target.style.background = C.paleMint}>
                {s}
              </button>
            ))}
          </div>
        </Card>
        <Card>
          <SectionLabel>Principio de transparencia</SectionLabel>
          <div style={{ fontSize: 12, color: C.muted, lineHeight: 1.6, marginTop: 6 }}>
            Cada respuesta que involucre un valor calculado muestra su formula, los datos de entrada y el modelo que lo produjo. Sin cajas negras.
          </div>
        </Card>
        <Card>
          <SectionLabel>Capacidades</SectionLabel>
          <div style={{ marginTop: 6, display: "flex", flexDirection: "column", gap: 6 }}>
            {["Consultar riesgo de incendio", "Buscar detecciones de especies", "Explicar metodologias", "Analizar tendencias climaticas"].map((c, i) => (
              <div key={i} style={{ fontSize: 11, color: C.text, display: "flex", alignItems: "center", gap: 6 }}>
                <div style={{ width: 5, height: 5, borderRadius: "50%", background: C.medGreen, flexShrink: 0 }} />
                {c}
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}
