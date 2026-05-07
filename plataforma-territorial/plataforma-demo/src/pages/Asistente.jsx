import { useState, useEffect, useRef } from "react";
import { Card } from "../components/Card.jsx";
import { SectionLabel } from "../components/SectionLabel.jsx";
import { chatMessages as initialMessages } from "../constants/demo_chat.js";
import styles from "./Asistente.module.css";

const CHAT_CARD_STYLE = { display: "flex", flexDirection: "column", padding: 0, overflow: "hidden" };

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
    <div className={styles.container}>
      <Card style={CHAT_CARD_STYLE}>
        {/* Header */}
        <div className={styles.chatHeader}>
          <div className={styles.chatHeaderTitle}>Asistente Territorial</div>
          <div className={styles.chatHeaderSubtitle}>Consulta datos y metodologias en lenguaje natural</div>
        </div>

        {/* Messages */}
        <div className={styles.messages}>
          {messages.map((m, i) => (
            <div key={i} className={`${styles.msgRow} ${m.role === "user" ? styles.msgRowUser : styles.msgRowAssistant}`}>
              <div className={`${styles.bubble} ${m.role === "user" ? styles.bubbleUser : styles.bubbleAssistant}`}>
                {m.text}
                {m.role === "assistant" && (
                  <div className={styles.bubbleSource}>
                    Fuente: base de datos local + formula documentada
                  </div>
                )}
              </div>
            </div>
          ))}
          {typing && (
            <div className={`${styles.msgRow} ${styles.msgRowAssistant}`}>
              <div className={styles.typing}>
                Consultando datos...
              </div>
            </div>
          )}
          <div ref={endRef} />
        </div>

        {/* Input */}
        <div className={styles.inputBar}>
          <input value={input} onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleSend()}
            placeholder="Pregunta sobre datos o metodologias..."
            className={styles.inputField}
          />
          <button onClick={handleSend} className={styles.sendBtn}>Enviar</button>
        </div>
      </Card>

      {/* Sidebar */}
      <div className={styles.sidebar}>
        <Card>
          <SectionLabel>Preguntas sugeridas</SectionLabel>
          <div className={styles.suggestions}>
            {suggestions.map((s, i) => (
              <button key={i} onClick={() => { setInput(s); }} className={styles.suggBtn}>
                {s}
              </button>
            ))}
          </div>
        </Card>
        <Card>
          <SectionLabel>Principio de transparencia</SectionLabel>
          <div className={styles.noticeBody}>
            Cada respuesta que involucre un valor calculado muestra su formula, los datos de entrada y el modelo que lo produjo. Sin cajas negras.
          </div>
        </Card>
        <Card>
          <SectionLabel>Capacidades</SectionLabel>
          <div className={styles.caps}>
            {["Consultar riesgo de incendio", "Buscar detecciones de especies", "Explicar metodologias", "Analizar tendencias climaticas"].map((c, i) => (
              <div key={i} className={styles.capRow}>
                <div className={styles.capDot} />
                {c}
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}
