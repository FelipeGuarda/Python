import { Component } from "react";
import { C } from "../constants/colors.js";

// ── Error Boundary ──
export class ErrorBoundary extends Component {
  constructor(props) { super(props); this.state = { error: null }; }
  static getDerivedStateFromError(error) { return { error }; }
  componentDidCatch(error, info) { console.error(error, info.componentStack); }
  render() {
    if (this.state.error) return (
      <div style={{ padding: 40, color: C.red, fontFamily: "monospace", fontSize: 13 }}>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>Error al cargar la vista</div>
        <div style={{ color: C.muted }}>{this.state.error.message}</div>
        <button onClick={() => this.setState({ error: null })}
          style={{ marginTop: 16, padding: "8px 16px", background: C.deepGreen, color: C.white, border: "none", borderRadius: 6, cursor: "pointer" }}>
          Reintentar
        </button>
      </div>
    );
    return this.props.children;
  }
}
