import { Component } from "react";
import styles from "./ErrorBoundary.module.css";

// ── Error Boundary ──
export class ErrorBoundary extends Component {
  constructor(props) { super(props); this.state = { error: null }; }
  static getDerivedStateFromError(error) { return { error }; }
  componentDidCatch(error, info) { console.error(error, info.componentStack); }
  render() {
    if (this.state.error) return (
      <div className={styles.errorBox}>
        <div className={styles.errorTitle}>Error al cargar la vista</div>
        <div className={styles.errorBody}>{this.state.error.message}</div>
        <button onClick={() => this.setState({ error: null })} className={styles.retryBtn}>
          Reintentar
        </button>
      </div>
    );
    return this.props.children;
  }
}
