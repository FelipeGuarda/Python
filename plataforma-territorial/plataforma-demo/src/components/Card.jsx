import { C } from "../constants/colors.js";

export function Card({ children, style = {} }) {
  return (
    <div style={{ background: C.white, borderRadius: 8, padding: 20, boxShadow: "0 1px 4px rgba(0,77,60,0.06)", ...style }}>
      {children}
    </div>
  );
}
