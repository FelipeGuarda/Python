import { C } from "../constants/colors.js";

export function SectionLabel({ children }) {
  return (
    <div style={{ fontSize: 10, fontFamily: "monospace", color: C.lightMuted, letterSpacing: 3, textTransform: "uppercase", marginBottom: 6 }}>
      {children}
    </div>
  );
}
