import { C } from "../constants/colors.js";

// Default props/styles for Recharts primitives. Frozen at module scope so
// the same object identity is reused on every render (avoids per-render
// allocations and lets React skip prop-diff work).

export const CHART_TICK_SM   = Object.freeze({ fontSize: 9,  fill: C.muted });
export const CHART_TICK_MD   = Object.freeze({ fontSize: 10, fill: C.muted });
export const CHART_TICK_CAT  = Object.freeze({ fontSize: 11, fill: C.text  });

export const CHART_AXIS_LINE = Object.freeze({ stroke: C.mint });

export const CHART_GRID      = Object.freeze({ strokeDasharray: "3 3", stroke: C.paleMint });

export const CHART_TOOLTIP          = Object.freeze({ borderRadius: 6, fontSize: 11 });
export const CHART_TOOLTIP_BORDERED = Object.freeze({ borderRadius: 6, fontSize: 11, border: `1px solid ${C.mint}` });
