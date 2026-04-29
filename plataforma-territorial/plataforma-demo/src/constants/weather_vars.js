import { C } from "./colors.js";

// ── WEATHER CONSTANTS ──
export const WEATHER_VARS = [
  { id: "albedo_Avg",       label: "Albedo",                 unit: "%",     color: C.warmSand },
  { id: "relative_humidity",label: "Humedad relativa",       unit: "%",     color: C.medGreen },
  { id: "precipitation",    label: "Precipitación",          unit: "mm",    color: C.deepGreen, type: "bar" },
  { id: "BP_mbar_Avg",      label: "Presión atmosférica",    unit: "hPa",   color: C.lightGreen },
  { id: "DT_Avg",           label: "Profundidad de nieve",   unit: "cm",    color: "#7EC8E3" },
  { id: "PtoRocio_Avg",     label: "Punto de rocío",         unit: "°C",    color: C.lightMuted },
  { id: "solar_radiation",  label: "Radiación solar",        unit: "W/m²",  color: C.amber },
  { id: "T107_10cm_Avg",    label: "Suelo a 10cm",           unit: "°C",    color: "#7BAE91" },
  { id: "T107_50cm_Avg",    label: "Suelo a 50cm",           unit: "°C",    color: "#A8CFA8" },
  { id: "temperature_air",  label: "Temperatura del aire",   unit: "°C",    color: C.red },
  { id: "wind_speed",       label: "Velocidad del viento",   unit: "km/h",  color: "#5BC0EB" },
];

export const WIND_SPEED_COLORS = ["#BDE8FF", "#5BC0EB", "#1A6B9A", "#2ECC71", "#E67E22", "#E74C3C"];
export const RESOLUTIONS = [
  { id: "15min", label: "15 min" },
  { id: "D",     label: "Diaria" },
  { id: "ME",    label: "Mensual" },
  { id: "Q",     label: "Estacional" },
];
export const VAR_FRIENDLY = {
  ...Object.fromEntries(WEATHER_VARS.map(v => [v.id, v.label])),
  wind_speed: "Velocidad del viento",
};
