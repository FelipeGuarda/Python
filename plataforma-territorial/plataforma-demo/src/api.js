/**
 * API service layer — connects React frontend to FastAPI backend.
 * Uses relative paths so the Vite proxy (/api → localhost:8000) handles routing.
 */

const API_BASE = "/api";

async function fetchJSON(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`API ${path}: ${res.status}`);
  return res.json();
}

// ── Weather ──

export async function getWeatherCurrent() {
  return fetchJSON("/weather/current");
}

export async function getWeatherHistory(hours = 24) {
  return fetchJSON(`/weather/history?hours=${hours}`);
}

export async function getWeatherForecast(hours = 168) {
  return fetchJSON(`/weather/forecast?hours=${hours}`);
}

// ── Fire Risk ──

export async function getFireRiskCurrent() {
  return fetchJSON("/fire-risk/current");
}

export async function getFireRiskForecast() {
  return fetchJSON("/fire-risk/forecast");
}

// ── Detections ──

export async function getRecentDetections(limit = 20) {
  return fetchJSON(`/detections/recent?limit=${limit}`);
}

export async function getSpeciesSummary() {
  return fetchJSON("/detections/species-summary");
}

export async function getCameraStations() {
  return fetchJSON("/detections/stations");
}

// ── Health ──

export async function getHealth() {
  return fetchJSON("/health");
}

// ── Data transformers (API shape → chart shape) ──

/** /api/weather/history → weatherData array for Recharts */
export function transformWeatherHistory(rows) {
  return rows.map((r) => ({
    hora: r.timestamp?.slice(11, 16) || "",
    temp: r.temperature_air != null ? Math.round(r.temperature_air * 10) / 10 : null,
    humedad: r.relative_humidity != null ? Math.round(r.relative_humidity * 10) / 10 : null,
    viento: r.wind_speed != null ? Math.round(r.wind_speed * 3.6 * 10) / 10 : null,
  }));
}

/** /api/fire-risk/forecast → weeklyRisk array for Recharts */
export function transformRiskForecast(rows) {
  const dayNames = ["Dom", "Lun", "Mar", "Mie", "Jue", "Vie", "Sab"];
  return rows.map((r) => {
    const d = new Date(r.date);
    return {
      dia: dayNames[d.getDay()],
      riesgo: Math.round(r.rule_based?.total || 0),
      temp: r.weather?.temperature_c != null ? Math.round(r.weather.temperature_c) : null,
      humedad: r.weather?.relative_humidity_pct != null ? Math.round(r.weather.relative_humidity_pct) : null,
    };
  });
}

/** /api/detections/species-summary → speciesData array for Recharts */
export function transformSpeciesSummary(rows) {
  return rows.slice(0, 12).map((r) => ({
    nombre: r.species || "Desconocido",
    detecciones: r.total_detections || 0,
    individuos: r.total_individuals || 0,
    lastSeen: r.last_seen,
  }));
}
