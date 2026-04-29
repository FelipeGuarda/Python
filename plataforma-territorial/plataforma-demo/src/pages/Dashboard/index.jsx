import { useState, useEffect, useRef, useMemo } from "react";
import {
  getWeatherCurrent,
  getFireRiskCurrent, getFireRiskForecast, getFireRiskHistory, getSpeciesSummary,
  getStationSummary, getDielActivity, getCampaignStats, getSpeciesList, getSpeciesOverlap,
  getGeography, getSpecies,
  transformRiskForecast, transformSpeciesSummary, transformDielActivity,
} from "../../api.js";
import { C } from "../../constants/colors.js";
import { useAPI } from "../../hooks/useAPI.js";
import { RiesgoTab } from "./tabs/RiesgoTab.jsx";
import { MeteoTab } from "./tabs/MeteoTab.jsx";
import { CamarasTab } from "./tabs/CamarasTab.jsx";
import { FaunaTab } from "./tabs/FaunaTab.jsx";

const DAY_NAMES = ["Dom", "Lun", "Mar", "Mie", "Jue", "Vie", "Sab"];

function getMondayOf(dateStr) {
  const d = new Date(dateStr + "T12:00:00");
  const day = d.getDay();
  d.setDate(d.getDate() + (day === 0 ? -6 : 1 - day));
  return d.toISOString().split("T")[0];
}

// ── PAGE: Dashboard ──
export function Dashboard() {
  const [tab, setTab] = useState("riesgo");
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const { data: riskCurrent } = useAPI(getFireRiskCurrent, null, []);
  const { data: riskForecast } = useAPI(getFireRiskForecast, transformRiskForecast, []);
  const { data: riskHistory } = useAPI(getFireRiskHistory, transformRiskForecast, []);
  const { data: speciesApiData } = useAPI(getSpeciesSummary, transformSpeciesSummary, [], tab === "fauna");
  const { data: weatherCurrent } = useAPI(getWeatherCurrent, null, []);
  const { data: ctStations } = useAPI(getStationSummary, null, [], tab === "camaras");
  const { data: dielData } = useAPI(getDielActivity, transformDielActivity, [], tab === "fauna");
  const { data: ctStats } = useAPI(getCampaignStats, null, [], tab === "camaras");
  const { data: geo } = useAPI(getGeography, null, []);
  const { data: speciesCatalog } = useAPI(getSpecies, null, []);
  const totalStations = geo?.camera_trap_count ?? null;

  // Build lowercase name lookups (both latin and spanish) for invasive/priority
  // classification — fed by /api/config/species. Replaces three regex literals
  // that drifted across this file.
  const speciesIndex = useMemo(() => {
    if (!speciesCatalog) return null;
    const inv = new Set();
    const pri = new Set();
    for (const s of speciesCatalog) {
      if (s.is_invasive) { inv.add(s.latin.toLowerCase()); inv.add(s.spanish.toLowerCase()); }
      if (s.is_priority) { pri.add(s.latin.toLowerCase()); pri.add(s.spanish.toLowerCase()); }
    }
    return { inv, pri };
  }, [speciesCatalog]);
  const isInvasive = (name) => !!(name && speciesIndex?.inv.has(String(name).toLowerCase()));
  const isPriority = (name) => !!(name && speciesIndex?.pri.has(String(name).toLowerCase()));

  // ── Species comparator state ──
  const [speciesList, setSpeciesList] = useState([]);
  const [sp1Sel, setSp1Sel] = useState("");
  const [sp2Sel, setSp2Sel] = useState("");
  const [overlapData, setOverlapData] = useState(null);
  const [overlapLoading, setOverlapLoading] = useState(false);
  const [camBoundary, setCamBoundary] = useState(null);
  const speciesLoaded = useRef(false);

  const riskTotal = riskCurrent?.rule_based?.total ? Math.round(riskCurrent.rule_based.total) : 0;
  const mlVal = riskCurrent?.ml_probability != null ? Math.round(riskCurrent.ml_probability * 100) : null;
  const wx = riskCurrent?.weather || {};
  const speciesChartData = speciesApiData;

  // ── Load species list + boundary the first time the camaras tab is opened ──
  useEffect(() => {
    if (tab !== "camaras" || speciesLoaded.current) return;
    speciesLoaded.current = true;
    fetch("/data/boundary.geojson").then(r => r.json()).then(setCamBoundary);
    getSpeciesList().then(list => {
      setSpeciesList(list);
      if (list.length >= 2) {
        const defSp1 = (list.find(s => s.scientific_name === "Puma concolor") ?? list[0]).scientific_name;
        const defSp2 = (list.find(s => s.scientific_name === "Lycalopex culpaeus") ?? list[1]).scientific_name;
        setSp1Sel(defSp1);
        setSp2Sel(defSp2);
        setOverlapLoading(true);
        getSpeciesOverlap(defSp1, defSp2)
          .then(d => { setOverlapData(d); setOverlapLoading(false); })
          .catch(() => setOverlapLoading(false));
      }
    });
  }, [tab]);

  function applyOverlap() {
    if (!sp1Sel || !sp2Sel || sp1Sel === sp2Sel || overlapLoading) return;
    setOverlapLoading(true);
    getSpeciesOverlap(sp1Sel, sp2Sel)
      .then(d => { setOverlapData(d); setOverlapLoading(false); })
      .catch(() => setOverlapLoading(false));
  }

  // ── Bar chart: fixed 3-week window (prev week + current week + next week) ──
  const todayStr = useMemo(() => new Date().toLocaleDateString("sv-SE", { timeZone: "America/Santiago" }), []);

  const allRiskData = useMemo(() => {
    const hist = riskHistory || [];
    const fore = riskForecast || [];
    const map = new Map();
    hist.forEach(d => map.set(d.date, d));
    fore.forEach(d => map.set(d.date, d));
    return Array.from(map.values()).sort((a, b) => a.date.localeCompare(b.date));
  }, [riskHistory, riskForecast]);

  const windowData = useMemo(() => {
    // Always: Monday of previous week → Sunday of next week (21 days fixed)
    const windowStart = new Date(getMondayOf(todayStr) + "T12:00:00");
    windowStart.setDate(windowStart.getDate() - 7);
    const lookup = new Map(allRiskData.map(d => [d.date, d]));
    return Array.from({ length: 21 }, (_, i) => {
      const d = new Date(windowStart);
      d.setDate(d.getDate() + i);
      const dateStr = d.toISOString().split("T")[0];
      const found = lookup.get(dateStr);
      const dayNum = d.getDate();
      const dayName = DAY_NAMES[d.getDay()];
      return found
        ? { ...found, diaLabel: `${dayName} ${dayNum}` }
        : { date: dateStr, riesgo: null, color: null, isHistorical: null, diaLabel: `${dayName} ${dayNum}` };
    });
  }, [allRiskData, todayStr]);

  const todayDiaLabel = useMemo(() => {
    const d = new Date(todayStr + "T12:00:00");
    return `${DAY_NAMES[d.getDay()]} ${d.getDate()}`;
  }, [todayStr]);

  const tabs = [
    { id: "riesgo", label: "Riesgo de Incendio" },
    { id: "meteo", label: "Meteorologia" },
    { id: "camaras", label: "Camaras Trampa" },
    { id: "fauna", label: "Fauna" },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "calc(100vh - 56px)" }}>
      {/* Tabs */}
      <div style={{ display: "flex", gap: 0, borderBottom: `2px solid ${C.mint}`, padding: "0 16px", flexShrink: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            background: tab === t.id ? C.white : "transparent", border: "none",
            borderBottom: tab === t.id ? `2px solid ${C.deepGreen}` : "2px solid transparent",
            padding: "10px 20px", cursor: "pointer", fontSize: 13,
            fontFamily: "'Trebuchet MS', sans-serif", color: tab === t.id ? C.text : C.muted,
            fontWeight: tab === t.id ? 700 : 400, borderRadius: "6px 6px 0 0",
            marginBottom: -2, transition: "all 0.2s",
          }}>{t.label}</button>
        ))}
      </div>

      {/* Tab content */}
      <div style={{ flex: 1, overflowY: "auto", padding: 16 }}>
      {tab === "riesgo" && mounted && (
        <RiesgoTab
          riskCurrent={riskCurrent}
          riskTotal={riskTotal}
          mlVal={mlVal}
          wx={wx}
          windowData={windowData}
          todayDiaLabel={todayDiaLabel}
        />
      )}

      {tab === "meteo" && mounted && <MeteoTab />}

      {tab === "camaras" && mounted && (
        <CamarasTab
          dielData={dielData}
          ctStats={ctStats}
          speciesList={speciesList}
          sp1Sel={sp1Sel}
          setSp1Sel={setSp1Sel}
          sp2Sel={sp2Sel}
          setSp2Sel={setSp2Sel}
          applyOverlap={applyOverlap}
          overlapLoading={overlapLoading}
          overlapData={overlapData}
          totalStations={totalStations}
          camBoundary={camBoundary}
          geo={geo}
          isInvasive={isInvasive}
          isPriority={isPriority}
        />
      )}

      {tab === "fauna" && mounted && (
        <FaunaTab
          speciesChartData={speciesChartData}
          speciesApiData={speciesApiData}
          ctStats={ctStats}
          isInvasive={isInvasive}
          isPriority={isPriority}
        />
      )}
      </div>
    </div>
  );
}
