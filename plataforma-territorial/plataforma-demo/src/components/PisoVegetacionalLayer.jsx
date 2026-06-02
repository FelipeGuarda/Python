import { useEffect, useState } from "react";
import { GeoJSON } from "react-leaflet";

// Color palette for the BIOTOPO field. Three semantic groups so the eye reads
// the type before the density: greens = Bosque (mature forest), ochres =
// Renoval (regrowth), blues/violets = open vegetation (Matorral / Pradera /
// Estepa). Within Bosque and Renoval, darker = denser. Nothing else references
// these — tweak here.
const BIOTOPO_COLORS = {
  "Bosque Denso":        "#1B5E20",
  "Bosque Semidenso":    "#388E3C",
  "Bosque Abierto":      "#81C784",
  "Bosque Achaparrado":  "#689F38",
  "Renoval Denso":       "#8B5A2B",
  "Renoval Semidenso":   "#C9852E",
  "Renoval Abierto":     "#E5C57A",
  "Matorral Mesófito":   "#4FC3F7",
  "Pradera":             "#9575CD",
  "Estepa Altoandina":   "#5E35B1",
};
const FALLBACK_COLOR = "#9E9E9E";

const styleFeature = (feature) => ({
  stroke: false,
  fillColor: BIOTOPO_COLORS[feature.properties?.BIOTOPO] ?? FALLBACK_COLOR,
  fillOpacity: 0.55,
});

const onEachFeature = (feature, layer) => {
  const p = feature.properties || {};
  const biotopo = p.BIOTOPO ?? "—";
  const distrito = p.DISTRITO ?? "—";
  const supe = typeof p.Supe === "number" ? `${p.Supe.toFixed(2)} ha` : "—";
  layer.bindPopup(
    `<div style="font-family:'Trebuchet MS',sans-serif;min-width:160px">
      <div style="font-weight:700;font-size:13px;margin-bottom:4px">${biotopo}</div>
      <div style="font-size:11px;color:#666">Distrito: ${distrito}</div>
      <div style="font-size:11px;color:#666">Superficie: ${supe}</div>
    </div>`
  );
};

export function PisoVegetacionalLayer() {
  const [data, setData] = useState(null);

  useEffect(() => {
    let cancelled = false;
    fetch("/data/piso_vegetacional.geojson")
      .then((r) => r.json())
      .then((json) => { if (!cancelled) setData(json); })
      .catch((e) => console.error("piso_vegetacional load failed", e));
    return () => { cancelled = true; };
  }, []);

  if (!data) return null;
  return <GeoJSON data={data} style={styleFeature} onEachFeature={onEachFeature} />;
}
