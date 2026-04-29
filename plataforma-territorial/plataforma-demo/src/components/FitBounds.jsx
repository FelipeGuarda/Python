import { useEffect } from "react";
import { useMap } from "react-leaflet";
import L from "leaflet";

// Helper: fit map to boundary when GeoJSON loads
export function FitBounds({ geojson }) {
  const map = useMap();
  useEffect(() => {
    if (geojson) {
      const layer = L.geoJSON(geojson);
      map.fitBounds(layer.getBounds(), { padding: [30, 30] });
    }
  }, [geojson, map]);
  return null;
}
