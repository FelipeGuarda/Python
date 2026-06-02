"""Convert the Bosque Pehuén vegetational floor shapefile to GeoJSON.

Source: data/piso_vegetacional_source/veg_foto_BP.{shp,shx,dbf,prj,cpg}
- CRS in .prj is WGS_1984_UTM_Zone_18S → reprojected to EPSG:4326 (Leaflet)
- .cpg correctly declares UTF-8 (the .dbf carries "ó" as the 2-byte UTF-8 0xC3 0xB3)

Outputs the GeoJSON to the three locations used by the rest of the project:
  data/piso_vegetacional.geojson                          (canonical source copy)
  plataforma-demo/public/data/piso_vegetacional.geojson   (dev-served)
  plataforma-demo/dist/data/piso_vegetacional.geojson     (built output)
"""
from __future__ import annotations

import json
from pathlib import Path

import shapefile
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "piso_vegetacional_source" / "veg_foto_BP.shp"
OUTPUTS = [
    ROOT / "data" / "piso_vegetacional.geojson",
    ROOT / "plataforma-demo" / "public" / "data" / "piso_vegetacional.geojson",
    ROOT / "plataforma-demo" / "dist" / "data" / "piso_vegetacional.geojson",
]

# UTM 18S (WGS84) → WGS84 lat/lon. always_xy keeps (lon, lat) order in output.
to_wgs84 = Transformer.from_crs("EPSG:32718", "EPSG:4326", always_xy=True).transform


def reproject_ring(ring):
    return [list(to_wgs84(x, y)) for x, y in ring]


def signed_area(ring):
    """Shoelace signed area in source coords (y-up). >0 = CCW, <0 = CW."""
    s = 0.0
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def shape_to_geometry(shape):
    """pyshp polygon → GeoJSON Polygon / MultiPolygon, reprojected to EPSG:4326.

    ESRI convention: outer rings are clockwise (signed_area < 0), holes are CCW.
    We group each outer ring with its trailing holes.
    """
    parts = list(shape.parts) + [len(shape.points)]
    rings = [shape.points[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)]

    polygons = []
    current = None
    for ring in rings:
        reprojected = reproject_ring(ring)
        is_outer = signed_area(ring) < 0
        if is_outer:
            if current is not None:
                polygons.append(current)
            current = [reprojected]
        else:
            if current is None:
                # Stray hole with no preceding outer — keep it to avoid data loss
                current = [reprojected]
            else:
                current.append(reprojected)
    if current is not None:
        polygons.append(current)

    if len(polygons) == 1:
        return {"type": "Polygon", "coordinates": polygons[0]}
    return {"type": "MultiPolygon", "coordinates": polygons}


def main():
    if not SRC.exists():
        raise SystemExit(f"Missing source shapefile: {SRC}")

    sf = shapefile.Reader(str(SRC), encoding="utf-8")
    field_names = [f[0] for f in sf.fields[1:]]

    features = []
    for shape_rec in sf.shapeRecords():
        props = dict(zip(field_names, shape_rec.record))
        # ID column is all zeros in this dataset; drop it to keep payload tidy
        props.pop("ID", None)
        features.append({
            "type": "Feature",
            "properties": props,
            "geometry": shape_to_geometry(shape_rec.shape),
        })

    fc = {"type": "FeatureCollection", "features": features}
    payload = json.dumps(fc, ensure_ascii=False)

    for out in OUTPUTS:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)}  ({len(features)} features, {len(payload):,} bytes)")


if __name__ == "__main__":
    main()
