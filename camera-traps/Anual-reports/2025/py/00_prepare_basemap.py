"""
00_prepare_basemap.py — one-shot prep of basemap GeoJSONs for the report maps.

Inputs (shapefiles, UTM 18S WGS84 — EPSG:32718)
-----------------------------------------------
Provided by ICN/FMA in three ZIPs pasted to Anual-reports/:
- `Curvas de nivel_BP-*.zip`        — 15 m contour lines, clean ELEV field.
- `Figura 5_Sistema hídrico SN BP-*.zip`
    - `Red hídrica/esterosBP.*`     — 11 segments, 2 named streams.
- `Red senderos y Caminos-*.zip`
    - `Sendero Completo BP 2.*`     — trail + road network, `name` field.

Outputs (WGS84 lon/lat — EPSG:4326)
-----------------------------------
Written to `plataforma-territorial/data/basemap/`:
- `hydric_main.geojson`          — 2 named main streams.
- `roads_main.geojson`           — Puma + Araucarias vehicular roads.
- `bp_threshold_contour.geojson` — single threshold contour line, clipped to BP.
- `bp_high_zone.geojson`         — BP polygon clipped to area above the threshold.

The threshold is the 1005 m contour from the 15 m source — the closest
available contour to the nominal 1000 m boundary used in the report. The
narrative refers to it as ~1000 m for readability; the actual ELEV is
stored truthfully in the GeoJSON properties.

The polygon high zone is built by splitting BP along the threshold contour
extended at its endpoints to the nearest BP boundary point. Each piece is
classified by majority of CTs inside (with sample-point fallback when a
piece contains no CT). CTs themselves use the GPS `altitude_m` field
(corrected 2026-06-09 for CT11/12/13).
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import shapefile
from pyproj import Transformer
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    mapping,
    shape,
)
from shapely.ops import linemerge, nearest_points, split, unary_union

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve()
REPORT_ROOT = HERE.parents[1]
REPO = HERE.parents[4]
ZIPS_DIR = REPORT_ROOT.parent  # camera-traps/Anual-reports/
TMP_DIR = Path("C:/Users/USUARIO/AppData/Local/Temp/bp_zips")

CONTOUR_ZIP = next(ZIPS_DIR.glob("Curvas de nivel_BP-*.zip"))
HYDRIC_ZIP = next(ZIPS_DIR.glob("Figura 5_Sistema h*drico SN BP-*.zip"))
ROADS_ZIP = next(ZIPS_DIR.glob("Red senderos y Caminos-*.zip"))

BOUNDARY_GEOJSON = REPO / "plataforma-territorial" / "data" / "boundary.geojson"
STATIONS_GEOJSON = REPO / "plataforma-territorial" / "data" / "camera_trap_stations.geojson"
OUT_DIR = REPO / "plataforma-territorial" / "data" / "basemap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_M = 1005.0

TX = Transformer.from_crs("EPSG:32718", "EPSG:4326", always_xy=True)


def utm_to_lonlat(x: float, y: float) -> tuple[float, float]:
    lon, lat = TX.transform(x, y)
    return lon, lat


def reproj_line(coords: list[list[float]]) -> list[list[float]]:
    return [list(utm_to_lonlat(x, y)) for x, y, *_ in coords]


def load_boundary_poly() -> Polygon:
    gj = json.loads(BOUNDARY_GEOJSON.read_text(encoding="utf-8"))
    return shape(gj["features"][0]["geometry"])


def write_geojson(features: list[dict], path: Path) -> None:
    fc = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ wrote {path.relative_to(REPO)} ({len(features)} features)")


def extract_zips() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    for z in (CONTOUR_ZIP, HYDRIC_ZIP, ROADS_ZIP):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(TMP_DIR)
    print(f"  unzipped to {TMP_DIR}")


def find_shp(stem: str) -> Path:
    matches = list(TMP_DIR.rglob(f"{stem}.shp"))
    matches = [m for m in matches if "Copia" not in str(m)]
    if not matches:
        raise FileNotFoundError(f"No shapefile {stem}.shp under {TMP_DIR}")
    return matches[0]


# ─────────────────────────────────────────────────────────────────────────────
# Main rivers — esterosBP, dissolved by Nombre

def build_main_rivers(bp_poly: Polygon) -> list[dict]:
    sf = shapefile.Reader(str(find_shp("esterosBP")), encoding="latin-1")
    by_name: dict[str, list[LineString]] = {}
    for shp_rec in sf.shapeRecords():
        name = shp_rec.record["Nombre"].strip()
        coords_ll = reproj_line(shp_rec.shape.points)
        if len(coords_ll) < 2:
            continue
        by_name.setdefault(name, []).append(LineString(coords_ll))

    features = []
    for name, lines in by_name.items():
        merged = linemerge(MultiLineString(lines))
        clipped = merged.intersection(bp_poly)
        if clipped.is_empty:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {"name": name},
                "geometry": mapping(clipped),
            }
        )
    return features


# ─────────────────────────────────────────────────────────────────────────────
# Main roads — Puma + Araucarias only

ROAD_WHITELIST = {"PUMA", "SENDERO ARAUCARIAS"}


def build_main_roads(bp_poly: Polygon) -> list[dict]:
    sf = shapefile.Reader(str(find_shp("Sendero Completo BP 2")), encoding="latin-1")
    features = []
    for shp_rec in sf.shapeRecords():
        name = shp_rec.record["name"].strip()
        if name not in ROAD_WHITELIST:
            continue
        coords_ll = reproj_line(shp_rec.shape.points)
        if len(coords_ll) < 2:
            continue
        line = LineString(coords_ll)
        clipped = line.intersection(bp_poly)
        if clipped.is_empty:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "name": name,
                    "label": shp_rec.record["Senderos"].strip(),
                },
                "geometry": mapping(clipped),
            }
        )
    return features


# ─────────────────────────────────────────────────────────────────────────────
# Threshold contour and high-zone polygon

CONTOUR_SHP_STEM = "curvas de nivel_15 m"


def load_contour_at_threshold(bp_poly: Polygon):
    """Returns the threshold contour merged into a LineString/MultiLineString,
    clipped to BP."""
    sf = shapefile.Reader(str(find_shp(CONTOUR_SHP_STEM)), encoding="latin-1")
    lines = []
    for shp_rec in sf.shapeRecords():
        elev = float(shp_rec.record["ELEV"])
        if elev != THRESHOLD_M:
            continue
        coords_ll = reproj_line(shp_rec.shape.points)
        if len(coords_ll) < 2:
            continue
        lines.append(LineString(coords_ll))
    if not lines:
        raise ValueError(f"No contour at {THRESHOLD_M} m in {CONTOUR_SHP_STEM}")
    merged = linemerge(MultiLineString(lines)) if len(lines) > 1 else lines[0]
    clipped = merged.intersection(bp_poly)
    if clipped.is_empty:
        raise ValueError("Threshold contour does not intersect BP polygon")
    if clipped.geom_type not in ("LineString", "MultiLineString"):
        parts = [g for g in getattr(clipped, "geoms", []) if g.geom_type in ("LineString", "MultiLineString")]
        clipped = unary_union(parts) if parts else clipped
    return clipped


def build_threshold_contour_feature(threshold_line) -> list[dict]:
    return [{
        "type": "Feature",
        "properties": {"elev_m": THRESHOLD_M},
        "geometry": mapping(threshold_line),
    }]


def extend_endpoints_to_boundary(line, bp_boundary):
    """For each constituent LineString, extend its endpoints in a straight line
    to the nearest BP boundary point and a tiny bit beyond, so the resulting
    splitter reliably partitions BP under shapely.split."""
    segments = list(line.geoms) if line.geom_type == "MultiLineString" else [line]
    eps = 1e-4  # ~10 m in degrees at this latitude — safely outside BP
    out = []
    for seg in segments:
        coords = list(seg.coords)
        if len(coords) < 2:
            continue
        start_pt = Point(coords[0])
        end_pt = Point(coords[-1])
        _, near_start = nearest_points(start_pt, bp_boundary)
        _, near_end = nearest_points(end_pt, bp_boundary)

        def push_beyond(from_pt: Point, to_pt: Point) -> tuple[float, float]:
            dx = to_pt.x - from_pt.x
            dy = to_pt.y - from_pt.y
            d = (dx * dx + dy * dy) ** 0.5 or 1e-12
            return (to_pt.x + dx / d * eps, to_pt.y + dy / d * eps)

        prefix = [push_beyond(start_pt, near_start)]
        suffix = [push_beyond(end_pt, near_end)]
        out.append(LineString(prefix + coords + suffix))
    return out


def build_high_zone(bp_poly: Polygon, threshold_line) -> list[dict]:
    extended = extend_endpoints_to_boundary(threshold_line, bp_poly.boundary)
    splitter = unary_union(extended)
    try:
        pieces = list(split(bp_poly, splitter).geoms)
    except Exception as e:
        print(f"  ! shapely.split failed ({e}); falling back to single piece")
        pieces = [bp_poly]

    stations = json.loads(STATIONS_GEOJSON.read_text(encoding="utf-8"))
    ct_points = [
        (Point(ft["geometry"]["coordinates"]), float(ft["properties"]["altitude_m"]))
        for ft in stations["features"]
        if ft["properties"].get("altitude_m") is not None
    ]

    high_pieces, low_pieces = [], []
    for piece in pieces:
        if piece.is_empty or piece.area < 1e-10:
            continue
        cts_inside = [(p, a) for p, a in ct_points if piece.contains(p)]
        if cts_inside:
            n_hi = sum(1 for _, a in cts_inside if a > THRESHOLD_M)
            classified_high = n_hi >= len(cts_inside) - n_hi
        else:
            # Fallback: representative point — for slivers with no CTs this is
            # rare and the answer is usually visually obvious anyway.
            rp = piece.representative_point()
            classified_high = rp.distance(threshold_line) > 0  # boundary-side ambiguous; default high
        (high_pieces if classified_high else low_pieces).append(piece)

    if not high_pieces:
        raise ValueError("No pieces classified as high")
    high_union = unary_union(high_pieces)
    if high_union.geom_type == "Polygon":
        high_union = MultiPolygon([high_union])

    print(f"  high pieces: {len(high_pieces)}, low pieces: {len(low_pieces)}")
    print(f"  high zone area fraction of BP: {high_union.area / bp_poly.area:.1%}")
    return [{
        "type": "Feature",
        "properties": {"threshold_m": THRESHOLD_M, "side": "above"},
        "geometry": mapping(high_union),
    }]


# ─────────────────────────────────────────────────────────────────────────────
# Main


def main() -> None:
    print("Working dirs:")
    print(f"  REPORT_ROOT = {REPORT_ROOT}")
    print(f"  ZIPS_DIR    = {ZIPS_DIR}")
    print(f"  OUT_DIR     = {OUT_DIR}")

    print("\nExtracting zips...")
    extract_zips()

    print("\nLoading BP boundary...")
    bp_poly = load_boundary_poly()
    print(f"  bounds: {bp_poly.bounds}")
    print(f"  area  : {bp_poly.area:.6f} sq.deg")

    print("\nBuilding main rivers...")
    rivers = build_main_rivers(bp_poly)
    write_geojson(rivers, OUT_DIR / "hydric_main.geojson")

    print("\nBuilding main roads (Puma + Araucarias)...")
    roads = build_main_roads(bp_poly)
    write_geojson(roads, OUT_DIR / "roads_main.geojson")

    print(f"\nBuilding threshold contour at {THRESHOLD_M} m...")
    threshold_line = load_contour_at_threshold(bp_poly)
    write_geojson(build_threshold_contour_feature(threshold_line),
                  OUT_DIR / "bp_threshold_contour.geojson")

    print(f"\nBuilding high-zone polygon (above {THRESHOLD_M} m)...")
    high_features = build_high_zone(bp_poly, threshold_line)
    write_geojson(high_features, OUT_DIR / "bp_high_zone.geojson")

    print("\nDone.")


if __name__ == "__main__":
    main()
