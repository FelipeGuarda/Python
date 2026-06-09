"""
00_prepare_basemap.py — one-shot prep of basemap GeoJSONs for the report maps.

Inputs (shapefiles, UTM 18S WGS84 — EPSG:32718)
-----------------------------------------------
Provided by ICN/IDE in the two ZIPs pasted to Anual-reports/:
- `Figura 5_Sistema hídrico SN BP.zip`
    - `Red hídrica/esterosBP.*`  — 11 segments, 2 named streams (main rivers).
    - `Curvas de nivel/cn_5m.*`  — 5 m contour lines (ELEV field).
- `Red senderos y Caminos.zip`
    - `Sendero Completo BP 2.*`  — trail + road network, `Tipo` field.

Outputs (WGS84 lon/lat — EPSG:4326)
-----------------------------------
Written to `plataforma-territorial/data/basemap/`:
- `hydric_main.geojson`        — 2 named main streams.
- `roads_main.geojson`         — vehicular roads only (Tipo == "Vehicular").
- `bp_threshold_contour.geojson` — single 1000 m contour line, clipped to BP.
- `bp_high_zone.geojson`       — BP polygon clipped to area above 1000 m.

NOTE on the high-zone polygon. The polygon is derived from the **cn_5m 1000 m
contour line itself** (clipped to BP, with endpoints extended to the polygon
boundary so the contour fully partitions BP). Each resulting piece is
classified high/low by majority vote over a grid of sample points, where each
sample is matched to its nearest cn_5m contour ELEV — but only among contours
that lie physically near BP (≤ 0.005°, ~500 m), to avoid contamination from
the absurdly low (5–25 m) contours that exist somewhere west of BP and would
otherwise pollute nearest-neighbor queries.

The polygon edge and the visual 1000 m contour are now derived from the same
source — no double-line/misalignment artifact.

Stations.yaml altitudes were used as cross-check (26/26 match after the
2026-06-09 correction of CT11/12/13). Earlier "errors" attributed to cn_5m
were in fact errors in the installation log: CT11/12/13 were entered as
819–820 m when they're actually 1102–1209 m per the field KMZ.

The script unzips the ZIPs to a temp dir, runs, then leaves them there.

Run once on the Linux/Windows machine that holds the canonical project; the
GeoJSON outputs are committed and consumed by `02_figures_tables.py` (which
keeps its lightweight numpy/json loader pattern — no shapely at render time).
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
from shapely.ops import linemerge, split, unary_union

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve()
REPORT_ROOT = HERE.parents[1]
REPO = HERE.parents[4]
ZIPS_DIR = REPORT_ROOT.parent  # camera-traps/Anual-reports/
TMP_DIR = Path("C:/Users/USUARIO/AppData/Local/Temp/bp_zips")

HYDRIC_ZIP = next(ZIPS_DIR.glob("Figura 5_Sistema h*drico SN BP-*.zip"))
ROADS_ZIP = next(ZIPS_DIR.glob("Red senderos y Caminos-*.zip"))

OUT_DIR = REPO / "plataforma-territorial" / "data" / "basemap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BOUNDARY_GEOJSON = REPO / "plataforma-territorial" / "data" / "boundary.geojson"
STATIONS_GEOJSON = REPO / "plataforma-territorial" / "data" / "camera_trap_stations.geojson"

THRESHOLD_M = 1000.0

# UTM 18S WGS84 → WGS84 lon/lat
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
    for z in (HYDRIC_ZIP, ROADS_ZIP):
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
# Main roads — Sendero Completo BP 2 filtered to two named vehicular roads
#
# We previously took everything with Tipo == "Vehicular", which included
# CAMINO SERVISIO — a knot of 17 short branches around the entrance service
# area that read as visual noise on the maps. The two through-roads we want
# are PUMA and SENDERO ARAUCARIAS.

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
# 1000 m contour and high-zone polygon

def load_all_contours_lonlat() -> list[tuple[float, LineString]]:
    """Returns [(elev_m, LineString), ...] in WGS84 lon/lat."""
    sf = shapefile.Reader(str(find_shp("cn_5m")), encoding="latin-1")
    out = []
    for shp_rec in sf.shapeRecords():
        elev = float(shp_rec.record["ELEV"])
        coords_ll = reproj_line(shp_rec.shape.points)
        if len(coords_ll) < 2:
            continue
        out.append((elev, LineString(coords_ll)))
    return out


def build_threshold_contour(
    contours: list[tuple[float, LineString]], bp_poly: Polygon
) -> tuple[list[dict], MultiLineString]:
    at_threshold = [ln for elev, ln in contours if elev == THRESHOLD_M]
    if not at_threshold:
        raise ValueError(f"No contour at {THRESHOLD_M} m in cn_5m")
    merged = linemerge(MultiLineString(at_threshold)) if len(at_threshold) > 1 else at_threshold[0]
    clipped = merged.intersection(bp_poly)
    if isinstance(clipped, (LineString, MultiLineString)):
        geom_for_feature = clipped
    else:
        # Could be GeometryCollection if any odd touches; keep only line parts
        parts = [g for g in getattr(clipped, "geoms", []) if g.geom_type in ("LineString", "MultiLineString")]
        if not parts:
            raise ValueError("Threshold contour did not intersect BP polygon as lines")
        geom_for_feature = unary_union(parts)
    feat = {
        "type": "Feature",
        "properties": {"elev_m": THRESHOLD_M},
        "geometry": mapping(geom_for_feature),
    }
    return [feat], geom_for_feature if hasattr(geom_for_feature, "geoms") else MultiLineString([geom_for_feature])


def build_high_zone_from_contour(
    bp_poly: Polygon,
    threshold_line,
    all_contours: list[tuple[float, LineString]],
) -> list[dict]:
    """
    Build the high-zone polygon by:
      1. Take the 1000 m contour (a MultiLineString clipped to BP).
      2. Extend each segment's endpoints outward so the line fully partitions
         the BP polygon.
      3. Split BP along the extended line(s).
      4. Classify each resulting piece by majority vote of sample points:
         each sample is matched to its nearest contour ELEV; the piece is
         "high" iff most samples are nearest a contour > THRESHOLD_M.
      5. Filter contours to those near BP (≤ 0.005° ≈ 500 m) when classifying,
         to avoid contamination from absurd 5–25 m features west of BP that
         pollute the nearest-neighbor query.
    """
    bp_minx, bp_miny, bp_maxx, bp_maxy = bp_poly.bounds
    extent = max(bp_maxx - bp_minx, bp_maxy - bp_miny) * 5

    if threshold_line.geom_type == "LineString":
        segments = [threshold_line]
    else:
        segments = list(threshold_line.geoms)

    extended_segments = []
    for seg in segments:
        coords = list(seg.coords)
        if len(coords) < 2:
            continue
        (x0, y0), (x1, y1) = coords[0], coords[1]
        dx, dy = x0 - x1, y0 - y1
        norm = (dx**2 + dy**2) ** 0.5 or 1.0
        ext0 = (x0 + dx / norm * extent, y0 + dy / norm * extent)
        (x0, y0), (x1, y1) = coords[-2], coords[-1]
        dx, dy = x1 - x0, y1 - y0
        norm = (dx**2 + dy**2) ** 0.5 or 1.0
        ext1 = (x1 + dx / norm * extent, y1 + dy / norm * extent)
        extended_segments.append(LineString([ext0, *coords, ext1]))

    splitter = unary_union(extended_segments)
    try:
        pieces = list(split(bp_poly, splitter).geoms)
    except Exception as e:
        print(f"  ! shapely.split failed ({e}); falling back to single piece")
        pieces = [bp_poly]

    # Pre-filter contours by ELEVATION RANGE — BP terrain is realistically
    # 800–1400 m based on the 26 corrected CT altitudes. Drops artifact
    # contours (5–25 m) that pollute nearest-neighbor queries.
    ELEV_MIN, ELEV_MAX = 700.0, 1500.0
    near_bp = [(e, ln) for e, ln in all_contours if ELEV_MIN <= e <= ELEV_MAX]
    print(f"  contours in [{ELEV_MIN:.0f}, {ELEV_MAX:.0f}] m: "
          f"{len(near_bp)} / {len(all_contours)} kept")

    # CTs as classification ground truth: any piece that contains one or more
    # CTs is classified by its CTs' altitudes. Pieces with no CT inside fall
    # back to majority vote of sample points' nearest contour ELEV.
    stations = json.loads(STATIONS_GEOJSON.read_text(encoding="utf-8"))
    ct_points = [
        (Point(ft["geometry"]["coordinates"]), float(ft["properties"]["altitude_m"]))
        for ft in stations["features"]
        if ft["properties"].get("altitude_m") is not None
    ]

    high_pieces = []
    low_pieces = []
    for piece in pieces:
        if piece.is_empty or piece.area < 1e-10:
            continue
        cts_inside = [(p, a) for p, a in ct_points if piece.contains(p)]
        if cts_inside:
            # Ground truth: majority of CTs in this piece
            n_hi = sum(1 for _, a in cts_inside if a > THRESHOLD_M)
            n_lo = len(cts_inside) - n_hi
            classified_high = n_hi >= n_lo
        else:
            # Fallback: contour-based vote
            samples = _sample_points_in_polygon(piece, n=25)
            high_votes = sum(
                1 for s in samples
                if min((s.distance(ln), e) for e, ln in near_bp)[1] > THRESHOLD_M
            )
            classified_high = high_votes > len(samples) - high_votes
        (high_pieces if classified_high else low_pieces).append(piece)

    high_union = unary_union(high_pieces) if high_pieces else None
    if high_union is None:
        raise ValueError("No pieces classified as high")
    if high_union.geom_type == "Polygon":
        high_union = MultiPolygon([high_union])

    feat = {
        "type": "Feature",
        "properties": {
            "threshold_m": THRESHOLD_M,
            "side": "above",
            "method": "cn_5m-1000m-contour-split",
        },
        "geometry": mapping(high_union),
    }
    print(f"  high pieces: {len(high_pieces)}, low pieces: {len(low_pieces)}")
    print(f"  high zone area fraction of BP: {high_union.area / bp_poly.area:.1%}")
    return [feat]


def _sample_points_in_polygon(poly: Polygon, n: int = 25) -> list[Point]:
    """Grid-sample up to n points inside the polygon. Includes representative_point."""
    minx, miny, maxx, maxy = poly.bounds
    side = max(2, int(n**0.5) + 1)
    samples = [poly.representative_point()]
    for i in range(side):
        for j in range(side):
            x = minx + (i + 0.5) * (maxx - minx) / side
            y = miny + (j + 0.5) * (maxy - miny) / side
            p = Point(x, y)
            if poly.contains(p):
                samples.append(p)
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Main


def main() -> None:
    print(f"Working dirs:")
    print(f"  REPORT_ROOT = {REPORT_ROOT}")
    print(f"  ZIPS_DIR    = {ZIPS_DIR}")
    print(f"  OUT_DIR     = {OUT_DIR}")

    print("\nExtracting zips...")
    extract_zips()

    print("\nLoading BP boundary...")
    bp_poly = load_boundary_poly()
    print(f"  bounds: {bp_poly.bounds}")
    print(f"  area  : {bp_poly.area:.6f} sq.deg")

    print("\nBuilding main rivers (esterosBP, dissolved by Nombre)...")
    rivers = build_main_rivers(bp_poly)
    write_geojson(rivers, OUT_DIR / "hydric_main.geojson")

    print("\nBuilding main roads (Tipo == Vehicular)...")
    roads = build_main_roads(bp_poly)
    write_geojson(roads, OUT_DIR / "roads_main.geojson")

    print(f"\nLoading all contours (cn_5m) and projecting to lon/lat...")
    contours = load_all_contours_lonlat()
    print(f"  loaded {len(contours)} contour line segments")

    print(f"\nBuilding threshold contour at {THRESHOLD_M} m...")
    thr_features, thr_line = build_threshold_contour(contours, bp_poly)
    write_geojson(thr_features, OUT_DIR / "bp_threshold_contour.geojson")

    print(f"\nBuilding high-zone polygon (above {THRESHOLD_M} m, via cn_5m 1000 m contour)...")
    high_features = build_high_zone_from_contour(bp_poly, thr_line, contours)
    write_geojson(high_features, OUT_DIR / "bp_high_zone.geojson")

    print("\nDone.")


if __name__ == "__main__":
    main()
