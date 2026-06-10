"""
_debug_compare_contours.py — A/B comparison of two contour sources.

Goal: render the 1005 m contour from the new `curvas de nivel_15 m.shp`
(delivered 2026-06-10) next to the v2 1000 m contour from `cn_5m.shp`
to evaluate whether the new file resolves the two known visual artifacts:
  (1) jagged western edge,
  (2) boundary-parallel "stubs".

Writes:
  - plataforma-territorial/data/basemap/bp_threshold_contour_15m.geojson
    (1005 m line from new 15 m source, clipped to BP)
  - camera-traps/Anual-reports/2025/figures/_debug_contour_compare.png
    (overlay of both lines + CTs + BP polygon)

Non-destructive: existing bp_threshold_contour.geojson, bp_high_zone.geojson
and the report figures are NOT touched.

Run from any cwd; paths are anchored on this file.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import shapefile
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, mapping, shape
from shapely.ops import linemerge, unary_union

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve()
REPORT_ROOT = HERE.parents[1]
REPO = HERE.parents[4]
ZIPS_DIR = REPORT_ROOT.parent  # camera-traps/Anual-reports/

NEW_ZIP = next(ZIPS_DIR.glob("Curvas de nivel_BP-*.zip"))
TMP_DIR = Path("C:/Users/USUARIO/AppData/Local/Temp/curvas_15m_BP")

BOUNDARY_GEOJSON = REPO / "plataforma-territorial" / "data" / "boundary.geojson"
STATIONS_GEOJSON = REPO / "plataforma-territorial" / "data" / "camera_trap_stations.geojson"
EXISTING_THRESHOLD = REPO / "plataforma-territorial" / "data" / "basemap" / "bp_threshold_contour.geojson"

OUT_GEOJSON = REPO / "plataforma-territorial" / "data" / "basemap" / "bp_threshold_contour_15m.geojson"
OUT_PNG = REPORT_ROOT / "figures" / "_debug_contour_compare.png"

NEW_THRESHOLD_M = 1005.0
TX = Transformer.from_crs("EPSG:32718", "EPSG:4326", always_xy=True)


def reproj(coords):
    return [list(TX.transform(x, y)) for x, y, *_ in coords]


def extract():
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(NEW_ZIP) as zf:
        zf.extractall(TMP_DIR)
    matches = list(TMP_DIR.rglob("curvas de nivel_15 m.shp"))
    if not matches:
        raise FileNotFoundError("curvas de nivel_15 m.shp not found in extracted ZIP")
    print(f"  unzipped to {TMP_DIR}")
    return matches[0]


def build_1005_line(shp_path: Path, bp_poly):
    sf = shapefile.Reader(str(shp_path), encoding="latin-1")
    lines = []
    all_elevs = []
    for shp_rec in sf.shapeRecords():
        elev = float(shp_rec.record["ELEV"])
        all_elevs.append(elev)
        if elev != NEW_THRESHOLD_M:
            continue
        coords = reproj(shp_rec.shape.points)
        if len(coords) < 2:
            continue
        lines.append(LineString(coords))
    print(f"  ELEV values seen: {sorted(set(all_elevs))[:5]} ... {sorted(set(all_elevs))[-5:]}")
    print(f"  features at {NEW_THRESHOLD_M:.0f} m: {len(lines)}")
    if not lines:
        raise ValueError(f"No contour at {NEW_THRESHOLD_M} m in the 15 m shapefile")
    merged = linemerge(MultiLineString(lines)) if len(lines) > 1 else lines[0]
    clipped = merged.intersection(bp_poly)
    if clipped.is_empty:
        raise ValueError("1005 m contour does not intersect BP polygon")
    if clipped.geom_type not in ("LineString", "MultiLineString"):
        parts = [g for g in getattr(clipped, "geoms", []) if g.geom_type in ("LineString", "MultiLineString")]
        clipped = unary_union(parts) if parts else clipped
    return clipped


def write_geojson(line_geom, path: Path):
    feat = {
        "type": "Feature",
        "properties": {"elev_m": NEW_THRESHOLD_M, "source": "curvas de nivel_15 m"},
        "geometry": mapping(line_geom),
    }
    fc = {"type": "FeatureCollection", "features": [feat]}
    path.write_text(json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ wrote {path}")


def line_xy(geom):
    """Return list of (xs, ys) tuples for plotting any LineString/MultiLineString."""
    if geom.geom_type == "LineString":
        xs, ys = zip(*geom.coords)
        return [(xs, ys)]
    out = []
    for g in geom.geoms:
        xs, ys = zip(*g.coords)
        out.append((xs, ys))
    return out


def render(bp_poly, new_line, existing_line, ct_points, out_png: Path):
    fig, ax = plt.subplots(figsize=(11, 9), dpi=150)

    # BP polygon outline
    if bp_poly.geom_type == "Polygon":
        polys = [bp_poly]
    else:
        polys = list(bp_poly.geoms)
    for poly in polys:
        xs, ys = zip(*poly.exterior.coords)
        ax.plot(xs, ys, color="#444444", lw=1.0, label="Bosque Pehuén")
        for ring in poly.interiors:
            rx, ry = zip(*ring.coords)
            ax.plot(rx, ry, color="#444444", lw=0.7, ls=":")

    # Existing 1000 m line (v2) — red
    for xs, ys in line_xy(existing_line):
        ax.plot(xs, ys, color="#d62728", lw=1.6, alpha=0.85, label="cn_5m 1000 m (v2 actual)")
        # only label once
        existing_line_label = None  # noqa

    # New 1005 m line — blue
    for xs, ys in line_xy(new_line):
        ax.plot(xs, ys, color="#1f77b4", lw=1.6, alpha=0.85, label="cn_15m 1005 m (nuevo)")

    # CTs
    for (lon, lat, idd, alt) in ct_points:
        color = "#2ca02c" if alt >= 1000 else "#999999"
        ax.scatter([lon], [lat], s=22, c=color, edgecolors="black", linewidths=0.4, zorder=5)
        ax.annotate(idd.replace("TC-", ""), (lon, lat), xytext=(3, 3),
                    textcoords="offset points", fontsize=6, color="black")

    # Dedup legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    ax.legend([h for h, _ in uniq], [l for _, l in uniq], loc="lower right", fontsize=9)

    ax.set_aspect("equal")
    ax.set_title("Contour A/B: cn_5m 1000 m (rojo) vs. cn_15m 1005 m (azul) — Bosque Pehuén",
                 fontsize=11)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  ✓ wrote {out_png}")


def main():
    print(f"NEW_ZIP   : {NEW_ZIP.name}")
    print(f"REPO      : {REPO}")
    print()

    print("[1/5] Extract ZIP")
    shp_path = extract()

    print("[2/5] Load BP polygon")
    bp_poly = shape(json.loads(BOUNDARY_GEOJSON.read_text(encoding="utf-8"))["features"][0]["geometry"])

    print(f"[3/5] Build {NEW_THRESHOLD_M:.0f} m line from new 15 m source")
    new_line = build_1005_line(shp_path, bp_poly)
    write_geojson(new_line, OUT_GEOJSON)

    print("[4/5] Load existing 1000 m line from v2")
    existing = json.loads(EXISTING_THRESHOLD.read_text(encoding="utf-8"))
    existing_line = shape(existing["features"][0]["geometry"])

    print("[5/5] Render comparison PNG")
    stations = json.loads(STATIONS_GEOJSON.read_text(encoding="utf-8"))
    ct_points = []
    for ft in stations["features"]:
        lon, lat = ft["geometry"]["coordinates"]
        idd = ft["properties"]["id"]
        alt = ft["properties"].get("altitude_m") or 0
        ct_points.append((lon, lat, idd, alt))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    render(bp_poly, new_line, existing_line, ct_points, OUT_PNG)

    print()
    print("Done. Inspect the PNG and decide whether to swap source in 00_prepare_basemap.py.")


if __name__ == "__main__":
    main()
