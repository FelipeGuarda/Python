"""
03_compute_proximity.py — diagnostic for road/water proximity thresholds.

For each of the 26 camera traps, compute the shortest geometric distance
(in meters) to:
  - the main road network   (roads_main.geojson — PUMA + SENDERO ARAUCARIAS)
  - the main hydric network (hydric_main.geojson — Estero la Cascada + San Marcos)

Distances are computed in UTM 18S (EPSG:32718). The CT points and the
linework all originate from this projection so the reprojection round-trip
is faithful.

Outputs (all stdout, no files written):
  1. Per-CT table sorted by camera_num, with elevation zone + road/water dist.
  2. Percentile summary (P10/P25/P50/P75/P90) for each distance.
  3. ASCII histogram per distance.
  4. Candidate thresholds with the resulting near/far split per CT.

Intended as a one-shot to inform the threshold decision before any figures
or per-species analysis. No outputs are persisted — re-run as needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, Point, shape
from shapely.ops import unary_union

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve()
REPORT_ROOT = HERE.parents[1]
REPO = HERE.parents[4]

STATIONS_GEOJSON = REPO / "plataforma-territorial" / "data" / "camera_trap_stations.geojson"
ROADS_GEOJSON = REPO / "plataforma-territorial" / "data" / "basemap" / "roads_main.geojson"
HYDRIC_GEOJSON = REPO / "plataforma-territorial" / "data" / "basemap" / "hydric_main.geojson"

THRESHOLD_ELEV_M = 1005.0

TX = Transformer.from_crs("EPSG:4326", "EPSG:32718", always_xy=True)


def to_utm(lon: float, lat: float) -> tuple[float, float]:
    x, y = TX.transform(lon, lat)
    return x, y


def reproject_geometry(geom: dict):
    """Reproject a GeoJSON LineString or MultiLineString to UTM 18S as shapely."""
    g = shape(geom)
    if g.geom_type == "LineString":
        return LineString([to_utm(x, y) for x, y in g.coords])
    if g.geom_type == "MultiLineString":
        return MultiLineString([
            LineString([to_utm(x, y) for x, y in seg.coords])
            for seg in g.geoms
        ])
    raise TypeError(f"Unsupported geometry: {g.geom_type}")


def load_stations() -> list[dict]:
    gj = json.loads(STATIONS_GEOJSON.read_text(encoding="utf-8"))
    out = []
    for ft in gj["features"]:
        p = ft["properties"]
        lon, lat = ft["geometry"]["coordinates"]
        x, y = to_utm(lon, lat)
        out.append({
            "camera_num": int(p["tc"]),
            "id": p["id"],
            "altitude_m": float(p["altitude_m"]) if p.get("altitude_m") is not None else None,
            "point": Point(x, y),
        })
    out.sort(key=lambda r: r["camera_num"])
    return out


def load_lines(path: Path):
    gj = json.loads(path.read_text(encoding="utf-8"))
    geoms = []
    names = []
    for ft in gj["features"]:
        geoms.append(reproject_geometry(ft["geometry"]))
        names.append(ft["properties"].get("name", "?"))
    union = unary_union(geoms)
    return geoms, names, union


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def ascii_hist(values: list[float], n_bins: int = 10, width: int = 40) -> list[str]:
    if not values:
        return ["  (no data)"]
    lo, hi = min(values), max(values)
    if hi == lo:
        hi = lo + 1.0
    edges = [lo + (hi - lo) * i / n_bins for i in range(n_bins + 1)]
    counts = [0] * n_bins
    for v in values:
        idx = min(int((v - lo) / (hi - lo) * n_bins), n_bins - 1)
        counts[idx] += 1
    max_c = max(counts) or 1
    lines = []
    for i in range(n_bins):
        bar_len = int(counts[i] / max_c * width)
        lines.append(f"  [{edges[i]:7.1f} – {edges[i+1]:7.1f}] "
                     f"{'█' * bar_len:<{width}} {counts[i]:>2}")
    return lines


def report_thresholds(label: str, dists: dict[int, float], candidates: list[float]) -> None:
    print(f"\n  Candidate thresholds for {label}:")
    print(f"  {'thr (m)':>8}  {'near':>4}  {'far':>4}  near CTs")
    for thr in candidates:
        near_cams = sorted(c for c, d in dists.items() if d <= thr)
        far_cams = sorted(c for c, d in dists.items() if d > thr)
        near_str = ", ".join(f"CT{c}" for c in near_cams) or "(none)"
        print(f"  {thr:>8.0f}  {len(near_cams):>4}  {len(far_cams):>4}  {near_str}")


def main() -> None:
    print("Loading inputs...")
    stations = load_stations()
    road_geoms, road_names, road_union = load_lines(ROADS_GEOJSON)
    water_geoms, water_names, water_union = load_lines(HYDRIC_GEOJSON)
    print(f"  stations: {len(stations)}")
    print(f"  roads   : {len(road_geoms)} features ({', '.join(road_names)})")
    print(f"  waters  : {len(water_geoms)} features ({', '.join(water_names)})")

    print("\nProjection: WGS84 (EPSG:4326) → UTM 18S (EPSG:32718); distances in metres.")

    rows = []
    for st in stations:
        d_road = st["point"].distance(road_union)
        d_water = st["point"].distance(water_union)
        zona_elev = "alta" if (st["altitude_m"] is not None and st["altitude_m"] > THRESHOLD_ELEV_M) else "baja"
        rows.append({
            "camera_num": st["camera_num"],
            "id": st["id"],
            "altitude_m": st["altitude_m"],
            "zona_elev": zona_elev,
            "road_dist_m": d_road,
            "water_dist_m": d_water,
        })

    # ── Per-CT table
    print("\n" + "=" * 78)
    print("Per-CT proximity table")
    print("=" * 78)
    print(f"  {'CT':>3}  {'alt_m':>6}  {'zona':>5}  {'d_road_m':>9}  {'d_water_m':>10}")
    for r in rows:
        alt = f"{r['altitude_m']:.0f}" if r['altitude_m'] is not None else "—"
        print(f"  {r['camera_num']:>3}  {alt:>6}  {r['zona_elev']:>5}  "
              f"{r['road_dist_m']:>9.1f}  {r['water_dist_m']:>10.1f}")

    # ── Percentile summary
    road_dists = sorted(r["road_dist_m"] for r in rows)
    water_dists = sorted(r["water_dist_m"] for r in rows)

    print("\n" + "=" * 78)
    print("Percentile summary (metres)")
    print("=" * 78)
    print(f"  {'feature':<8}  {'P10':>8}  {'P25':>8}  {'P50':>8}  {'P75':>8}  {'P90':>8}  {'max':>8}")
    for label, ds in [("roads", road_dists), ("water", water_dists)]:
        print(f"  {label:<8}  "
              f"{percentile(ds, 0.10):>8.1f}  "
              f"{percentile(ds, 0.25):>8.1f}  "
              f"{percentile(ds, 0.50):>8.1f}  "
              f"{percentile(ds, 0.75):>8.1f}  "
              f"{percentile(ds, 0.90):>8.1f}  "
              f"{ds[-1]:>8.1f}")

    # ── ASCII histograms
    print("\n" + "=" * 78)
    print("Distribution — distance to nearest road (m)")
    print("=" * 78)
    for line in ascii_hist(road_dists, n_bins=10):
        print(line)

    print("\n" + "=" * 78)
    print("Distribution — distance to nearest waterway (m)")
    print("=" * 78)
    for line in ascii_hist(water_dists, n_bins=10):
        print(line)

    # ── Candidate thresholds
    road_by_cam = {r["camera_num"]: r["road_dist_m"] for r in rows}
    water_by_cam = {r["camera_num"]: r["water_dist_m"] for r in rows}

    print("\n" + "=" * 78)
    print("Candidate splits")
    print("=" * 78)
    report_thresholds("ROADS", road_by_cam, [50, 100, 200, 500,
                                             percentile(road_dists, 0.50)])
    report_thresholds("WATER", water_by_cam, [25, 50, 100, 200,
                                              percentile(water_dists, 0.50)])

    # ── Elevation reference (for sanity)
    n_high = sum(1 for r in rows if r["zona_elev"] == "alta")
    n_low = len(rows) - n_high
    high_cams = sorted(r["camera_num"] for r in rows if r["zona_elev"] == "alta")
    low_cams = sorted(r["camera_num"] for r in rows if r["zona_elev"] == "baja")
    print("\n" + "=" * 78)
    print("Reference — elevation split (already settled)")
    print("=" * 78)
    print(f"  threshold: {THRESHOLD_ELEV_M:.0f} m   high: {n_high}   low: {n_low}")
    print(f"  high CTs: {', '.join(f'CT{c}' for c in high_cams)}")
    print(f"  low  CTs: {', '.join(f'CT{c}' for c in low_cams)}")


if __name__ == "__main__":
    main()
