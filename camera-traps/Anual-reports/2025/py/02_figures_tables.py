"""
02_figures_tables.py — Produce all figures for the 2025 annual report.

Inputs
------
- 2025/data/events_clean.parquet                                   (canonical post-verdict; written by apply_verdicts.py)
- plataforma-territorial/data/boundary.geojson                     (canonical BP polygon)
- plataforma-territorial/data/camera_trap_stations.geojson         (26 CT points)
- plataforma-territorial/data/basemap/hydric_main.geojson          (2 main streams)
- plataforma-territorial/data/basemap/roads_main.geojson           (vehicular roads)
- plataforma-territorial/data/basemap/bp_high_zone.geojson         (area above 1000 m)
- plataforma-territorial/data/basemap/bp_threshold_contour.geojson (1000 m contour)

Outputs (2025/figures/)
-----------------------
- 01_top_species.png            barplot of events per species, colored by origen
- 02_native_introduced.png      donut: nativas vs introducidas (eventos totales)
- 03_richness_total.png         map: # de especies registradas por CT
- 04_richness_nativas.png       map: # de especies nativas por CT
- 05_richness_introducidas.png  map: # de especies introducidas por CT
- 06_panel_por_especie.png      A4 panel — un mini-mapa por especie

Conventions
-----------
- All labels and titles in Spanish.
- Native palette: muted green (#3a7d44). Introduced: muted red (#bc4749).
- CTs with no detections appear as small gray "ghost" points on every map.
- Latitude/longitude shown directly; aspect set via 1/cos(mean_lat) so the
  plot looks geographically correct without bringing in geopandas/CRS.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon as MplPolygon

# Force UTF-8 on stdout/stderr so this script prints arrows (→) and accented
# species names on a default Windows console (cp1252) without crashing.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# Paths

# Resolve the repo root from this file's location so the script runs on any
# machine (Linux/Windows) without edits.
HERE = Path(__file__).resolve()
REPORT_ROOT = HERE.parents[1]               # .../camera-traps/Anual-reports/2025
REPO = HERE.parents[4]                      # .../Python
DATA = REPORT_ROOT / "data"
FIGS = REPORT_ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

EVENTS_PARQUET = DATA / "events_clean.parquet"
BOUNDARY_GEOJSON = REPO / "plataforma-territorial" / "data" / "boundary.geojson"
STATIONS_GEOJSON = REPO / "plataforma-territorial" / "data" / "camera_trap_stations.geojson"
BASEMAP_DIR = REPO / "plataforma-territorial" / "data" / "basemap"
HYDRIC_GEOJSON = BASEMAP_DIR / "hydric_main.geojson"
ROADS_GEOJSON = BASEMAP_DIR / "roads_main.geojson"
HIGH_ZONE_GEOJSON = BASEMAP_DIR / "bp_high_zone.geojson"
THRESHOLD_CONTOUR_GEOJSON = BASEMAP_DIR / "bp_threshold_contour.geojson"
THRESHOLD_M = 1000

# ─────────────────────────────────────────────────────────────────────────────
# Style

COL_NATIVA = "#3a7d44"
COL_INTROD = "#bc4749"
COL_GHOST = "#bfbfbf"          # cameras with zero detections
COL_BOUNDARY_FILL = "#f4ede4"  # "low" zone (below threshold)
COL_BOUNDARY_EDGE = "#7a6b5d"
COL_HIGH_FILL = "#d9c9a8"      # "high" zone overlay (above threshold) — darker tan
COL_CONTOUR = "#8a7359"        # 1000 m contour line
COL_WATER = "#5b8aa8"          # main rivers
COL_ROAD = "#666666"           # main vehicular roads

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# Loaders


def load_stations() -> pd.DataFrame:
    gj = json.loads(STATIONS_GEOJSON.read_text(encoding="utf-8"))
    rows = []
    for ft in gj["features"]:
        p = ft["properties"]
        lon, lat = ft["geometry"]["coordinates"]
        rows.append(
            {
                "camera_num": int(p["tc"]),
                "tc_id": p["id"],
                "name": p["name"],
                "grid_id": p.get("grid_id"),
                "altitude_m": p.get("altitude_m"),
                "lon": lon,
                "lat": lat,
            }
        )
    return pd.DataFrame(rows).sort_values("camera_num").reset_index(drop=True)


def load_boundary() -> np.ndarray:
    gj = json.loads(BOUNDARY_GEOJSON.read_text(encoding="utf-8"))
    geom = gj["features"][0]["geometry"]
    assert geom["type"] == "Polygon", f"Unexpected boundary type {geom['type']}"
    return np.asarray(geom["coordinates"][0])


def _iter_line_coords(geom: dict):
    """Yield each line's coords array from a LineString or MultiLineString geometry."""
    if geom["type"] == "LineString":
        yield np.asarray(geom["coordinates"])
    elif geom["type"] == "MultiLineString":
        for ls in geom["coordinates"]:
            yield np.asarray(ls)


def _iter_polygon_rings(geom: dict):
    """Yield each outer ring (closed) from a Polygon or MultiPolygon geometry."""
    if geom["type"] == "Polygon":
        yield np.asarray(geom["coordinates"][0])
    elif geom["type"] == "MultiPolygon":
        for poly in geom["coordinates"]:
            yield np.asarray(poly[0])


def load_lines_geojson(path: Path) -> list[np.ndarray]:
    """Flatten a GeoJSON of line features into a list of Nx2 arrays."""
    if not path.exists():
        return []
    gj = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for ft in gj["features"]:
        for arr in _iter_line_coords(ft["geometry"]):
            out.append(arr)
    return out


def load_polygons_geojson(path: Path) -> list[np.ndarray]:
    """Flatten a GeoJSON of polygon features into a list of ring arrays (outer ring only)."""
    if not path.exists():
        return []
    gj = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for ft in gj["features"]:
        for arr in _iter_polygon_rings(ft["geometry"]):
            out.append(arr)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Map helper


def draw_map_base(
    ax: plt.Axes,
    boundary: np.ndarray,
    basemap: dict | None = None,
    show_water: bool = True,
    show_roads: bool = True,
    show_high_zone: bool = True,
) -> None:
    """Fill the BP polygon, overlay basemap layers, set extent and aspect.

    `basemap` is a dict with optional keys: high_zone, contour, water, roads.
    Each value is a list of Nx2 numpy arrays. Pass it to avoid re-reading the
    GeoJSONs for every subplot in figure 06.
    """
    poly = MplPolygon(
        boundary,
        closed=True,
        facecolor=COL_BOUNDARY_FILL,
        edgecolor=COL_BOUNDARY_EDGE,
        linewidth=1.0,
        zorder=1,
    )
    ax.add_patch(poly)

    if basemap is None:
        basemap = {}

    # High-elevation zone — semi-transparent darker tan over the BP fill
    if show_high_zone:
        for ring in basemap.get("high_zone", []):
            ax.add_patch(
                MplPolygon(
                    ring,
                    closed=True,
                    facecolor=COL_HIGH_FILL,
                    edgecolor="none",
                    alpha=0.65,
                    zorder=1.3,
                )
            )
        # Threshold contour itself (single 1000 m line)
        for line in basemap.get("contour", []):
            ax.plot(
                line[:, 0],
                line[:, 1],
                color=COL_CONTOUR,
                linewidth=0.9,
                linestyle="-",
                zorder=1.5,
            )

    # Main rivers — thin solid blue
    if show_water:
        for line in basemap.get("water", []):
            ax.plot(
                line[:, 0],
                line[:, 1],
                color=COL_WATER,
                linewidth=1.0,
                linestyle="-",
                zorder=1.7,
            )

    # Main vehicular roads — gray dashed
    if show_roads:
        for line in basemap.get("roads", []):
            ax.plot(
                line[:, 0],
                line[:, 1],
                color=COL_ROAD,
                linewidth=0.9,
                linestyle=(0, (4, 2)),
                zorder=1.8,
            )

    pad_x = (boundary[:, 0].max() - boundary[:, 0].min()) * 0.05
    pad_y = (boundary[:, 1].max() - boundary[:, 1].min()) * 0.05
    ax.set_xlim(boundary[:, 0].min() - pad_x, boundary[:, 0].max() + pad_x)
    ax.set_ylim(boundary[:, 1].min() - pad_y, boundary[:, 1].max() + pad_y)

    mean_lat = float(boundary[:, 1].mean())
    ax.set_aspect(1.0 / math.cos(math.radians(mean_lat)))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def load_basemap() -> dict:
    return {
        "high_zone": load_polygons_geojson(HIGH_ZONE_GEOJSON),
        "contour": load_lines_geojson(THRESHOLD_CONTOUR_GEOJSON),
        "water": load_lines_geojson(HYDRIC_GEOJSON),
        "roads": load_lines_geojson(ROADS_GEOJSON),
    }


def add_basemap_legend(ax: plt.Axes, loc: str = "lower left") -> None:
    """Compact basemap legend for the three standalone richness maps."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=COL_HIGH_FILL, edgecolor="none", alpha=0.65,
              label=f"Sobre {THRESHOLD_M} m"),
        Line2D([0], [0], color=COL_CONTOUR, lw=0.9,
               label=f"Curva {THRESHOLD_M} m"),
        Line2D([0], [0], color=COL_WATER, lw=1.0, label="Curso de agua principal"),
        Line2D([0], [0], color=COL_ROAD, lw=0.9, linestyle=(0, (4, 2)),
               label="Camino vehicular"),
    ]
    leg = ax.legend(handles=handles, loc=loc, frameon=True, fontsize=7,
                    framealpha=0.85, edgecolor="#bbb")
    leg.set_zorder(5)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Top species barplot


def fig_top_species(events: pd.DataFrame) -> None:
    by_sp = (
        events.groupby(["spanish", "is_invasive"], as_index=False)
        .agg(n_events=("event_start", "size"), n_images=("n_images", "sum"))
        .sort_values("n_events", ascending=True)
    )
    colors = [COL_INTROD if inv else COL_NATIVA for inv in by_sp["is_invasive"]]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    bars = ax.barh(by_sp["spanish"], by_sp["n_events"], color=colors, edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("N° de eventos (episodios de 30 min)")
    ax.set_title("Especies registradas en Bosque Pehuén (oct 2024 – mar 2026)")

    # legend
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor=COL_NATIVA, label="Nativa"),
            Patch(facecolor=COL_INTROD, label="Introducida"),
        ],
        loc="lower right",
        frameon=False,
    )
    ax.margins(x=0.08)
    fig.savefig(FIGS / "01_top_species.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Native vs introduced donut


def fig_native_donut(events: pd.DataFrame) -> None:
    grp = events.groupby("is_invasive").agg(
        n_events=("event_start", "size"),
        n_species=("spanish", "nunique"),
    )
    n_nat_sp = int(grp.loc[False, "n_species"])
    n_inv_sp = int(grp.loc[True, "n_species"])
    n_nat_ev = int(grp.loc[False, "n_events"])
    n_inv_ev = int(grp.loc[True, "n_events"])
    labels = [
        f"Nativas\n{n_nat_sp} especies · {n_nat_ev} eventos",
        f"Introducidas\n{n_inv_sp} especies · {n_inv_ev} eventos",
    ]
    sizes = [n_nat_ev, n_inv_ev]
    colors = [COL_NATIVA, COL_INTROD]

    fig, ax = plt.subplots(figsize=(7, 6.5))
    wedges, _texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct=lambda p: f"{p:.0f}%",
        startangle=90,
        pctdistance=0.79,
        labeldistance=1.12,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=11),
    )
    for t in autotexts:
        t.set_color("white")
        t.set_fontweight("bold")
        t.set_fontsize(14)
    ax.set_title("Composición de detecciones por origen", pad=18)
    fig.savefig(FIGS / "02_native_introduced.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figures 3–5 — Richness maps


def _richness_per_camera(
    events: pd.DataFrame, stations: pd.DataFrame, subset: pd.DataFrame | None = None
) -> pd.DataFrame:
    src = events if subset is None else subset
    rich = (
        src.groupby("camera_num")["spanish"]
        .nunique()
        .rename("n_species")
        .reset_index()
    )
    return stations.merge(rich, on="camera_num", how="left").fillna({"n_species": 0})


def _draw_richness(
    df: pd.DataFrame,
    boundary: np.ndarray,
    title: str,
    out: Path,
    palette: str = "YlOrBr",
    basemap: dict | None = None,
    show_legend: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_map_base(ax, boundary, basemap=basemap)

    cams_zero = df[df["n_species"] == 0]
    cams_pos = df[df["n_species"] > 0]

    ax.scatter(
        cams_zero["lon"],
        cams_zero["lat"],
        s=40,
        c=COL_GHOST,
        marker="o",
        edgecolor="white",
        linewidth=0.6,
        zorder=2,
        label="Sin registros",
    )

    if not cams_pos.empty:
        sc = ax.scatter(
            cams_pos["lon"],
            cams_pos["lat"],
            s=70 + cams_pos["n_species"] * 35,
            c=cams_pos["n_species"],
            cmap=palette,
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
            vmin=1,
            vmax=max(2, int(df["n_species"].max())),
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
        cbar.set_label("N° de especies")
        cbar.locator = plt.MaxNLocator(integer=True)
        cbar.update_ticks()

    # CT labels
    for _, r in df.iterrows():
        ax.annotate(
            str(int(r["camera_num"])),
            (r["lon"], r["lat"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=7,
            color="black",
            alpha=0.8,
            zorder=4,
        )

    ax.set_title(title)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    if show_legend:
        # Second legend with basemap entries, anchored bottom-right so it doesn't
        # overlap the "Sin registros" entry on the left.
        add_basemap_legend(ax, loc="lower right")
    fig.savefig(out)
    plt.close(fig)


def fig_richness_maps(
    events: pd.DataFrame, stations: pd.DataFrame, boundary: np.ndarray, basemap: dict
) -> None:
    natives = events[~events["is_invasive"]]
    invasives = events[events["is_invasive"]]

    _draw_richness(
        _richness_per_camera(events, stations),
        boundary,
        "Riqueza de especies por cámara — Bosque Pehuén",
        FIGS / "03_richness_total.png",
        palette="YlOrBr",
        basemap=basemap,
        show_legend=True,  # only on figure 03; the report explains layers once
    )
    _draw_richness(
        _richness_per_camera(events, stations, natives),
        boundary,
        "Riqueza de especies nativas por cámara",
        FIGS / "04_richness_nativas.png",
        palette="Greens",
        basemap=basemap,
    )
    _draw_richness(
        _richness_per_camera(events, stations, invasives),
        boundary,
        "Riqueza de especies introducidas por cámara",
        FIGS / "05_richness_introducidas.png",
        palette="Reds",
        basemap=basemap,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Per-species panel


def fig_per_species_panel(
    events: pd.DataFrame, stations: pd.DataFrame, boundary: np.ndarray, basemap: dict
) -> None:
    # Order species by total events, descending; native first within ties.
    sp_order = (
        events.groupby(["spanish", "is_invasive"], as_index=False)
        .agg(n_events=("event_start", "size"))
        .sort_values(["n_events", "spanish"], ascending=[False, True])
    )
    species = sp_order["spanish"].tolist()
    invasive_lookup = dict(zip(sp_order["spanish"], sp_order["is_invasive"]))

    n = len(species)
    ncols = 3
    nrows = math.ceil(n / ncols)

    # A4 portrait at 200 dpi → 8.27 × 11.69 in.  Leave a bit of margin.
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8.27, 10.5))
    axes_flat = axes.ravel()

    # Global event-count max for consistent bubble scaling
    max_events_per_cam = (
        events.groupby(["spanish", "camera_num"]).size().max()
    )

    for ax, sp in zip(axes_flat, species):
        # Per-subplot: keep high-zone fill + contour + water for context, but
        # drop roads (too noisy at this size).
        draw_map_base(ax, boundary, basemap=basemap, show_roads=False)
        sub = events[events["spanish"] == sp]
        per_cam = sub.groupby("camera_num").size().rename("n").reset_index()
        merged = stations.merge(per_cam, on="camera_num", how="left").fillna({"n": 0})

        cams_zero = merged[merged["n"] == 0]
        cams_pos = merged[merged["n"] > 0]

        ax.scatter(
            cams_zero["lon"],
            cams_zero["lat"],
            s=12,
            c=COL_GHOST,
            marker="o",
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
        color = COL_INTROD if invasive_lookup[sp] else COL_NATIVA
        if not cams_pos.empty:
            ax.scatter(
                cams_pos["lon"],
                cams_pos["lat"],
                s=25 + (cams_pos["n"] / max_events_per_cam) * 220,
                c=color,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.85,
                zorder=3,
            )

        n_events_sp = int(sub.shape[0])
        n_cams_sp = int(cams_pos.shape[0])
        ax.set_title(f"{sp}\n({n_events_sp} eventos · {n_cams_sp} cámaras)", fontsize=10)

    # Hide extra axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend(
        handles=[
            Patch(facecolor=COL_NATIVA, label="Nativa"),
            Patch(facecolor=COL_INTROD, label="Introducida"),
            Patch(facecolor=COL_GHOST, label="Cámara sin registro de esa especie"),
            Patch(facecolor=COL_HIGH_FILL, alpha=0.65,
                  label=f"Sobre {THRESHOLD_M} m"),
            Line2D([0], [0], color=COL_WATER, lw=1.0,
                   label="Curso de agua principal"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )

    fig.suptitle(
        "Distribución de detecciones por especie — Bosque Pehuén\n"
        "(eventos oct 2024 – mar 2026, tamaño ∝ frecuencia)",
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(top=0.90, bottom=0.06, hspace=0.55, wspace=0.10)
    fig.savefig(FIGS / "06_panel_por_especie.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main


def main() -> None:
    print("Loading data...")
    events = pd.read_parquet(EVENTS_PARQUET)
    stations = load_stations()
    boundary = load_boundary()
    basemap = load_basemap()
    print(f"  events  : {len(events):,}")
    print(f"  stations: {len(stations)} (camera_num 1..{stations['camera_num'].max()})")
    print(f"  boundary: {len(boundary)} vertices")
    print(f"  basemap : high_zone={len(basemap['high_zone'])} "
          f"contour={len(basemap['contour'])} water={len(basemap['water'])} "
          f"roads={len(basemap['roads'])}")

    cams_in_events = set(events["camera_num"].unique())
    cams_missing = sorted(set(stations["camera_num"]) - cams_in_events)
    print(f"  cameras with zero events: {cams_missing}")

    print("\nGenerating figures...")
    fig_top_species(events)
    print("  ✓ 01_top_species.png")
    fig_native_donut(events)
    print("  ✓ 02_native_introduced.png")
    fig_richness_maps(events, stations, boundary, basemap)
    print("  ✓ 03_richness_total.png")
    print("  ✓ 04_richness_nativas.png")
    print("  ✓ 05_richness_introducidas.png")
    fig_per_species_panel(events, stations, boundary, basemap)
    print("  ✓ 06_panel_por_especie.png")

    print(f"\nAll figures written to {FIGS}")


if __name__ == "__main__":
    main()
