"""
02_figures_tables.py — Produce all figures for the 2025 annual report.

Inputs
------
- 2025/data/events_clean.parquet                                   (from step 01)
- plataforma-territorial/data/boundary.geojson                     (canonical BP polygon)
- plataforma-territorial/data/camera_trap_stations.geojson         (26 CT points)

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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon as MplPolygon

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

# ─────────────────────────────────────────────────────────────────────────────
# Style

COL_NATIVA = "#3a7d44"
COL_INTROD = "#bc4749"
COL_GHOST = "#bfbfbf"          # cameras with zero detections
COL_BOUNDARY_FILL = "#f4ede4"
COL_BOUNDARY_EDGE = "#7a6b5d"

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


# ─────────────────────────────────────────────────────────────────────────────
# Map helper


def draw_map_base(ax: plt.Axes, boundary: np.ndarray) -> None:
    """Fill the BP polygon, set extent and aspect."""
    poly = MplPolygon(
        boundary,
        closed=True,
        facecolor=COL_BOUNDARY_FILL,
        edgecolor=COL_BOUNDARY_EDGE,
        linewidth=1.0,
        zorder=1,
    )
    ax.add_patch(poly)

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
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_map_base(ax, boundary)

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
    fig.savefig(out)
    plt.close(fig)


def fig_richness_maps(events: pd.DataFrame, stations: pd.DataFrame, boundary: np.ndarray) -> None:
    natives = events[~events["is_invasive"]]
    invasives = events[events["is_invasive"]]

    _draw_richness(
        _richness_per_camera(events, stations),
        boundary,
        "Riqueza de especies por cámara — Bosque Pehuén",
        FIGS / "03_richness_total.png",
        palette="YlOrBr",
    )
    _draw_richness(
        _richness_per_camera(events, stations, natives),
        boundary,
        "Riqueza de especies nativas por cámara",
        FIGS / "04_richness_nativas.png",
        palette="Greens",
    )
    _draw_richness(
        _richness_per_camera(events, stations, invasives),
        boundary,
        "Riqueza de especies introducidas por cámara",
        FIGS / "05_richness_introducidas.png",
        palette="Reds",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Per-species panel


def fig_per_species_panel(
    events: pd.DataFrame, stations: pd.DataFrame, boundary: np.ndarray
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
        draw_map_base(ax, boundary)
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
    from matplotlib.patches import Patch
    fig.legend(
        handles=[
            Patch(facecolor=COL_NATIVA, label="Nativa"),
            Patch(facecolor=COL_INTROD, label="Introducida"),
            Patch(facecolor=COL_GHOST, label="Cámara sin registro de esa especie"),
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
    print(f"  events  : {len(events):,}")
    print(f"  stations: {len(stations)} (camera_num 1..{stations['camera_num'].max()})")
    print(f"  boundary: {len(boundary)} vertices")

    cams_in_events = set(events["camera_num"].unique())
    cams_missing = sorted(set(stations["camera_num"]) - cams_in_events)
    print(f"  cameras with zero events: {cams_missing}")

    print("\nGenerating figures...")
    fig_top_species(events)
    print("  ✓ 01_top_species.png")
    fig_native_donut(events)
    print("  ✓ 02_native_introduced.png")
    fig_richness_maps(events, stations, boundary)
    print("  ✓ 03_richness_total.png")
    print("  ✓ 04_richness_nativas.png")
    print("  ✓ 05_richness_introducidas.png")
    fig_per_species_panel(events, stations, boundary)
    print("  ✓ 06_panel_por_especie.png")

    print(f"\nAll figures written to {FIGS}")


if __name__ == "__main__":
    main()
