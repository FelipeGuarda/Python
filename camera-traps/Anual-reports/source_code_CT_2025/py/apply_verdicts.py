"""
apply_verdicts.py — Apply Felipe's manual photo-verdicts to the 2025 report data.

Reads (all under source_code_CT_2025/)
--------------------------------------
- data/records_clean_pre_correction.parquet   (preserved snapshot, do not edit)
- data/events_clean_pre_correction.parquet    (preserved snapshot, do not edit)
- data/manual_review_verdicts_2026-06-02.csv  (image-by-image verdicts)
- inputs/species.yaml

Writes
------
- data/records_clean.parquet      (canonical, with verdicts applied)
- data/events_clean.parquet       (canonical, with verdicts applied)
- data/corrections_report.md      (deltas vs pre-revision snapshot)

Logic
-----
For each verdict that matches a row in the pre-correction records snapshot
(joining on campaign + deployment_raw + File):
  - Verdadero  → keep row, no change
  - Falso + corrected_species_latin → rewrite scientificName / spanish /
    taxonomic_group / is_invasive / is_priority via species.yaml
  - Falso + "Sin animales" → drop the row

Verdicts that don't match any record (e.g. 100EK113 — un-mappable and
already excluded by the report; CT08 09260026 — pre-Conaf cutoff) are
ignored with a printed audit.

Then re-build events using the same 30-min episode rule as 01_data_prep.py.

This script is idempotent: it always reads from the preserved pre-correction
snapshots, so re-running it reproduces the canonical parquets exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

# Force UTF-8 on stdout/stderr so this script prints arrows (→) and accented
# species names on a default Windows console (cp1252) without crashing.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# Paths (all resolved relative to this script — no external repo dependency)

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]                      # source_code_CT_2025/
INPUTS = ROOT / "inputs"
DATA = ROOT / "data"
SPECIES_YAML = INPUTS / "species.yaml"

# Read from the archived pre-correction snapshot so the script is idempotent:
# re-running always recomputes the canonical (corrected) parquets from the
# preserved originals + the verdicts CSV.
RECORDS_IN = DATA / "records_clean_pre_correction.parquet"
EVENTS_IN = DATA / "events_clean_pre_correction.parquet"
VERDICTS_IN = DATA / "manual_review_verdicts_2026-06-02.csv"

RECORDS_OUT = DATA / "records_clean.parquet"
EVENTS_OUT = DATA / "events_clean.parquet"
REPORT_OUT = DATA / "corrections_report.md"

EPISODE_GAP = pd.Timedelta(minutes=30)


def load_species_catalog() -> dict[str, dict]:
    """{latin_name: {spanish, taxonomic_group, is_invasive, is_priority}}."""
    with open(SPECIES_YAML, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {
        s["latin"]: {
            "spanish": s["spanish"],
            "taxonomic_group": s["taxonomic_group"],
            "is_invasive": bool(s.get("is_invasive", False)),
            "is_priority": bool(s.get("is_priority", False)),
        }
        for s in data["species"]
    }


def build_events(df: pd.DataFrame, gap: pd.Timedelta = EPISODE_GAP) -> pd.DataFrame:
    df = df.sort_values(["camera_num", "scientificName", "timestamp_corrected"]).copy()
    key = ["camera_num", "scientificName"]
    df["__prev_ts"] = df.groupby(key)["timestamp_corrected"].shift()
    new_event = (df["timestamp_corrected"] - df["__prev_ts"]) > gap
    new_event = new_event.fillna(True)
    df["event_seq"] = new_event.groupby([df["camera_num"], df["scientificName"]]).cumsum()

    events = (
        df.groupby(["camera_num", "scientificName", "event_seq"], dropna=False)
        .agg(
            event_start=("timestamp_corrected", "min"),
            event_end=("timestamp_corrected", "max"),
            n_images=("timestamp_corrected", "size"),
            spanish=("spanish", "first"),
            taxonomic_group=("taxonomic_group", "first"),
            is_invasive=("is_invasive", "first"),
            campaigns=("campaign", lambda s: ",".join(sorted(set(s)))),
        )
        .reset_index(drop=False)
    )
    events = events.drop(columns=["event_seq"])
    events["event_duration_s"] = (
        (events["event_end"] - events["event_start"]).dt.total_seconds().astype(int)
    )
    return events


def per_camera_presence(events: pd.DataFrame) -> pd.DataFrame:
    """One row per species: n_cameras and which cameras."""
    g = events.groupby(["scientificName", "spanish", "is_invasive"], dropna=False)
    return (
        g.agg(
            n_events=("n_images", "size"),
            n_images=("n_images", "sum"),
            n_cameras=("camera_num", "nunique"),
            cameras=("camera_num", lambda s: ",".join(str(int(c)) for c in sorted(set(s)))),
        )
        .reset_index()
        .sort_values("n_events", ascending=False)
    )


def main() -> None:
    print("=" * 78)
    print("apply_verdicts.py — Re-compute 2025 report numbers using manual verdicts")
    print("=" * 78)

    records = pd.read_parquet(RECORDS_IN)
    events_orig = pd.read_parquet(EVENTS_IN)
    verdicts = pd.read_csv(VERDICTS_IN, encoding="utf-8")
    species_cat = load_species_catalog()

    print(f"\nLoaded records_clean.parquet: {len(records):,} rows")
    print(f"Loaded verdicts file        : {len(verdicts):,} verdicts")

    # ── Join verdicts to records on (campaign, deployment_raw, File)
    records["_key"] = (
        records["campaign"].astype(str)
        + "|"
        + records["deployment_raw"].astype(str)
        + "|"
        + records["File"].astype(str)
    )
    verdicts["_key"] = (
        verdicts["campaign"].astype(str)
        + "|"
        + verdicts["Deployments"].astype(str)
        + "|"
        + verdicts["File"].astype(str)
    )

    matched = verdicts[verdicts["_key"].isin(records["_key"])].copy()
    unmatched = verdicts[~verdicts["_key"].isin(records["_key"])].copy()
    print(f"\nVerdicts matched to records : {len(matched):,}")
    print(f"Verdicts NOT in records     : {len(unmatched):,} (already excluded by report)")
    if not unmatched.empty:
        print("  Unmatched verdicts (informational):")
        for _, r in unmatched.iterrows():
            print(
                f"    {r['campaign']} {r['Deployments']} {r['File']} "
                f"({r['original_label']}) — {r['notes'] or ''}"
            )

    # ── Apply Falso verdicts
    verdicts_by_key = matched.set_index("_key")[
        ["verdict", "corrected_species_spanish", "corrected_species_latin"]
    ]

    drops: list[str] = []
    rewrites: list[dict] = []

    corrected = records.copy()
    for key, v in verdicts_by_key.iterrows():
        if v["verdict"] == "Verdadero":
            continue
        # Falso branch
        if (
            pd.isna(v["corrected_species_latin"])
            or v["corrected_species_latin"] == ""
        ):
            # "Sin animales" or similar — drop the row
            drops.append(key)
            continue
        latin = v["corrected_species_latin"]
        meta = species_cat.get(latin)
        if meta is None:
            print(f"  [WARN] {latin!r} not in species.yaml — skipping rewrite for {key}")
            continue
        mask = corrected["_key"] == key
        before_sp = corrected.loc[mask, "scientificName"].iloc[0]
        corrected.loc[mask, "scientificName"] = latin
        corrected.loc[mask, "spanish"] = meta["spanish"]
        corrected.loc[mask, "taxonomic_group"] = meta["taxonomic_group"]
        corrected.loc[mask, "is_invasive"] = meta["is_invasive"]
        corrected.loc[mask, "is_priority"] = meta["is_priority"]
        rewrites.append({"key": key, "from": before_sp, "to": latin})

    if drops:
        corrected = corrected[~corrected["_key"].isin(drops)].copy()

    corrected = corrected.drop(columns=["_key"])
    print(f"\nRewrites applied            : {len(rewrites)}")
    print(f"Rows dropped (sin animales) : {len(drops)}")
    print(f"Corrected records           : {len(corrected):,} rows "
          f"(was {len(records):,})")

    # ── Re-build events
    events_corr = build_events(corrected)
    print(f"Corrected events            : {len(events_corr):,} (was {len(events_orig):,})")

    # ── Write outputs
    corrected.to_parquet(RECORDS_OUT, index=False)
    events_corr.to_parquet(EVENTS_OUT, index=False)
    print(f"\nWrote → {RECORDS_OUT}")
    print(f"Wrote → {EVENTS_OUT}")

    # ── Deltas
    orig_summary = per_camera_presence(events_orig)
    new_summary = per_camera_presence(events_corr)
    delta = orig_summary.merge(
        new_summary,
        on=["scientificName", "spanish", "is_invasive"],
        suffixes=("_orig", "_new"),
        how="outer",
    ).fillna(0)
    for col in ("n_events_orig", "n_events_new", "n_images_orig", "n_images_new",
                "n_cameras_orig", "n_cameras_new"):
        delta[col] = delta[col].astype(int)
    delta["Δ_cameras"] = delta["n_cameras_new"] - delta["n_cameras_orig"]
    delta["Δ_events"] = delta["n_events_new"] - delta["n_events_orig"]
    delta["Δ_images"] = delta["n_images_new"] - delta["n_images_orig"]
    delta = delta.sort_values("Δ_cameras")

    # ── Markdown report
    lines: list[str] = []
    lines.append("# Corrections Report — Informe Anual 2025 CT")
    lines.append("")
    lines.append(f"Source: `{VERDICTS_IN.name}` (Felipe, 2026-06-02)\n")
    lines.append(
        f"Records: {len(records):,} → {len(corrected):,} "
        f"(rewrites={len(rewrites)}, drops={len(drops)})  "
    )
    lines.append(f"Events : {len(events_orig):,} → {len(events_corr):,}\n")
    lines.append("## Per-species delta (cameras / events / images)\n")
    lines.append("| Spanish | Latin | Invasive | Cams (orig→new) | Δ cams | Events (orig→new) | Δ events | Images (orig→new) | Δ imgs |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in delta.iterrows():
        spn = r["spanish"] if pd.notna(r["spanish"]) and r["spanish"] else "—"
        latin = r["scientificName"]
        inv = "✓" if r["is_invasive"] else ""
        lines.append(
            f"| {spn} | {latin} | {inv} | "
            f"{r['n_cameras_orig']}→{r['n_cameras_new']} | {r['Δ_cameras']:+d} | "
            f"{r['n_events_orig']}→{r['n_events_new']} | {r['Δ_events']:+d} | "
            f"{r['n_images_orig']}→{r['n_images_new']} | {r['Δ_images']:+d} |"
        )
    lines.append("")
    lines.append("## Per-camera presence (corrected)\n")
    pres = (
        events_corr.groupby(["scientificName", "spanish"], dropna=False)
        .agg(cameras=("camera_num", lambda s: ", ".join(f"CT{int(c):02d}" for c in sorted(set(s)))))
        .reset_index()
        .sort_values("spanish")
    )
    lines.append("| Spanish | Latin | Cameras |")
    lines.append("|---|---|---|")
    for _, r in pres.iterrows():
        lines.append(f"| {r['spanish']} | {r['scientificName']} | {r['cameras']} |")
    lines.append("")
    lines.append("## Rewrites applied (per image)\n")
    lines.append("| Key | From | To |")
    lines.append("|---|---|---|")
    for w in rewrites:
        lines.append(f"| `{w['key']}` | {w['from']} | {w['to']} |")
    if drops:
        lines.append("")
        lines.append("## Rows dropped (sin animales)\n")
        for d in drops:
            lines.append(f"- `{d}`")

    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote → {REPORT_OUT}")

    # ── Print key deltas
    print("\n" + "=" * 78)
    print("KEY DELTAS (corrected vs original)")
    print("=" * 78)
    print(delta[["spanish", "scientificName", "n_cameras_orig", "n_cameras_new",
                 "Δ_cameras", "Δ_events", "Δ_images"]].to_string(index=False))


if __name__ == "__main__":
    main()
