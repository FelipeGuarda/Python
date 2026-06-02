"""
list_ciervo_guina_images.py — Manual-review aid for the 2025 annual report.

Lists every image tagged Cervus elaphus (Ciervo rojo) or Leopardus guigna (Güiña)
in the three campaign review CSVs, with paths to the on-disk thumbnails (when
present on this Windows machine) and to the raw filePath (for the Linux box).

Outputs:
- camera-traps/Anual-reports/2025/data/manual_review_ciervo_guina.csv
- Markdown tables printed to stdout, grouped by species and camera.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REPO = Path(r"C:\Users\USUARIO\Dev\Python")
CAMPAIGNS = REPO / "camera-traps" / "data" / "campaigns"
EXPORTS = REPO / "camera-traps" / "exports"
OUT_DIR = REPO / "camera-traps" / "Anual-reports" / "2025" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAMPAIGN_FILES = {
    "otono_2025": CAMPAIGNS / "otono_2025" / "new_labeled_data_reviewed.csv",
    "primavera_2025": CAMPAIGNS / "primavera_2025" / "new_labeled_data_reviewed.csv",
    "pv_2025_2026": CAMPAIGNS / "pv_2025_2026" / "new_labeled_data_reviewed.csv",
}

# Where the species thumbnails live, per campaign.  primavera_2025 has no
# Windows-side export (Linux-only reprocess).
SPECIES_EXPORT_DIR = {
    "otono_2025": EXPORTS / "Otoño 2025" / "species",
    "primavera_2025": None,
    "pv_2025_2026": EXPORTS / "Primavera-verano 2025-2026" / "species",
}

SPECIES_FOLDER = {
    "Cervus elaphus": "Ciervo_rojo_Cervus_elaphus",
    "Leopardus guigna": "Guina_Leopardus_guigna",
}

SPANISH = {
    "Cervus elaphus": "Ciervo rojo",
    "Leopardus guigna": "Güiña",
}

CONFLICTS_CSV = (
    CAMPAIGNS / "label_conflicts_primavera_vs_pv_2026-05-27.csv"
)

CT_RE = re.compile(r"^(?:CT|TC)0*(\d+)(?:_M.*)?$", re.IGNORECASE)


def extract_camera_num(deployment: str) -> int | None:
    if not isinstance(deployment, str):
        return None
    m = CT_RE.match(deployment.strip())
    return int(m.group(1)) if m else None


def load_campaign(campaign: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, encoding="utf-8")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["DateTime"].astype(str).str.strip(), errors="coerce")
    df["camera_num"] = df["Deployments"].apply(extract_camera_num)
    df["scientificName"] = df["scientificName"].fillna("").astype(str).str.strip()
    df["observationType"] = df["observationType"].fillna("").astype(str).str.strip()
    df["observationComments"] = (
        df.get("observationComments", "").fillna("").astype(str).str.strip()
    )
    df["reviewOutcome"] = df.get("reviewOutcome", "").fillna("").astype(str).str.strip()
    df["File"] = df.get("File", "").fillna("").astype(str).str.strip()
    df["filePath"] = df.get("filePath", "").fillna("").astype(str).str.strip()
    df["RootFolder"] = df.get("RootFolder", "").fillna("").astype(str).str.strip()
    df["Deployments"] = df["Deployments"].fillna("").astype(str).str.strip()
    df["campaign"] = campaign
    return df


def apply_date_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror 01_data_prep.py: flag dropped rows, fix CT15/CT16 +8yr in otono.

    We do NOT drop rows here — for a manual-review listing we want to *see*
    every candidate.  We add a `date_fix` column so the reviewer can spot
    records the report would have excluded.
    """
    df = df.copy()
    df["timestamp_corrected"] = df["timestamp"]
    df["date_fix"] = "none"

    # Otoño CT15/CT16 → +8yr (year-only)
    for cam in (15, 16):
        mask = (
            (df["campaign"] == "otono_2025")
            & (df["camera_num"] == cam)
            & (df["timestamp"].dt.year == 2017)
        )
        if mask.any():
            df.loc[mask, "timestamp_corrected"] = df.loc[mask, "timestamp"] + pd.DateOffset(years=8)
            df.loc[mask, "date_fix"] = "+8yr"

    # Otoño CT19 stuck-clock: report drops these (>=80% identical ts).  Flag.
    mask19 = (
        (df["campaign"] == "otono_2025")
        & (df["camera_num"] == 19)
        & (df["timestamp"].dt.year == 2017)
    )
    if mask19.any():
        df.loc[mask19, "date_fix"] = "dropped_clock_frozen_in_report"

    # Primavera / PV TC16 with 2017 ts: dropped pre-deploy in report.  Flag.
    for camp in ("primavera_2025", "pv_2025_2026"):
        mask16 = (
            (df["campaign"] == camp)
            & (df["camera_num"] == 16)
            & (df["timestamp"].dt.year == 2017)
        )
        if mask16.any():
            df.loc[mask16, "date_fix"] = "dropped_predeploy_in_report"

    return df


def thumbnail_path(row) -> tuple[str, bool]:
    """Return (path-as-string, exists-on-disk).  Empty string if no export for this campaign."""
    base = SPECIES_EXPORT_DIR.get(row["campaign"])
    folder = SPECIES_FOLDER.get(row["scientificName"])
    if base is None or folder is None or not row["File"]:
        return ("", False)
    # exports use deployment + filename, lowercased extension, with `.` → `_` in deployment.
    deploy = row["Deployments"]
    safe_deploy = deploy.replace(".", "_")
    fname = row["File"]
    stem, _, ext = fname.rpartition(".")
    candidate = base / folder / f"{safe_deploy}_{stem}.{ext.lower()}"
    if candidate.exists():
        return (str(candidate), True)
    # Fallback: try as-is filename
    candidate2 = base / folder / f"{safe_deploy}_{fname}"
    if candidate2.exists():
        return (str(candidate2), True)
    # Return the expected (non-existing) path for the reviewer's record
    return (str(candidate), False)


def load_conflicts() -> dict[tuple[str, str], tuple[str, str]]:
    """Return {(deployment, filename): (primavera_label, pv_label)} for label-conflict images."""
    if not CONFLICTS_CSV.exists():
        return {}
    df = pd.read_csv(CONFLICTS_CSV, low_memory=False, encoding="utf-8")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    out: dict[tuple[str, str], tuple[str, str]] = {}
    for _, r in df.iterrows():
        key = (str(r["camera"]).strip(), str(r["fileName"]).strip())
        out[key] = (
            str(r.get("primavera_2025_label", "")).strip(),
            str(r.get("pv_label", "")).strip(),
        )
    return out


def main() -> None:
    print("=" * 78)
    print("list_ciervo_guina_images.py — Manual review for ciervo & güiña")
    print("=" * 78)

    raw = pd.concat(
        [load_campaign(c, p) for c, p in CAMPAIGN_FILES.items()],
        ignore_index=True,
    )
    print(f"\nTotal rows across 3 campaigns: {len(raw):,}")

    df = raw[raw["scientificName"].isin(["Cervus elaphus", "Leopardus guigna"])].copy()
    print(f"Rows tagged ciervo or güiña  : {len(df):,}")

    df = apply_date_corrections(df)

    # Thumbnail paths + on-disk flag
    paths = df.apply(thumbnail_path, axis=1)
    df["thumbnail_path"] = [p[0] for p in paths]
    df["thumbnail_on_disk"] = [p[1] for p in paths]

    # Conflict flag (with the other-pass label for context)
    conflicts = load_conflicts()
    df["label_conflict_pv_vs_primavera"] = [
        (dep, fn) in conflicts for dep, fn in zip(df["Deployments"], df["File"])
    ]
    df["conflict_other_label"] = [
        (
            f"primavera={conflicts[(dep, fn)][0]} | pv={conflicts[(dep, fn)][1]}"
            if (dep, fn) in conflicts
            else ""
        )
        for dep, fn in zip(df["Deployments"], df["File"])
    ]

    # Order columns for review
    cols = [
        "campaign",
        "camera_num",
        "Deployments",
        "scientificName",
        "timestamp",
        "timestamp_corrected",
        "date_fix",
        "File",
        "filePath",
        "thumbnail_path",
        "thumbnail_on_disk",
        "observationComments",
        "reviewOutcome",
        "label_conflict_pv_vs_primavera",
        "conflict_other_label",
    ]
    out = df[cols].sort_values(
        ["scientificName", "camera_num", "timestamp_corrected"]
    ).reset_index(drop=True)

    out_csv = OUT_DIR / "manual_review_ciervo_guina.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\nWrote → {out_csv}")

    # ── Per-camera summary
    print("\n# Per-camera image counts (raw rows in CSVs, before report dedup/event collapse)")
    summary = (
        out.groupby(["scientificName", "camera_num"])
        .agg(n_images=("File", "size"), campaigns=("campaign", lambda s: ",".join(sorted(set(s)))))
        .reset_index()
    )
    print(summary.to_string(index=False))

    # ── Markdown tables
    for latin in ("Cervus elaphus", "Leopardus guigna"):
        sub = out[out["scientificName"] == latin].copy()
        spn = SPANISH[latin]
        print(f"\n\n## {spn} ({latin}) — {len(sub)} image rows\n")
        if sub.empty:
            print("_(no rows)_")
            continue
        header = (
            "| Campaign | CT | Deployment | Date (corrected) | File | Thumbnail (on Windows) | "
            "Source filePath | Review | Conflict? |"
        )
        sep = "|" + "|".join(["---"] * 9) + "|"
        print(header)
        print(sep)
        mapped = sub[sub["camera_num"].notna()]
        unmapped = sub[sub["camera_num"].isna()]
        for _, r in mapped.iterrows():
            thumb = r["thumbnail_path"] if r["thumbnail_path"] else "_(no export — Linux side)_"
            on_disk = "✓" if r["thumbnail_on_disk"] else ("—" if not r["thumbnail_path"] else "✗ missing")
            ts = (
                r["timestamp_corrected"].strftime("%Y-%m-%d %H:%M:%S")
                if pd.notna(r["timestamp_corrected"])
                else "n/a"
            )
            conflict = (
                f"⚠️ {r['conflict_other_label']}" if r["label_conflict_pv_vs_primavera"] else ""
            )
            fix_note = "" if r["date_fix"] == "none" else f" ({r['date_fix']})"
            print(
                f"| {r['campaign']} | CT{int(r['camera_num']):02d} | {r['Deployments']} | "
                f"{ts}{fix_note} | `{r['File']}` | `{thumb}` {on_disk} | "
                f"`{r['filePath']}` | {r['reviewOutcome']} | {conflict} |"
            )
        if not unmapped.empty:
            print(
                f"\n**Unmappable deployments (CT number could not be parsed — dropped by the report):**\n"
            )
            print(header)
            print(sep)
            for _, r in unmapped.iterrows():
                ts = (
                    r["timestamp_corrected"].strftime("%Y-%m-%d %H:%M:%S")
                    if pd.notna(r["timestamp_corrected"])
                    else "n/a"
                )
                print(
                    f"| {r['campaign']} | **?** | {r['Deployments']} | "
                    f"{ts} | `{r['File']}` | _(no export)_ — | "
                    f"`{r['filePath']}` | {r['reviewOutcome']} | (suspected CT05 per session log) |"
                )


if __name__ == "__main__":
    main()
