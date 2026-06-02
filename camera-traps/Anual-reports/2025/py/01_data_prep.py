"""
01_data_prep.py — Build the cleaned record table for the 2025 annual report.

Inputs
------
- camera-traps/data/campaigns/{otono_2025, primavera_2025, pv_2025_2026}/new_labeled_data_reviewed.csv
- camera-traps/Anual-reports/Registro de monitoreo CT.xlsx  (sheet: Registro de instalacion)
- data-pipeline/species.yaml                                 (canonical species catalog)

Output
------
- camera-traps/Anual-reports/2025/data/records_clean.parquet  (one row per image record)
- camera-traps/Anual-reports/2025/data/events_clean.parquet   (one row per 30-min episode)
- Console summary printed to stdout.

Rules applied
-------------
1. Date correction
   - Otoño 2025 CT15 / CT16 / CT19: timestamps recorded as 2017 are clock-bug records.
     Offset = (install_year - 2017) computed from `Registro de instalacion`.
     Records flagged "stuck" (>=80% identical timestamps) are dropped.
   - Primavera 2025 TC16_M13.2 and PV 2025-2026 TC16_M13.2: 2017-dated records belong
     to a pre-redeployment period (per user). We cannot derive a clean offset without
     knowing the original installation date — so these records are DROPPED for this
     report, with the count printed for the user to confirm.
2. Conaf-era cutoff: keep only records with corrected timestamp >= 2024-10-01.
3. Animal filter: keep observationType == "animal" with non-empty scientificName.
4. Small-species filter: drop all taxonomic_group == "ave" or "invertebrado", plus the
   small mammals {Monito del monte, Ratón cola larga} and the legacy "Rata Negra".
5. 30-min independent-event filter: per (camera, species), consecutive detections
   within 30 minutes collapse into one event.

Conventions
-----------
- camera_num: integer 1..26 extracted from the deployment ID (CT01 / TC1_M7.2 → 1).
- "100EK113" and any deployment that cannot be mapped to a CT number is dropped
  with a count printed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# Paths

HERE = Path(__file__).resolve()
REPORT_ROOT = HERE.parents[1]               # .../camera-traps/Anual-reports/2025
REPO = HERE.parents[4]                      # .../Python
CAMPAIGNS = REPO / "camera-traps" / "data" / "campaigns"
REGISTRY_XLSX = REPO / "camera-traps" / "Anual-reports" / "Registro de monitoreo CT.xlsx"
SPECIES_YAML = REPO / "data-pipeline" / "species.yaml"

OUT_DIR = REPORT_ROOT / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAMPAIGN_FILES = {
    "otono_2025": CAMPAIGNS / "otono_2025" / "new_labeled_data_reviewed.csv",
    "primavera_2025": CAMPAIGNS / "primavera_2025" / "new_labeled_data_reviewed.csv",
    "pv_2025_2026": CAMPAIGNS / "pv_2025_2026" / "new_labeled_data_reviewed.csv",
}

CONAF_CUTOFF = pd.Timestamp("2024-10-01")
EPISODE_GAP = pd.Timedelta(minutes=30)

# Small mammals to drop alongside all birds.  "Rata negra" is not in species.yaml
# (only "Ratón cola larga" and "Monito del monte" are listed) but appears in the
# legacy 2022-24 dataset, so we match it by name as a defensive measure.
SMALL_MAMMALS_DROP = {"Monito del monte", "Ratón cola larga", "Rata negra"}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers

CT_RE = re.compile(r"^(?:CT|TC)0*(\d+)(?:_M.*)?$", re.IGNORECASE)


def extract_camera_num(deployment: str) -> int | None:
    """CT01 → 1, TC10_M3.2 → 10. Returns None for un-mappable IDs (e.g. 100EK113)."""
    if not isinstance(deployment, str):
        return None
    m = CT_RE.match(deployment.strip())
    return int(m.group(1)) if m else None


def load_species_catalog() -> tuple[pd.DataFrame, dict[str, str]]:
    """Return (catalog_df, spanish_to_latin lookup).

    The lookup includes both the canonical Spanish name and any
    `spanish_aliases`, all lowercased — used to recover records where the
    reviewer wrote a Spanish common name in `observationComments` but left
    `scientificName` empty (e.g. "chingue", "pudu").
    """
    with open(SPECIES_YAML, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    rows = []
    sp_to_latin: dict[str, str] = {}
    for s in data["species"]:
        rows.append(
            {
                "scientificName": s["latin"],
                "spanish": s["spanish"],
                "taxonomic_group": s["taxonomic_group"],
                "is_invasive": bool(s.get("is_invasive", False)),
                "is_priority": bool(s.get("is_priority", False)),
            }
        )
        sp_to_latin[s["spanish"].lower()] = s["latin"]
        for alias in s.get("spanish_aliases") or []:
            sp_to_latin[alias.lower()] = s["latin"]
    return pd.DataFrame(rows), sp_to_latin


def load_install_registry() -> pd.DataFrame:
    df = pd.read_excel(
        REGISTRY_XLSX,
        sheet_name="Registro de instalacion",
        engine="openpyxl",
    )
    df = df.rename(
        columns={
            "Fecha": "install_date_raw",
            "Hora": "install_time_raw",
            "N° de Cámara Trampa": "camera_num_raw",
            "N° de grilla de monitoreo": "grid_id",
            "N° de tarjeta de memoria": "sd_card",
            "Observadores": "observers",
            "Notas": "notes",
        }
    )
    df = df.dropna(subset=["install_date_raw", "camera_num_raw"]).copy()

    # camera_num may be "22*" etc. — strip non-digits
    df["camera_num"] = (
        df["camera_num_raw"].astype(str).str.extract(r"(\d+)", expand=False).astype("Int64")
    )
    df = df.dropna(subset=["camera_num"]).copy()
    df["camera_num"] = df["camera_num"].astype(int)

    # Combine date + time into a single install timestamp.  Time may be missing
    # for some rows; default to midnight in that case.
    def _combine(row) -> pd.Timestamp:
        d = pd.Timestamp(row["install_date_raw"]).normalize()
        t = row["install_time_raw"]
        if pd.isna(t):
            return d
        # openpyxl returns datetime.time for time-typed cells
        return d + pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)

    df["install_date"] = df.apply(_combine, axis=1)

    return df[["camera_num", "install_date", "grid_id", "sd_card", "notes"]].reset_index(
        drop=True
    )


def load_campaign(campaign: str, path: Path) -> pd.DataFrame:
    """Load one campaign CSV, normalize types."""
    df = pd.read_csv(path, low_memory=False, encoding="utf-8")
    # Strip column whitespace and BOM artifacts
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["DateTime"].astype(str).str.strip(), errors="coerce")
    df["camera_num"] = df["Deployments"].apply(extract_camera_num)
    # pandas 3.x preserves NaN through astype(str); fillna explicitly so the
    # downstream `scientificName != ""` filter actually drops missing values.
    df["scientificName"] = df["scientificName"].fillna("").astype(str).str.strip()
    df["observationType"] = df["observationType"].fillna("").astype(str).str.strip()
    df["observationComments"] = df.get("observationComments", "").fillna("").astype(str).str.strip()
    df["reviewOutcome"] = df.get("reviewOutcome", "").fillna("").astype(str).str.strip()
    df["campaign"] = campaign
    df["deployment_raw"] = df["Deployments"].astype(str)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Date correction

def stuck_camera_check(group: pd.DataFrame, threshold: float = 0.8) -> bool:
    """True if >=`threshold` fraction of timestamps are identical (clock frozen)."""
    if len(group) < 3:
        return False
    counts = group["timestamp"].value_counts(normalize=True)
    return float(counts.iloc[0]) >= threshold


def correct_dates(
    df: pd.DataFrame, registry: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply per-camera date corrections.  Returns (corrected_df, audit_rows)."""
    audit: list[dict] = []
    df = df.copy()
    df["timestamp_corrected"] = df["timestamp"]
    df["date_fix"] = "none"

    install_lookup = registry.set_index("camera_num")["install_date"].to_dict()

    # Otoño 2025 cameras with 2017 clock bug.
    #
    # CT15 and CT16: clean +8yr offset (year-only).  Their 2017 month/day spans
    # are consistent with their installation dates.
    #
    # CT19: a clean +8yr would place all 101 records in 2025-01-01..01-05, i.e.
    # ~25 days *before* the 2025-01-30 installation.  Per user instruction,
    # anchor the first record to the install date instead — preserves the
    # relative spacing of records while pinning the start to a known truth.
    otono_year_offset_cams = {15, 16}
    otono_anchor_cams = {19}
    otono_2017_cams = otono_year_offset_cams | otono_anchor_cams

    for cam in otono_2017_cams:
        mask = (
            (df["campaign"] == "otono_2025")
            & (df["camera_num"] == cam)
            & (df["timestamp"].dt.year == 2017)
        )
        sub = df.loc[mask]
        if sub.empty:
            continue

        install = install_lookup.get(cam)
        if install is None:
            audit.append(
                {
                    "camera_num": cam,
                    "campaign": "otono_2025",
                    "issue": "no_install_date",
                    "n_records": len(sub),
                }
            )
            continue

        if stuck_camera_check(sub):
            audit.append(
                {
                    "camera_num": cam,
                    "campaign": "otono_2025",
                    "issue": "clock_frozen",
                    "n_records": len(sub),
                    "action": "dropped",
                }
            )
            df.loc[mask, "date_fix"] = "dropped_clock_frozen"
            continue

        if cam in otono_year_offset_cams:
            offset = install.year - 2017
            corrected = sub["timestamp"] + pd.DateOffset(years=offset)
            df.loc[mask, "timestamp_corrected"] = corrected
            df.loc[mask, "date_fix"] = f"+{offset}yr"
            audit.append(
                {
                    "camera_num": cam,
                    "campaign": "otono_2025",
                    "issue": "year_2017_offset",
                    "n_records": len(sub),
                    "action": f"+{offset}yr",
                    "install_date": install.date().isoformat(),
                    "corrected_range": (
                        f"{corrected.min().date()} .. {corrected.max().date()}"
                    ),
                }
            )
        else:  # anchor-to-install mode
            delta = install - sub["timestamp"].min()
            corrected = sub["timestamp"] + delta
            df.loc[mask, "timestamp_corrected"] = corrected
            df.loc[mask, "date_fix"] = "anchored_to_install"
            audit.append(
                {
                    "camera_num": cam,
                    "campaign": "otono_2025",
                    "issue": "year_2017_anchored",
                    "n_records": len(sub),
                    "action": f"shift by {delta}",
                    "install_date": install.date().isoformat(),
                    "corrected_range": (
                        f"{corrected.min().date()} .. {corrected.max().date()}"
                    ),
                }
            )

    # TC16_M13.2 in Primavera 2025 and PV 2025-26: 2017 records = pre-redeployment.
    # No reliable offset → drop these records (user confirmed in conversation).
    for camp in ("primavera_2025", "pv_2025_2026"):
        mask = (
            (df["campaign"] == camp)
            & (df["camera_num"] == 16)
            & (df["timestamp"].dt.year == 2017)
        )
        n = int(mask.sum())
        if n == 0:
            continue
        df.loc[mask, "date_fix"] = "dropped_predeploy"
        audit.append(
            {
                "camera_num": 16,
                "campaign": camp,
                "issue": "predeploy_2017_dropped",
                "n_records": n,
                "action": "dropped",
            }
        )

    return df, audit


# ─────────────────────────────────────────────────────────────────────────────
# Episode grouping

def build_events(df: pd.DataFrame, gap: pd.Timedelta = EPISODE_GAP) -> pd.DataFrame:
    """Per (camera_num, scientificName), collapse images within `gap` into one event."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Main

def main() -> None:
    print("=" * 78)
    print("01_data_prep.py — Informe Anual 2025 (Bosque Pehuén)")
    print("=" * 78)

    species_cat, sp_to_latin = load_species_catalog()
    registry = load_install_registry()
    print(f"Species catalog rows : {len(species_cat)}")
    print(f"Install registry rows: {len(registry)}  "
          f"(unique cameras: {registry['camera_num'].nunique()})")

    raw = pd.concat(
        [load_campaign(c, p) for c, p in CAMPAIGN_FILES.items()],
        ignore_index=True,
    )
    print(f"\nRaw records loaded   : {len(raw):,}")
    print(raw.groupby("campaign").size().to_string())

    # ── Drop deployments that don't map to a CT number (e.g., 100EK113)
    unmapped = raw[raw["camera_num"].isna()]
    if not unmapped.empty:
        print(
            f"\n[FLAG] Deployments without CT mapping → dropping {len(unmapped):,} rows"
        )
        print(unmapped.groupby(["campaign", "deployment_raw"]).size().to_string())
    df = raw.dropna(subset=["camera_num"]).copy()
    df["camera_num"] = df["camera_num"].astype(int)

    # ── Date corrections
    df, audit = correct_dates(df, registry)
    print("\n" + "-" * 78)
    print("DATE-FIX AUDIT")
    print("-" * 78)
    for a in audit:
        print(a)

    # Drop records flagged as dropped_*
    drop_mask = df["date_fix"].str.startswith("dropped")
    print(f"\nDropped by date audit: {int(drop_mask.sum()):,}")
    df = df.loc[~drop_mask].copy()

    # ── Conaf-era cutoff
    pre_cutoff = df[df["timestamp_corrected"] < CONAF_CUTOFF]
    print(
        f"\nDropping {len(pre_cutoff):,} records before Conaf cutoff {CONAF_CUTOFF.date()}"
    )
    df = df[df["timestamp_corrected"] >= CONAF_CUTOFF].copy()

    # ── Recover Spanish-only labels.  Some animal records have empty
    # `scientificName` but a Spanish common name in `observationComments`
    # (e.g. "chingue", "pudu") — recover those by mapping through species.yaml.
    recover_mask = (
        (df["observationType"] == "animal")
        & (df["scientificName"] == "")
        & (df["observationComments"] != "")
    )
    recoverable = df.loc[recover_mask].copy()
    recoverable["__recovered_latin"] = (
        recoverable["observationComments"].str.lower().map(sp_to_latin)
    )
    recovered = recoverable.dropna(subset=["__recovered_latin"])
    if not recovered.empty:
        print("\nRecovered records via Spanish-name fallback (species.yaml):")
        print(
            recovered.groupby(["observationComments", "__recovered_latin"])
            .size()
            .to_string()
        )
        df.loc[recovered.index, "scientificName"] = recovered["__recovered_latin"]

    # ── Animal filter
    before = len(df)
    df = df[(df["observationType"] == "animal") & (df["scientificName"] != "")].copy()
    print(f"\nAnimal filter        : {before:,} → {len(df):,} (kept rows with scientificName)")

    # ── Join species catalog
    df = df.merge(species_cat, on="scientificName", how="left")
    unmatched = df[df["taxonomic_group"].isna()]
    if not unmatched.empty:
        print(
            f"\n[FLAG] {len(unmatched):,} records with scientificName not in species.yaml:"
        )
        # Note: when scientificName is the empty string the value_counts table
        # collapses to a single empty-label row, so we print the raw list of
        # distinct values to keep the audit honest.
        print(unmatched["scientificName"].astype(str).value_counts().head(20).to_string())

    # ── Small-species filter
    keep_mam = (df["taxonomic_group"] == "mamifero") & (~df["spanish"].isin(SMALL_MAMMALS_DROP))
    dropped_taxa = df.loc[~keep_mam].copy()
    dropped_summary = (
        dropped_taxa.groupby(["taxonomic_group", "spanish"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    print("\nSmall-species / non-mammal filter — dropping:")
    print(dropped_summary.to_string(index=False))
    df = df[keep_mam].copy()

    # ── Build events
    events = build_events(df)
    print(f"\nFinal record rows    : {len(df):,}")
    print(f"Final event rows     : {len(events):,}  (30-min episode rule per camera×species)")

    # ── Summaries for sign-off
    print("\n" + "=" * 78)
    print("SUMMARY FOR SIGN-OFF")
    print("=" * 78)

    print("\n# Events per camera (top 30)")
    print(
        events.groupby("camera_num")
        .size()
        .sort_values(ascending=False)
        .head(30)
        .to_string()
    )

    print("\n# Events per species (full list — these are the species in the report)")
    sp = (
        events.groupby(["spanish", "scientificName", "is_invasive"], dropna=False)
        .agg(n_events=("n_images", "size"), n_images=("n_images", "sum"))
        .reset_index()
        .sort_values("n_events", ascending=False)
    )
    print(sp.to_string(index=False))

    n_native = int((~sp["is_invasive"]).sum())
    n_invasive = int(sp["is_invasive"].sum())
    print(
        f"\nSpecies kept: {len(sp)}  "
        f"(nativas: {n_native}, introducidas: {n_invasive})"
    )

    print("\n# Records per (campaign, year) AFTER correction")
    df["year"] = df["timestamp_corrected"].dt.year
    print(df.groupby(["campaign", "year"]).size().to_string())

    # ── Write outputs
    df.drop(columns=["__prev_ts"], errors="ignore").to_parquet(
        OUT_DIR / "records_clean.parquet", index=False
    )
    events.to_parquet(OUT_DIR / "events_clean.parquet", index=False)
    print(f"\nWrote → {OUT_DIR / 'records_clean.parquet'}")
    print(f"Wrote → {OUT_DIR / 'events_clean.parquet'}")


if __name__ == "__main__":
    main()
