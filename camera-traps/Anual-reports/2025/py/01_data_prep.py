"""
01_data_prep.py — Build the cleaned record table for the 2025 annual report.

Inputs
------
- camera-traps/data/campaigns/{otono_2025, primavera_2025}/observations.parquet
  (canonical observation tables — produced by `python timestamps.py --campaign <name>`.
  ONE ROW PER STILL in the gated export, not one per reviewed record, so rule 3 below is
  what turns this into a detection table rather than an image inventory.)
- camera-traps/data/CANONICAL_STATE.json  (the published contract; verified on load)
  Each holds ONE ROW PER STILL, not per reviewed record, so most rows are blank/human/
  vehicle and the animal filter below is what narrows it. Row counts jumped ~16x on
  2026-08-19 for that reason; the kept-record count did not move with them.
- data-pipeline/species.yaml                                 (canonical species catalog)

Output
------
- camera-traps/Anual-reports/2025/data/records_baseline.parquet  (one row per image record, before manual verdicts)
- camera-traps/Anual-reports/2025/data/events_baseline.parquet   (one row per 30-min episode, before manual verdicts)

These are the inputs to `apply_verdicts.py`, which writes the canonical
`records_clean.parquet` / `events_clean.parquet` consumed by 02_figures_tables.
- Console summary printed to stdout.

Rules applied
-------------
1. Timestamp validity: rows arrive already clock-repaired by `timestamps.py`, which
   owns that decision (anchors in each campaign's `deployment_anchors.csv`). This
   script only *filters* on the resulting flags — it does not repair. Rows with
   valid_date=FALSE cannot be placed in time and are excluded from the report.
2. Conaf-era cutoff: keep only records with corrected timestamp >= 2024-10-01.
3. Animal filter: keep observation_type == "animal" with non-empty species_latin.
4. Small-species filter: drop all taxonomic_group == "ave" or "invertebrado", plus the
   small mammals {Monito del monte, Ratón cola larga} and the legacy "Rata Negra".
5. 30-min independent-event filter: per (camera, species), consecutive detections
   within 30 minutes collapse into one event.

Conventions
-----------
- camera_num is resolved upstream by `camtrap.stations`; station-name grammars and the
  `100EK113` unrenamed-folder case are no longer this script's concern.
- Records shared by two campaigns are collapsed upstream by `read_campaigns()`, with
  the later campaign's label winning. Any species disagreement is printed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

# camera-traps repo root — so `camtrap` is importable when this runs from py/
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from camtrap import canonical_state
from camtrap.observations import read_campaigns

# Force UTF-8 on stdout/stderr so this script prints arrows (→) and accented
# species names on a default Windows console (cp1252) without crashing.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# Paths

HERE = Path(__file__).resolve()
REPORT_ROOT = HERE.parents[1]               # .../camera-traps/Anual-reports/2025
REPO = HERE.parents[4]                      # .../Python
SPECIES_YAML = REPO / "data-pipeline" / "species.yaml"

OUT_DIR = REPORT_ROOT / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Otoño 2026 is deliberately excluded — this report covers oct 2024 – mar 2026, and that
# campaign runs to may 2026. Reconfirmed 2026-08-19: still out, on scope not oversight.
# pv_2025_2026 dropped 2026-08-19: not a campaign but a second review pass over
# primavera_2025, and it was overriding primavera's re-review — see CAMPAIGN_ORDER in
# camtrap/observations.py. Its data was deleted 2026-08-20 after being measured to
# hold no unique records — see camera-traps/README.md, Campaign History.
REPORT_CAMPAIGNS = ("otono_2025", "primavera_2025")

CONAF_CUTOFF = pd.Timestamp("2024-10-01")
EPISODE_GAP = pd.Timedelta(minutes=30)

# Small mammals to drop alongside all birds.  "Rata negra" is not in species.yaml
# (only "Ratón cola larga" and "Monito del monte" are listed) but appears in the
# legacy 2022-24 dataset, so we match it by name as a defensive measure.
SMALL_MAMMALS_DROP = {"Monito del monte", "Ratón cola larga", "Rata negra"}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers

def load_species_catalog() -> pd.DataFrame:
    """Species attributes, joined onto records by scientificName.

    The canonical table stores only the species *key*; these attributes live here so
    a species.yaml correction propagates without re-ingesting every campaign. The
    Spanish-name recovery that used to live alongside this lookup now happens once,
    upstream, in `camtrap.observations`.
    """
    with open(SPECIES_YAML, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return pd.DataFrame(
        {
            "scientificName": s["latin"],
            "spanish": s["spanish"],
            "taxonomic_group": s["taxonomic_group"],
            "is_invasive": bool(s.get("is_invasive", False)),
            "is_priority": bool(s.get("is_priority", False)),
        }
        for s in data["species"]
    )


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

    species_cat = load_species_catalog()
    print(f"Species catalog rows : {len(species_cat)}")

    # ── Verify the canonical contract BEFORE reading anything.
    #    This report is a consumer, and on 2026-08-19 the table it reads went from 3,359
    #    rows to 35,807 without a single error being raised anywhere. It stayed correct by
    #    luck — it already filtered on observation_type — and luck is not a control. If
    #    the parquets no longer match their published state, stop here rather than
    #    producing a report from a table nobody has vouched for.
    state = canonical_state.verify()
    print(f"Canonical contract  : schema_version {state['schema_version']}, "
          f"{state['n_rows_total']:,} rows across {len(state['campaigns'])} campaigns, "
          f"{state['n_stations_total']} stations")

    # ── Load the canonical observation tables (clock repair, station resolution and
    #    cross-campaign dedup already applied — see camtrap/observations.py)
    print()
    canonical = read_campaigns(*REPORT_CAMPAIGNS)
    canonical_state.assert_columns(canonical, state)
    print(f"\nCanonical records    : {len(canonical):,}")
    print(canonical.groupby("campaign").size().to_string())

    # Report-side column names. The canonical table is the contract; this rename is
    # the only place the report's legacy vocabulary is spoken.
    df = canonical.rename(
        columns={
            "datetime": "timestamp_corrected",
            "species_latin": "scientificName",
            "observation_type": "observationType",
            "file_name": "File",
            "review_outcome": "reviewOutcome",
        }
    )
    df["camera_num"] = df["camera_num"].astype(int)

    # ── Timestamp validity. `timestamps.py` owns the repair; this script only
    #    filters. valid_date=FALSE means the record cannot be placed in time at all,
    #    so it cannot be assigned to a campaign year or tested against the cutoff.
    invalid = df[~df["valid_date"]]
    if not invalid.empty:
        print(f"\n[FLAG] Excluding {len(invalid):,} rows with valid_date=FALSE "
              f"(unrepaired camera-clock resets):")
        print(invalid.groupby(["campaign", "camera_num"]).size().to_string())
    df = df[df["valid_date"]].copy()

    # Time-of-day is unreliable for some repaired rows. No figure in this report is
    # diel, so they are kept — but the count is surfaced so that stays a conscious
    # choice if a time-of-day figure is ever added.
    n_no_tod = int((~df["valid_time_of_day"]).sum())
    if n_no_tod:
        print(f"\nNote: {n_no_tod:,} kept rows have valid_time_of_day=FALSE — fine for "
              f"counts/occupancy, NOT usable for any activity-pattern figure.")

    # ── Conaf-era cutoff
    pre_cutoff = df[df["timestamp_corrected"] < CONAF_CUTOFF]
    print(
        f"\nDropping {len(pre_cutoff):,} records before Conaf cutoff {CONAF_CUTOFF.date()}"
    )
    df = df[df["timestamp_corrected"] >= CONAF_CUTOFF].copy()

    # ── Animal filter
    #    (Spanish-only labels were already recovered upstream in camtrap.observations)
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

    # ── Write outputs (inputs to apply_verdicts.py)
    records_out = OUT_DIR / "records_baseline.parquet"
    events_out = OUT_DIR / "events_baseline.parquet"
    df.drop(columns=["__prev_ts"], errors="ignore").to_parquet(records_out, index=False)
    events.to_parquet(events_out, index=False)
    print(f"\nWrote → {records_out}")
    print(f"Wrote → {events_out}")


if __name__ == "__main__":
    main()
