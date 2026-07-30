"""The canonical observation table — what one reviewed camera-trap record *is*.

Every consumer (annual report, pehuen, data-pipeline, platform) reads this shape and
nothing else. It is written once at ingest, so the Timelapse2 export quirks are
resolved in one place rather than re-derived per consumer:

    * `filePath` is populated in otono_2025 / primavera_2025 and empty in
      pv_2025_2026 / otono_2026 -> `rel_path` is always resolved here.
    * `timestamp` is populated only in primavera_2025; `DateTime` everywhere -> the
      repaired `datetime_corrected` from timestamps.py is the only time column here.
    * Three station spellings across four campaigns -> `camera_num` only, via
      `camtrap.stations`.
    * Reviewers using "Otro (especificar)" leave `scientificName` empty with the
      Spanish name in `observationComments` -> resolved to `species_latin` here.

Deliberately NOT stored: species attributes (`spanish`, `taxonomic_group`,
`is_invasive`, `is_priority`). The table carries the species *key*;
`data-pipeline/species.yaml` carries the attributes, joined at use. Baking them in
would freeze a catalogue copy into every Parquet file, so a species.yaml correction
would need a full re-ingest to propagate.

All rows are kept, including non-animal and empty-species rows — `observation_type`
lets consumers filter. Discarding them here would make trap-effort questions
unanswerable from the canonical table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camtrap import stations
from classify_campaign.species import spanish_to_latin

CAMPAIGNS_ROOT = Path(__file__).resolve().parents[1] / "data" / "campaigns"
CANONICAL_FILENAME = "observations.parquet"

# The contract. Consumers may rely on these names and dtypes.
CANONICAL_COLUMNS: dict[str, str] = {
    "campaign":          "string",
    "camera_num":        "Int16",
    "station_canonical": "string",          # CT05 — display/join convenience, derived
    "datetime":          "datetime64[ns]",  # repaired; NaT when unrepairable
    "valid_date":        "boolean",
    "valid_time_of_day": "boolean",
    "repair_method":     "string",
    "observation_type":  "string",
    "species_latin":     "string",          # '' when not an identified animal
    "review_outcome":    "string",
    "file_name":         "string",
    "rel_path":          "string",          # forward slashes, case preserved
}

DEDUP_KEY = ["camera_num", "file_name", "datetime"]

# Chronological order of review. When the same image was reviewed in two campaigns the
# LATER review supersedes the earlier one — the same reason `reviewOutcome=corrected`
# exists. This is NOT alphabetical order, and getting it wrong silently reverts
# adjudicated labels: primavera_2025 and pv_2025_2026 overlap by 396 records, 31 of
# which carry different species, and `label_conflicts_primavera_vs_pv_2026-05-27.csv`
# records pv as the resolution already loaded into DuckDB.
CAMPAIGN_ORDER = [
    "otono_2025",
    "primavera_2025",
    "pv_2025_2026",
    "otono_2026",
]


class UnorderedCampaign(ValueError):
    """A campaign has no entry in CAMPAIGN_ORDER, so precedence is undefined."""


def _precedence(campaign: str) -> int:
    try:
        return CAMPAIGN_ORDER.index(campaign)
    except ValueError as exc:
        raise UnorderedCampaign(
            f"campaign {campaign!r} is not in CAMPAIGN_ORDER. Add it in chronological "
            f"review order — it decides which label wins when two campaigns contain "
            f"the same image."
        ) from exc


def _resolve_rel_path(row: pd.Series) -> str:
    """filePath when populated, else RelativePath + File. Forward slashes."""
    fp = str(row.get("filePath") or "").strip()
    if not fp:
        rel = str(row.get("RelativePath") or "").strip()
        name = str(row.get("File") or "").strip()
        fp = f"{rel}/{name}" if rel and name else name
    return fp.replace("\\", "/")


def to_canonical(corrected: pd.DataFrame, campaign: str) -> pd.DataFrame:
    """Project a timestamps.py-corrected frame onto the canonical schema.

    Raises stations.UnknownStation if any station is unrecognised — see that module
    for why this is fatal rather than a dropped row.
    """
    src = corrected.fillna("")
    stations.validate(src["Deployments"].astype(str), campaign)

    out = pd.DataFrame(index=src.index)
    out["campaign"] = campaign
    out["camera_num"] = [
        stations.resolve(str(s), campaign) for s in src["Deployments"]
    ]
    out["station_canonical"] = [stations.canonical_id(n) for n in out["camera_num"]]
    out["datetime"] = pd.to_datetime(src["datetime_corrected"], errors="coerce")

    for flag in ("valid_date", "valid_time_of_day"):
        out[flag] = src[flag].astype(str).str.strip().str.lower().isin({"true", "1"})

    out["repair_method"] = src["repair_method"].astype(str).str.strip()
    out["observation_type"] = src["observationType"].astype(str).str.strip()
    out["review_outcome"] = src.get("reviewOutcome", "").astype(str).str.strip()
    out["file_name"] = src["File"].astype(str).str.strip()
    out["rel_path"] = src.apply(_resolve_rel_path, axis=1)

    # Species: prefer scientificName; recover "Otro (especificar)" rows from the
    # Spanish common name the reviewer typed into observationComments.
    latin = src["scientificName"].astype(str).str.strip()
    comments = src.get("observationComments", "").astype(str).str.strip().str.lower()
    lookup = spanish_to_latin()
    recovered = comments.map(lookup).fillna("")
    out["species_latin"] = latin.where(latin != "", recovered)

    n_recovered = int(((latin == "") & (out["species_latin"] != "")).sum())
    if n_recovered:
        print(f"    recovered {n_recovered} species from observationComments")

    return out.astype(CANONICAL_COLUMNS).reset_index(drop=True)


def write_canonical(corrected: pd.DataFrame, campaign: str, out_path: Path) -> int:
    """Write the canonical table for one campaign. Returns row count."""
    canonical = to_canonical(corrected, campaign)
    canonical.to_parquet(out_path, index=False)
    return len(canonical)


def read_canonical(campaign: str, *, root: Path = CAMPAIGNS_ROOT) -> pd.DataFrame:
    """One campaign's canonical table."""
    path = root / campaign / CANONICAL_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python timestamps.py --campaign {campaign}` first."
        )
    return pd.read_parquet(path)


def read_campaigns(
    *campaigns: str,
    root: Path = CAMPAIGNS_ROOT,
    dedup: bool = True,
) -> pd.DataFrame:
    """Concatenated canonical tables — the common case for any analysis.

    With dedup=True (default), records appearing in more than one campaign are
    collapsed, keeping the one from the LATEST campaign in CAMPAIGN_ORDER. This is not
    hypothetical: primavera_2025 and pv_2025_2026 overlap by 396 records — the same SD
    cards were partly re-ingested, including one camera-5 card that appears in
    primavera under the unrenamed folder name `100EK113`.

    Any dropped record whose species label DIFFERS from the kept one is reported
    individually. Silent label changes are the one failure mode this must not have.
    """
    frames = [read_canonical(c, root=root) for c in campaigns]
    df = pd.concat(frames, ignore_index=True)
    if not dedup:
        return df

    df["_prec"] = [_precedence(c) for c in df["campaign"]]
    df = df.sort_values(DEDUP_KEY + ["_prec"], na_position="last")

    # keep='last' -> the highest-precedence (latest) campaign survives.
    dups = df.duplicated(subset=DEDUP_KEY, keep="last") & df["datetime"].notna()
    survivors = df[~dups]

    if dups.any():
        dropped = df[dups]
        print(
            f"Cross-campaign dedup : {int(dups.sum())} duplicate record(s) collapsed "
            f"on {tuple(DEDUP_KEY)}; latest campaign wins"
        )
        for (cam, camp), n in dropped.groupby(["camera_num", "campaign"], observed=True).size().items():
            print(f"  camera {cam:>2} — {n:>3} row(s) superseded from {camp}")

        # Merge rather than index lookup: DEDUP_KEY mixes a nullable Int16, a string
        # and a Timestamp, and a silent lookup miss here would hide exactly the
        # label changes this check exists to surface.
        pairs = dropped.merge(
            survivors[DEDUP_KEY + ["species_latin", "campaign"]],
            on=DEDUP_KEY, suffixes=("_dropped", "_kept"),
        )
        conflicts = pairs[
            pairs["species_latin_dropped"].fillna("") != pairs["species_latin_kept"].fillna("")
        ]
        if len(conflicts):
            print(f"  [LABEL CONFLICT] {len(conflicts)} superseded record(s) carried a "
                  f"different species than the surviving copy:")
            for r in conflicts.itertuples():
                old = r.species_latin_dropped or "(none)"
                new = r.species_latin_kept or "(none)"
                print(f"    CT{r.camera_num:02d} {r.file_name}: "
                      f"{r.campaign_dropped} said {old} -> kept {new} "
                      f"({r.campaign_kept})")

    return survivors.drop(columns="_prec").reset_index(drop=True)
