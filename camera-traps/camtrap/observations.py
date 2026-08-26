"""The canonical observation table — what one camera-trap record *is*.

ONE ROW PER STILL IN THE GATED EXPORT, not one row per reviewed record. Changed
2026-08-19; see `compose_ingest_frame` for why a reviewed-only row set made a
zero-detection station indistinguishable from one that was never deployed.

Every consumer (annual report, pehuen, data-pipeline, platform) reads this shape and
nothing else. It is written once at ingest, so the Timelapse2 export quirks are
resolved in one place rather than re-derived per consumer:

    * `rel_path` is always resolved here rather than read. Measured 2026-08-26 across
      all three `ImageData_total.csv` files: `filePath` and `timestamp` are present as
      columns and **empty in every row of all three** (0 / 8,997 · 0 / 16,904 ·
      0 / 9,906), while `RelativePath` and `DateTime` are complete. This comment used
      to say `filePath` was populated in two campaigns and `timestamp` in one; that
      was true of the exports it was written against and stopped being true when they
      were re-made. Which is the point: the guarantee is "resolved in one place", not
      any particular list of quirks, and a list of quirks in a comment decays.
    * The repaired `datetime_corrected` from timestamps.py is the only time column
      here — never a raw export timestamp, whatever a future export populates.
    * A camera clock can fail three separable ways -> `valid_date`,
      `valid_time_of_day` and `valid_effort` travel with every row rather than one
      usable/not-usable flag. A pure year error preserves time-of-day exactly, so
      those rows stay valid for activity and overlap analysis.
    * Three station spellings across four campaigns -> `camera_num` only, via
      `camtrap.stations`.
    * A reviewer's verdict lives in two columns that can disagree — the typed
      `scientificName` and a free-text Spanish `observationComments` that may hold a
      species, a negation ("no es un animal") or a note to self -> `observation_type`,
      `species_latin` and `review_resolution` all come from `resolve_review()` here.
      `observationType` as exported is NOT copied through: it reads `animal` on every
      row of an animal-only export, including rows the reviewer said hold no animal.

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

from camtrap import exports, stations
from camtrap import episodes
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
    # Station-level, not per-row: FALSE means this camera's operating period is
    # unknown, so its trap-nights are unknowable and it must leave the effort
    # DENOMINATOR as well as the numerator. Every row of an excluded station carries
    # FALSE, including rows whose own date is fine — see camtrap/clocks.py.
    "valid_effort":      "boolean",
    "repair_method":     "string",
    "observation_type":  "string",
    "species_latin":     "string",          # '' when not an identified animal
    "review_outcome":    "string",
    # Which rule resolved this row's type and species — see RESOLUTION_* below. It
    # travels in the table rather than in a log because a marker nobody can query is a
    # marker nobody acts on. `sweep_only` is the majority: a still the sweep categorised
    # and the species review never opened.
    "review_resolution": "string",
    "file_name":         "string",
    "rel_path":          "string",          # forward slashes, case preserved
    # The reviewer's own words, carried verbatim. `review_resolution` says WHICH rule
    # fired; this says what the reviewer actually wrote, and the two answer different
    # questions. It is here because the 815-row type defect was only findable by reading
    # this text — a table that resolves comments but discards them cannot be audited
    # against the reviewer again. Empty on sweep-only rows.
    "observation_comments": "string",
    # CLIP's cosine similarity for the proposed species, as recorded by the classifier
    # before human review. 509/502/800 distinct values across the three campaigns, so it
    # is real per-row data and unrecoverable once the Timelapse export is gone. NaN on
    # sweep-only rows — CLIP never saw them.
    "classification_probability": "Float64",
    # The independence rule, decided upstream and carried rather than re-derived:
    # `campaign|station|species|n`, one id per detection episode, `pd.NA` where the row
    # has no identified species or no clock. Named for its threshold on purpose — a
    # different gap is a different column, so two analyses cannot compare counts built
    # on different definitions of one event. See camtrap/episodes.py.
    "episode_30min":     "string",
}

DEDUP_KEY = ["camera_num", "file_name", "datetime"]

# Chronological order of review. When the same image was reviewed in two campaigns the
# LATER review supersedes the earlier one — the same reason `reviewOutcome=corrected`
# exists. This is NOT alphabetical order, and getting it wrong silently reverts
# adjudicated labels.
#
# `pv_2025_2026` was REMOVED 2026-08-19. It was never a campaign — it is a second
# review pass over primavera_2025, which the field record settles outright. While it sat
# here it outranked primavera, so once primavera was re-ingested from the full 26-station
# download its 606 freshly reviewed rows were being silently replaced by pv's April
# labels: `read_campaigns` returned 169 primavera rows instead of 744, reverting
# adjudicated species (CT20 09240308 went from Pteroptochos tectus back to Lepus
# europaeus). The directory and its parquet are kept as provenance; reading them through
# `read_campaigns` now raises UnorderedCampaign, which is the intended fail-loud.
CAMPAIGN_ORDER = [
    "otono_2025",
    "primavera_2025",
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


# =============================================================================
# Review-comment resolution
# =============================================================================
#
# The review pass records its conclusion in TWO columns that can disagree with each
# other and with the category sweep in ImageData_total.csv:
#
#     scientificName        the typed identification, often empty
#     observationComments   free-text Spanish — sometimes a species, sometimes a
#                           negation ("no es un animal"), sometimes a note to self
#
# Left unresolved this is not a cosmetic problem: 815 rows across the three campaigns
# carried observationType=animal while the reviewer had written that the frame holds no
# animal, which overstated primavera's animal count by 50.6% (744 against 494) and
# counted 10 people and 4 vehicles as animals.
#
# TWO PRECEDENCE RULES, decided with Felipe 2026-08-19, and they point opposite ways:
#
#   1. The review NAMES an animal, the sweep says human or vehicle -> animal wins.
#      A vehicle moving through the park carries people, and dogs follow them, so all
#      three are in frame at once and only one observationType fits. Ranking is
#      animal > vehicle > human. 37 rows: 13 Perro, 23 Caballo, 1 Vaca.
#
#   2. The review NEGATES the animal, the sweep says animal -> the review wins.
#      Here the sweep's `animal` is the false positive the review is correcting, so
#      rule 1 must NOT fire, or all 250 of primavera's corrections revert.
#
# The discriminator is whether the review NAMES something or NEGATES it, which is why
# R1 tests scientificName and R2 tests the comment. Verified 2026-08-19: no row in any
# campaign has both a species and a negating comment, so the two never race.
#
# The sweep's own observationType is deliberately NOT an input. An earlier draft let a
# sweep `human` outrank a negating comment on the grounds that the specific beats the
# generic; Felipe's ruling is that the review always wins, because it is the later and
# closer look. That makes this resolution a pure function of the reviewed row — no
# cross-file join — and it costs nothing, because the sweep's labels stay untouched in
# ImageData_total.csv, which is where anchor_candidates.py reads `human` frames to
# propose clock anchors from install and retrieval photos.

# Spanish negation or non-animal category -> the Camtrap DP type it really is. The type
# spellings come from camtrap/exports.py, which owns that vocabulary; restating them
# here as literals would put one decision in two places.
COMMENT_TO_TYPE: dict[str, str] = {
    "no es un animal": exports.TYPE_BLANK,
    "no reconocible":  exports.TYPE_UNKNOWN,
    "humano":          exports.TYPE_HUMAN,
    "vehiculo":        exports.TYPE_VEHICLE,
}

# Comments too coarse to be a species. DECIDED 2026-08-19 (Felipe): these stay
# `unknown`, because a specific species cannot be identified from them. `ave` is a class
# and `roedor` an order, and Camtrap DP's scientificName would accept a higher rank, but
# recording one would assert more than the reviewer saw. `churrete` is ruled `unknown`
# too — Cinclodes is a genus of several species here and the comment does not pick one.
#
# Two comments that were in this set are NOT here any more, having been adjudicated as
# identifiable animals and added to species.yaml, which is where species decisions live:
# `conejo` -> Oryctolagus cuniculus and `pitío` -> Colaptes pitius. They now resolve
# through R3 like any other Spanish common name.
COARSE_COMMENTS = frozenset({
    "ave", "roedor", "churrete",
})

# Notes to self left in place of an identification, reviewed 2026-08-19 and also ruled
# `unknown`. Still tagged apart from the coarse-taxon rows and from the 499 plain "could
# not tell" ones, because the three are different kinds of not-knowing and a later
# question about any one of them should be answerable without re-reading the CSVs.
REVIEW_NOTE_COMMENTS = frozenset({
    "identificar", "no reconocible pero identificar", "error de imagen",
})

# Values of the `review_resolution` column. They name WHICH RULE fired, so a consumer
# can ask why a row says what it says without reopening the reviewed CSV.
RESOLUTION_SPECIES_NAMED        = "species_named"
RESOLUTION_TYPE_FROM_COMMENT    = "type_from_comment"
RESOLUTION_SPECIES_FROM_COMMENT = "species_from_comment"
RESOLUTION_COARSE_COMMENT       = "unknown_coarse_comment"
RESOLUTION_REVIEW_NOTE          = "unknown_review_note"
# The row was never reviewed — it is a still the sweep categorised and the species
# review never opened. Its type is the sweep's and its species is empty, by definition.
RESOLUTION_SWEEP_ONLY           = "sweep_only"


class UnmappedReviewComment(ValueError):
    """A reviewer comment no rule covers. Names every one of them, with counts.

    Fail-closed on purpose, and it aggregates rather than dying on the first: guessing
    would put a fabricated verdict in the canonical table, and raising one comment at a
    time would make a ten-comment backlog take ten runs to discover.
    """


def _column(reviewed: pd.DataFrame, name: str) -> pd.Series:
    """One review column as text, always on `reviewed`'s index.

    A missing column yields empty strings rather than an empty Series: a short Series
    would silently misalign every subsequent .isin() and .loc[] against a frame of a
    different length.
    """
    if name not in reviewed.columns:
        return pd.Series("", index=reviewed.index, dtype=object)
    return reviewed[name].fillna("").astype(str).str.strip()


def _normalise(reviewed: pd.DataFrame) -> pd.Series:
    """Comment text as the rule tables key it: stripped and lowercased.

    Accents are NOT folded. species.yaml already lists accented and unaccented spellings
    side by side, so folding here would second-guess that catalogue; an unlisted accented
    variant is caught by UnmappedReviewComment instead of being silently mapped.
    """
    return _column(reviewed, "observationComments").str.lower()


def audit_review_comments(reviewed: pd.DataFrame) -> dict[str, int]:
    """Comments on empty-species rows that no rule covers, with row counts.

    Empty dict means resolve_review() will not raise. Separate from resolve_review so a
    caller can survey a reviewed CSV without having to catch an exception to do it.
    """
    latin = _column(reviewed, "scientificName")
    comments = _normalise(reviewed)
    known = set(COMMENT_TO_TYPE) | set(spanish_to_latin())
    known |= COARSE_COMMENTS | REVIEW_NOTE_COMMENTS
    unmapped = comments[(latin == "") & ~comments.isin(known)]
    return {k: int(v) for k, v in unmapped.value_counts().items()}


def resolve_review(reviewed: pd.DataFrame) -> pd.DataFrame:
    """Resolve each reviewed row into its canonical type, species and rule tag.

    Returns a frame indexed like `reviewed` with columns `observation_type`,
    `species_latin` and `review_resolution`. Raises UnmappedReviewComment if any
    empty-species row carries a comment no rule covers.
    """
    unmapped = audit_review_comments(reviewed)
    if unmapped:
        listed = "\n".join(
            f"    {comment or '(empty comment)'!r:<36} {n:>5} row(s)"
            for comment, n in sorted(unmapped.items(), key=lambda kv: -kv[1])
        )
        raise UnmappedReviewComment(
            f"{sum(unmapped.values())} reviewed row(s) have an empty scientificName "
            f"and a comment no rule covers:\n{listed}\n"
            f"  Every one needs a decision before ingest — a species belongs in "
            f"species.yaml, a negation in COMMENT_TO_TYPE, a coarse taxon or an "
            f"open question in COARSE_COMMENTS (camtrap/observations.py).\n"
            f"  An empty comment on an empty species means the row was never actually "
            f"reviewed; that is `unclassified`, not `unknown`, and no rule here can "
            f"invent a verdict for it."
        )

    latin = _column(reviewed, "scientificName")
    comments = _normalise(reviewed)
    from_catalogue = comments.map(spanish_to_latin())

    out = pd.DataFrame(index=reviewed.index)
    # Built in reverse precedence: each rule overwrites the weaker ones and R1 lands
    # last, which keeps the ranking readable as a sequence rather than nested where().
    # Every rule assigns its own tag — nothing is left to the initial value, so a row
    # the rules failed to cover shows up as '' in the assertion below instead of
    # inheriting a plausible-looking verdict.
    out["observation_type"] = exports.TYPE_UNKNOWN
    out["species_latin"] = ""
    out["review_resolution"] = ""

    note = comments.isin(REVIEW_NOTE_COMMENTS)
    out.loc[note, "review_resolution"] = RESOLUTION_REVIEW_NOTE

    taxon = comments.isin(COARSE_COMMENTS)
    out.loc[taxon, "review_resolution"] = RESOLUTION_COARSE_COMMENT

    named_in_comment = (latin == "") & from_catalogue.notna()
    out.loc[named_in_comment, "observation_type"] = exports.TYPE_ANIMAL
    out.loc[named_in_comment, "species_latin"] = from_catalogue[named_in_comment]
    out.loc[named_in_comment, "review_resolution"] = RESOLUTION_SPECIES_FROM_COMMENT

    typed = (latin == "") & comments.isin(COMMENT_TO_TYPE)
    out.loc[typed, "observation_type"] = comments[typed].map(COMMENT_TO_TYPE)
    out.loc[typed, "species_latin"] = ""
    out.loc[typed, "review_resolution"] = RESOLUTION_TYPE_FROM_COMMENT

    named = latin != ""
    out.loc[named, "observation_type"] = exports.TYPE_ANIMAL
    out.loc[named, "species_latin"] = latin[named]
    out.loc[named, "review_resolution"] = RESOLUTION_SPECIES_NAMED

    # audit_review_comments() above should make this unreachable. It is checked anyway
    # because the alternative to an assertion here is a row reaching the canonical table
    # with a verdict no rule produced, which is the failure this whole module exists to
    # prevent.
    uncovered = int((out["review_resolution"] == "").sum())
    if uncovered:
        raise UnmappedReviewComment(
            f"{uncovered} row(s) matched no resolution rule despite passing the comment "
            f"audit — the rule tables and audit_review_comments() have drifted apart"
        )
    return out


# =============================================================================
# Which rows the canonical table describes
# =============================================================================
#
# EVERY STILL IN THE GATED EXPORT — not just the ones the species review opened.
# Changed 2026-08-19; before that the row set was the reviewed CSV, and the cost was
# that a station which recorded no animal vanished from the table entirely. Seven
# station-campaigns were missing: CT23 in otoño 2025, CT01/CT06/CT17/CT22 in primavera
# (6, 21, 7 and 18 frames each, none with an animal), CT02/CT12 in otoño 2026. A station
# absent from the table is indistinguishable from a station that was never deployed,
# which makes it fine as a detection numerator and wrong as a trap-effort denominator —
# and this module's own docstring already promised otherwise.
#
# The review supplies the verdict where one exists; the sweep supplies it where none
# does. That is not a reversal of the rule that the review beats the sweep: the sweep is
# the DEFAULT for rows the review never saw, never a competitor to a row it did.

# Columns taken from the review, not from the sweep. They are dropped from the export
# side before the merge (see compose_ingest_frame) because the all-images export carries
# them as EMPTY columns of the same CamtrapDP schema — letting those win is what made
# every reviewed row arrive looking unreviewed.
REVIEW_COLUMNS = [
    "scientificName", "observationComments", "reviewOutcome", "classificationProbability",
]
REVIEWED_FLAG = "_reviewed"


class ReviewedRowNotInExport(ValueError):
    """A reviewed row has no counterpart in the all-images export.

    Means the export is not, in fact, all the images — the one thing the export gate
    exists to guarantee — so the clock diagnosis ran on a different set of frames than
    the rows being written.
    """


def compose_ingest_frame(
    total: pd.DataFrame,
    reviewed: pd.DataFrame,
    campaign: str,
) -> pd.DataFrame:
    """Every still in `total`, with `reviewed`'s verdict attached where one exists.

    Adds `_reviewed`. The returned frame keeps `total`'s columns, so the caller hands it
    to the same clock repair as before — the repair works per (camera, file) and does not
    care whether a row was reviewed.
    """
    key = ["_camera_num", "_file_name"]

    def keyed(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["_camera_num"] = [
            stations.resolve(str(s), campaign) for s in out["Deployments"]
        ]
        out["_file_name"] = out["File"].astype(str).str.strip()
        return out

    left = keyed(total)
    right = keyed(reviewed)

    # The review's own columns only. Everything else about a row comes from the export,
    # so the two files cannot disagree about a frame's path, station or stamp.
    present = [c for c in REVIEW_COLUMNS if c in right.columns]
    right = right[key + present].drop_duplicates(subset=key, keep="first")

    # The all-images export carries these column NAMES too, empty, because it is the same
    # CamtrapDP schema. Dropping them from the export side is what makes the review the
    # only source of a verdict — leave them in and the merge suffixes the review's values
    # to `scientificName_rev`, the empty export column wins, and every reviewed row
    # arrives looking unreviewed.
    left = left.drop(columns=[c for c in present if c in left.columns])

    merged = left.merge(right, on=key, how="left", indicator=True)

    # Anti-join in the other direction. This check used to live in
    # timestamps.repair_campaign, where it could still fire because the reviewed rows
    # were the frame being merged; now that the export is the frame, it has to be asked
    # here or it stops being asked at all.
    orphans = right.merge(left[key], on=key, how="left", indicator=True)
    orphans = orphans[orphans["_merge"] == "left_only"]
    if len(orphans):
        examples = ", ".join(orphans["_file_name"].head(5))
        raise ReviewedRowNotInExport(
            f"{campaign}: {len(orphans)} reviewed row(s) do not appear in "
            f"{exports.TOTAL_EXPORT_FILENAME}. Examples: {examples}. Re-export all "
            f"images from the same Timelapse2 project the review was done in."
        )

    merged[REVIEWED_FLAG] = merged["_merge"] == "both"
    for col in REVIEW_COLUMNS:
        if col not in merged.columns:
            merged[col] = ""
        merged[col] = merged[col].fillna("")
    return merged.drop(columns=["_merge"])


def resolve_observation(frame: pd.DataFrame) -> pd.DataFrame:
    """Resolve type, species and rule tag for reviewed AND un-reviewed rows.

    Reviewed rows go through resolve_review()'s verdict rules. Un-reviewed rows take the
    sweep's `observationType` verbatim with an empty species — there is nothing else to
    take, and inventing a species for a frame nobody opened is the failure this module
    exists to prevent.
    """
    reviewed_mask = (
        frame[REVIEWED_FLAG].astype(bool)
        if REVIEWED_FLAG in frame.columns
        else pd.Series(True, index=frame.index)
    )

    out = pd.DataFrame(index=frame.index)
    out["observation_type"] = ""
    out["species_latin"] = ""
    out["review_resolution"] = ""

    if reviewed_mask.any():
        out.loc[reviewed_mask] = resolve_review(frame[reviewed_mask]).values

    unreviewed = ~reviewed_mask
    if unreviewed.any():
        sweep = (
            frame.loc[unreviewed, exports.OBSERVATION_TYPE_COLUMN]
            .fillna("").astype(str).str.strip().str.lower()
        )
        # `unclassified` on an un-reviewed row is the export gate's business, not this
        # module's: the gate refuses a campaign that still holds any, so reaching here
        # with one means the gate was bypassed.
        unassigned = sweep.isin(exports.UNASSIGNED_TYPES)
        if unassigned.any():
            raise UnmappedReviewComment(
                f"{int(unassigned.sum())} un-reviewed row(s) carry an unassigned "
                f"observationType ({sorted(sweep[unassigned].unique())}). The export gate "
                f"refuses a campaign in that state, so this frame did not come through it."
            )
        out.loc[unreviewed, "observation_type"] = sweep
        out.loc[unreviewed, "species_latin"] = ""
        out.loc[unreviewed, "review_resolution"] = RESOLUTION_SWEEP_ONLY

    return out


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

    for flag in ("valid_date", "valid_time_of_day", "valid_effort"):
        if flag not in src.columns:
            # Defaulting a missing flag to False would silently drop whole stations
            # out of every rate denominator; defaulting to True would silently
            # readmit fabricated dates. Neither is a safe guess.
            raise ValueError(
                f"{campaign}: corrected frame has no {flag!r} column. Re-run "
                f"`python timestamps.py --campaign {campaign}` — this column is "
                f"written by the segment-aware repair and an older _corrected.csv "
                f"predates it."
            )
        out[flag] = src[flag].astype(str).str.strip().str.lower().isin({"true", "1"})

    out["repair_method"] = src["repair_method"].astype(str).str.strip()
    out["review_outcome"] = src.get("reviewOutcome", "").astype(str).str.strip()
    out["file_name"] = src["File"].astype(str).str.strip()
    out["rel_path"] = src.apply(_resolve_rel_path, axis=1)

    # Verbatim, not normalised: `_normalise()` lowercases for RULE MATCHING, but the
    # audit value of this column is that it is what the reviewer typed. Casing and
    # accents included.
    out["observation_comments"] = (
        src.get("observationComments", "").astype(str).str.strip()
    )
    # errors='coerce' rather than a raise: a blank on a sweep-only row is correct, and a
    # malformed score is a classifier artefact that should not block an ingest.
    out["classification_probability"] = pd.to_numeric(
        src.get("classificationProbability", ""), errors="coerce"
    ).astype("Float64")

    # `observationType` from the reviewed CSV is NOT copied through: it reads `animal`
    # on every row of an animal-only export, including the rows the reviewer went on to
    # say hold no animal. resolve_review() is what decides the type.
    resolved = resolve_observation(src)
    out["observation_type"] = resolved["observation_type"]
    out["species_latin"] = resolved["species_latin"]
    out["review_resolution"] = resolved["review_resolution"]

    tally = resolved["review_resolution"].value_counts()
    for rule, n in tally.items():
        print(f"    {rule}: {n}")

    # Computed here and not by a consumer for two reasons: the rule had already drifted
    # between the two copies that existed downstream (523 events against 696), and
    # `clock_segment` is not in this table, so a consumer cannot keep an episode from
    # spanning a clock reset even if it wanted to.
    out[episodes.COLUMN] = episodes.label(out, segments=src.get("clock_segment"))
    print(f"    {episodes.COLUMN}: {out[episodes.COLUMN].nunique(dropna=True)} episode(s) "
          f"over {int(out[episodes.COLUMN].notna().sum())} row(s)")

    # Reindexed, not just cast: CANONICAL_COLUMNS declares an order and the written
    # table should match it, so the contract can be asserted as written rather than as
    # whichever order the assignments above happened to run in.
    return out[list(CANONICAL_COLUMNS)].astype(CANONICAL_COLUMNS).reset_index(drop=True)


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
    collapsed, keeping the one from the LATEST campaign in CAMPAIGN_ORDER.

    THE CURRENT THREE CAMPAIGNS DO NOT OVERLAP, measured 2026-08-25: 0 duplicate
    DEDUP_KEY rows across all 35,807. The 396-record overlap this docstring used to
    cite was primavera_2025 against pv_2025_2026, and pv was retired 2026-08-19.
    31 `(camera_num, file_name)` pairs do recur across campaigns, and NONE share a
    datetime — that is `MMDDnnnn.JPG` counter recycling between years, which the key
    separates correctly and which must NOT be deduplicated (3.5: two files with the
    same name in the same station are both kept).

    So dedup is a no-op today and stays because a partial re-ingest reintroduces the
    overlap the moment one happens. It is cheap insurance, not dead code.

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
