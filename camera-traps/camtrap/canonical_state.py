"""What the canonical tables currently ARE, as a fact a consumer can check.

Four projects read `observations.parquet` — the 2025 annual report, pehuén, the DuckDB
`ct_*` rebuild and the territorial platform — in three languages, so no shared library
can serve them. What they can all read is a small JSON file next to the data.

This module owns one decision: **what counts as the published state of the canonical
tables, and what a consumer must check before trusting them.** Not how to read a
parquet, not what an observation means — only the contract's content.

Why it exists at all, and why the check is fail-closed: on 2026-08-19 the canonical
tables went from 3,359 rows to 35,807 (one row per still, not one per reviewed row).
Every consumer kept running. `01_data_prep.py` happened to filter on `observation_type`
and stayed correct; nothing verified that, and a consumer that did not filter would have
counted 32,448 blank stills as observations without a single error. A contract nobody
verifies is a comment.

Usage
-----
    python -m camtrap.canonical_state --publish     # after re-ingesting any campaign
    python -m camtrap.canonical_state              # verify, exit 1 on divergence

From a consumer (Python):
    state = canonical_state.load()
    canonical_state.assert_columns(df, state)

From a consumer in R or SQL: read the JSON, compare `columns` against your frame's
names and `campaigns[<name>].n_rows` against your row count, and stop if they differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from camtrap.deployments import DEPLOYMENTS_FILENAME
from camtrap.observations import (
    CAMPAIGNS_ROOT,
    CANONICAL_COLUMNS,
    CANONICAL_FILENAME,
)

# Bump when a column is added, removed, renamed or retyped, or when the SHAPE of this
# description changes. Consumers compare this first: a mismatch means "your code was
# written against a different table", which is a clearer thing to report than a
# missing-column KeyError several frames later.
#
#   2 -> 3 (2026-08-24): each campaign now also describes its published deployment
#   windows. Effort is a DENOMINATOR -- a wrong row count is visible because a species
#   appears or does not, while a wrong denominator silently rescales every rate in a
#   report and nothing looks broken. It therefore belongs inside the thing consumers
#   verify, not beside it.
SCHEMA_VERSION = 3

STATE_FILENAME = "CANONICAL_STATE.json"
DEFAULT_STATE_PATH = CAMPAIGNS_ROOT.parent / STATE_FILENAME

# The campaigns whose parquets are part of the published state. Deliberately explicit
# rather than "every directory found on disk": a retired directory left in place for
# provenance must not silently become part of the contract, which is exactly how
# `pv_2025_2026` came to outrank primavera in CAMPAIGN_ORDER.
PUBLISHED_CAMPAIGNS = ("otono_2025", "primavera_2025", "otono_2026")


class CanonicalStateError(RuntimeError):
    """The canonical tables do not match their published state."""


def _sha256(path: Path) -> str | None:
    """None, not a crash, when the file is absent: a campaign published before the
    deployment windows existed is a legitimate state to describe."""
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _describe_deployments(campaign: str, root: Path) -> dict:
    """The published effort for one campaign, summarised so the gate can check it.

    `n_deployments` counts every station the field record dates, including those with
    no images; `camera_days` sums only the ones that have images, because that is the
    number a detection rate may legitimately divide by.
    """
    path = root / campaign / DEPLOYMENTS_FILENAME
    if not path.exists():
        return {"n_deployments": None, "n_deployments_with_media": None,
                "camera_days": None, "deployments_sha256": None}
    dep = pd.read_csv(path)
    with_media = dep[dep["has_media"].astype(bool)]
    return {
        "n_deployments": int(len(dep)),
        "n_deployments_with_media": int(len(with_media)),
        "camera_days": int(with_media["field_days"].dropna().sum()),
        "deployments_sha256": _sha256(path),
    }


def _describe(campaign: str, root: Path) -> dict:
    df = pd.read_parquet(root / campaign / CANONICAL_FILENAME)
    dt = pd.to_datetime(df["datetime"], errors="coerce")
    reviewed = df["review_resolution"] != "sweep_only"
    return {
        "n_rows": int(len(df)),
        "n_stations": int(df["camera_num"].nunique()),
        "stations": sorted(df["station_canonical"].dropna().unique().tolist()),
        "n_reviewed": int(reviewed.sum()),
        "n_sweep_only": int((~reviewed).sum()),
        "observation_types": {
            k: int(v) for k, v in df["observation_type"].value_counts().sort_index().items()
        },
        "n_animal_rows": int((df["observation_type"] == "animal").sum()),
        "n_species": int(df.loc[df["species_latin"] != "", "species_latin"].nunique()),
        # NaT-safe: a campaign whose every clock is unrepairable has no date range, and
        # that is a legitimate state to publish rather than a crash.
        "datetime_min": None if dt.isna().all() else str(dt.min()),
        "datetime_max": None if dt.isna().all() else str(dt.max()),
        "n_valid_date_false": int((~df["valid_date"].fillna(False)).sum()),
        "n_valid_effort_false": int((~df["valid_effort"].fillna(False)).sum()),
        **_describe_deployments(campaign, root),
    }


def build(*, root: Path = CAMPAIGNS_ROOT) -> dict:
    """The current state of the canonical tables, read from disk."""
    campaigns = {c: _describe(c, root) for c in PUBLISHED_CAMPAIGNS}
    return {
        "schema_version": SCHEMA_VERSION,
        "columns": list(CANONICAL_COLUMNS),
        "dtypes": dict(CANONICAL_COLUMNS),
        "campaigns": campaigns,
        "n_rows_total": sum(c["n_rows"] for c in campaigns.values()),
        # Union, not sum: campaigns share stations, and the count that matters to a
        # consumer is how many distinct cameras the whole dataset covers.
        "n_stations_total": len({s for c in campaigns.values() for s in c["stations"]}),
    }


def publish(*, root: Path = CAMPAIGNS_ROOT, out: Path = DEFAULT_STATE_PATH) -> dict:
    """Write the state file. Run after re-ingesting ANY campaign."""
    state = build(root=root)
    out.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return state


def load(path: Path = DEFAULT_STATE_PATH) -> dict:
    if not path.exists():
        raise CanonicalStateError(
            f"{path} not found. Run `python -m camtrap.canonical_state --publish` "
            f"after re-ingesting the campaigns."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def diff(published: dict, current: dict) -> list[str]:
    """Human-readable differences. Empty list means they agree."""
    out: list[str] = []
    if published.get("schema_version") != current["schema_version"]:
        out.append(
            f"schema_version: published {published.get('schema_version')} "
            f"!= current {current['schema_version']}"
        )
    if published.get("columns") != current["columns"]:
        pub, cur = set(published.get("columns", [])), set(current["columns"])
        if pub - cur:
            out.append(f"columns removed since publish: {sorted(pub - cur)}")
        if cur - pub:
            out.append(f"columns added since publish: {sorted(cur - pub)}")
        if pub == cur:
            out.append("column ORDER changed (the contract declares an order)")
    for name in PUBLISHED_CAMPAIGNS:
        p, c = published.get("campaigns", {}).get(name), current["campaigns"][name]
        if p is None:
            out.append(f"{name}: absent from the published state")
            continue
        for key in ("n_rows", "n_stations", "n_reviewed", "n_animal_rows"):
            if p.get(key) != c[key]:
                out.append(f"{name}.{key}: published {p.get(key)} != current {c[key]}")
    return out


def verify(*, root: Path = CAMPAIGNS_ROOT, path: Path = DEFAULT_STATE_PATH) -> dict:
    """Raise unless the parquets on disk match the published state."""
    published = load(path)
    problems = diff(published, build(root=root))
    if problems:
        raise CanonicalStateError(
            "The canonical tables have changed since CANONICAL_STATE.json was published:\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\n\nEither re-publish (`python -m camtrap.canonical_state --publish`) if the "
              "change is intended, or find out who rewrote the parquets. Consumers of this "
              "table read the published numbers; a silent change is how a rebuild reaches a "
              "report without anyone noticing."
        )
    return published


def assert_columns(frame: pd.DataFrame, state: dict | None = None) -> None:
    """For a Python consumer: refuse a frame that is not the contracted shape.

    Checks names, not order — a consumer that selects columns by name is unaffected by
    order, and holding it to the order would fail loudly for no benefit. `verify()` is
    what guards order, because the WRITER is the one that owes it.
    """
    state = state or load()
    missing = [c for c in state["columns"] if c not in frame.columns]
    if missing:
        raise CanonicalStateError(
            f"Frame is missing canonical column(s) {missing}. Expected schema_version "
            f"{state['schema_version']} with columns {state['columns']}. If this frame "
            f"came from observations.parquet, the file predates the current contract — "
            f"re-run `python timestamps.py --campaign <name>` for every campaign."
        )


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--publish", action="store_true",
                    help="Write CANONICAL_STATE.json from the parquets on disk")
    args = ap.parse_args(argv)

    if args.publish:
        state = publish()
        print(f"Published {DEFAULT_STATE_PATH}")
        print(f"  schema_version {state['schema_version']}, "
              f"{len(state['columns'])} columns")
        for name, c in state["campaigns"].items():
            print(f"  {name:16s} {c['n_rows']:>6,} rows  {c['n_stations']:>2} stations  "
                  f"{c['n_reviewed']:>5,} reviewed  {c['n_animal_rows']:>5,} animal")
        print(f"  TOTAL            {state['n_rows_total']:>6,} rows  "
              f"{state['n_stations_total']:>2} stations")
        return 0

    try:
        state = verify()
    except CanonicalStateError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"OK — canonical tables match CANONICAL_STATE.json "
          f"(schema_version {state['schema_version']}, "
          f"{state['n_rows_total']:,} rows, {state['n_stations_total']} stations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
