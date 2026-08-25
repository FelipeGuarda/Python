"""The consumer half of the canonical contract: is this database current with what
camera-traps published?

THE DIRECTION IS THE DESIGN AND IS NEVER REVERSED. camera-traps publishes; data-pipeline
verifies. camera-traps must not learn that DuckDB exists -- so this module reads the
published JSON as a FILE and does not import `camtrap`. If it imported the producer, the
check would be running the producer's own code against the producer's own data and could
only ever agree with itself.

WHY IT EXISTS. On 2026-08-19 the canonical tables went from 3,359 rows to 35,807 and not
one consumer raised an error. `camtrap/canonical_state.py` closed that on the producer
side; until today `grep -r CANONICAL_STATE data-pipeline/` returned nothing, so the
contract was published and unread. A contract nobody verifies is a comment.

FAIL-CLOSED. A missing, unreadable or unparseable state file means REFUSE, not proceed --
the same posture as the flatten preconditions, and for the same reason: the failure being
prevented is a silent one. Proceeding on a missing contract is exactly the case where the
database quietly keeps serving last month's numbers.

WHAT THIS DOES NOT CHECK. Whether the published state matches the parquets on disk. That
is the producer's job (`camtrap.canonical_state.verify`) and duplicating it here would be
the second place a repair has to reach. This module answers one narrower question: has
this database ingested the state that is currently published?
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import duckdb

_ENV_VAR = "FMA_CANONICAL_STATE"

# data-pipeline/src/canonical_gate.py -> parents[2] = monorepo root
_DEFAULT_PATH = (
    Path(__file__).resolve().parents[2] / "camera-traps" / "data" / "CANONICAL_STATE.json"
)

#: Bump when this module learns to read a new contract shape. Refusing an unknown
#: schema_version is the point: a producer that changed shape must be read deliberately.
#:
#: 3 (2026-08-24) adds each campaign's published deployment windows -- n_deployments,
#: camera_days and a hash of deployments.csv. Nothing here had to change to CHECK them:
#: fingerprint() hashes the whole campaign description, so a hand-edited deployments.csv
#: moves its sha256, moves the fingerprint, and the gate reports the database stale.
SUPPORTED_SCHEMA_VERSION = 3


class CanonicalGateError(RuntimeError):
    """The contract is missing, unreadable, or the database is not current with it."""


def state_path() -> Path:
    override = os.getenv(_ENV_VAR)
    return Path(override) if override else _DEFAULT_PATH


def load_published(path: Path | None = None) -> dict:
    """The published contract. Raises rather than returning a default."""
    path = path or state_path()
    if not path.exists():
        raise CanonicalGateError(
            f"{path} not found. camera-traps publishes it with "
            f"`python -m camtrap.canonical_state --publish` after re-ingesting a "
            f"campaign. Refusing to ingest against an unknown contract."
        )
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CanonicalGateError(f"{path} is not valid JSON: {exc}") from exc

    version = state.get("schema_version")
    if version != SUPPORTED_SCHEMA_VERSION:
        raise CanonicalGateError(
            f"{path} declares schema_version {version}; this pipeline was written "
            f"against {SUPPORTED_SCHEMA_VERSION}. The canonical table changed shape. "
            f"Read camera-traps/camtrap/observations.py (CANONICAL_COLUMNS) and update "
            f"src/parsers/canonical_ct.py deliberately — do not just bump the number."
        )
    if not state.get("campaigns"):
        raise CanonicalGateError(f"{path} declares no campaigns.")
    return state


def fingerprint(declared: dict) -> str:
    """A hash of everything the contract says about one campaign.

    Row counts alone are too coarse. The 815-row review repair (V2-REVIEW 1.3) moved
    `observation_types` and `n_animal_rows` while leaving `n_rows` untouched — exactly
    the change a row-count check cannot see, and exactly the change a consumer most
    needs to notice. Hashing the whole description catches it.
    """
    return hashlib.sha256(
        json.dumps(declared, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def recorded(con: duckdb.DuckDBPyConnection) -> dict[str, dict]:
    """What this database last ingested, per campaign. Empty dict if never."""
    rows = con.execute(
        "SELECT campaign, n_rows, n_stations, parquet_hash, ingested_at "
        "FROM ct_ingest_state"
    ).fetchall()
    return {
        r[0]: {"n_rows": r[1], "n_stations": r[2], "fingerprint": r[3],
               "ingested_at": r[4]}
        for r in rows
    }


def check(con: duckdb.DuckDBPyConnection, path: Path | None = None) -> list[str]:
    """Differences between the published contract and this database. Empty = current.

    Returns findings instead of raising, because the caller decides: `--check` prints
    them, an ingest run acts on them. Every finding names both numbers, since "they
    differ" without the direction does not say which way to fix it.
    """
    published = load_published(path)
    have = recorded(con)
    findings: list[str] = []

    for name, declared in published["campaigns"].items():
        mine = have.get(name)
        if mine is None:
            findings.append(f"{name}: never ingested (published {declared['n_rows']} rows)")
            continue
        if mine["n_rows"] != declared["n_rows"]:
            findings.append(
                f"{name}.n_rows: database {mine['n_rows']} != published {declared['n_rows']}"
            )
        if mine["n_stations"] != declared["n_stations"]:
            findings.append(
                f"{name}.n_stations: database {mine['n_stations']} "
                f"!= published {declared['n_stations']}"
            )
        if mine["fingerprint"] != fingerprint(declared):
            findings.append(
                f"{name}: the published description changed since ingest even though "
                f"the row and station counts still agree — species, observation types, "
                f"the date range or the deployment windows moved. Rebuild."
            )

    for name in set(have) - set(published["campaigns"]):
        findings.append(
            f"{name}: present in the database but NOT published — a retired campaign "
            f"still being served. `pv_2025_2026` was exactly this."
        )

    # The row totals can agree per campaign while the tables themselves are empty,
    # if ct_ingest_state was written and the load then failed.
    if not findings:
        actual = con.execute("SELECT COUNT(*) FROM ct_observations").fetchone()[0]
        expected = sum(c["n_rows"] for c in published["campaigns"].values())
        if actual != expected:
            findings.append(
                f"ct_observations holds {actual} rows but ct_ingest_state claims "
                f"{expected}. The state table and the data disagree."
            )

    return findings


def record(con: duckdb.DuckDBPyConnection, published: dict) -> None:
    """Stamp what was just ingested. Call ONLY after the load succeeded.

    Written last on purpose: if the load raises, the state is not stamped and the next
    `check` still reports the campaign as stale. Stamping first would let a failed
    ingest look successful forever.
    """
    now = datetime.now(timezone.utc)
    con.execute("DELETE FROM ct_ingest_state")
    for name, declared in published["campaigns"].items():
        con.execute(
            "INSERT INTO ct_ingest_state "
            "(campaign, n_rows, n_stations, parquet_hash, ingested_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [name, declared["n_rows"], declared["n_stations"],
             fingerprint(declared), now],
        )


def assert_current(con: duckdb.DuckDBPyConnection, path: Path | None = None) -> None:
    """Raise unless the database is current with the published contract."""
    findings = check(con, path)
    if findings:
        raise CanonicalGateError(
            "This database is not current with the canonical tables camera-traps "
            "published:\n" + "\n".join(f"  - {f}" for f in findings)
            + "\n\nRun `python run_fetch.py --ct` to rebuild."
        )
