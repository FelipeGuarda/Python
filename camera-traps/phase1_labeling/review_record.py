"""phase1_labeling/review_record.py — the reviewer's decisions, on disk.

WHAT THIS OWNS

    How a review decision (species, outcome, co-presence tags, when it was made) is
    written into `new_labeled_data_reviewed.csv` and read back out of it. The review
    app owns the screens; this module owns the file. It imports no Streamlit, so the
    rules below are tested without a browser.

TWO WAYS TO SAVE, BECAUSE THEY PROMISE DIFFERENT THINGS

    write_review() rebuilds the file from the classifier's CSV plus the decisions. That
    is the normal review, and it is only safe while the reviewed CSV still derives from
    that classifier CSV.

    write_tags() rewrites ONLY `observationTags` in the reviewed CSV that already
    exists. It is the retro pass over campaigns reviewed before the checkboxes existed,
    and it must not be able to change a species, an outcome or a date. Measured
    2026-11-13: re-exporting otoño 2025 through write_review would have lost every
    decision (the reviewed CSV writes `CT09\\x.JPG`, the classifier CSV `CT09/x.JPG`)
    and dropped 12 reviewed images that the classifier CSV does not have. A reviewed
    CSV is a record, not a cache.

THE REVIEW DATE IS WHEN THE DECISION WAS MADE

    `classificationTimestamp` is set when the reviewer confirms, and kept on every
    later save. It used to be re-stamped on every row at every export, so reopening a
    campaign moved all its review dates to the day it was reopened.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from camtrap import copresence

REVIEWED_FILENAME = "new_labeled_data_reviewed.csv"
CLASSIFIED_BY = "CLIP zero-shot + human review"

# Timelapse2 Camtrap DP columns this module reads or writes.
_PATH, _REL, _FILE = "filePath", "RelativePath", "File"
_TYPE, _TAGS, _OUTCOME = "observationType", "observationTags", "reviewOutcome"
_SPECIES, _LATIN = "observationComments", "scientificName"
_BY, _WHEN = "classifiedBy", "classificationTimestamp"

_BOM = "﻿"


class ReviewRecordError(RuntimeError):
    """A save that would lose or corrupt reviewed work. Nothing was written."""


@dataclass(frozen=True)
class Decision:
    """One reviewed image, as the reviewed CSV records it."""
    species: str          # Spanish name, as the app shows it
    outcome: str          # "confirmed" | "corrected"
    tags: str = ""        # an observationTags value; build it with copresence.join()
    timestamp: str = ""   # when the decision was made; kept as-is on later saves
    path: str = ""        # the image's relative path, "/"-separated, original case


def now() -> str:
    """A review date in the format the reviewed CSV has always used."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def image_path(row: dict) -> str:
    """The image's relative path, "/"-separated. `filePath` when Timelapse filled it,
    otherwise RelativePath + File (campaigns where it did not)."""
    fp = (row.get(_PATH) or "").strip()
    if not fp:
        rel, name = (row.get(_REL) or "").strip(), (row.get(_FILE) or "").strip()
        fp = f"{rel}/{name}" if rel and name else ""
    return fp.replace("\\", "/")


def row_key(row: dict) -> str:
    """The identity of an image across the classifier CSV, the reviewed CSV and the
    MegaDetector JSON: its path, "/"-separated and lowercased. '' when it has none."""
    return image_path(row).lower()


def read_decisions(campaign_dir: Path) -> dict[str, Decision]:
    """Every decided image in the campaign's reviewed CSV, by row_key. {} if none yet."""
    path = Path(campaign_dir) / REVIEWED_FILENAME
    if not path.exists():
        return {}
    _, header, rows = _read(path)
    out: dict[str, Decision] = {}
    for values in rows:
        row = dict(zip(header, values))
        outcome, key = (row.get(_OUTCOME) or "").strip(), row_key(row)
        if outcome and key:
            out[key] = Decision(
                species=(row.get(_SPECIES) or "").strip(),
                outcome=outcome,
                tags=copresence.join(copresence.parse(row.get(_TAGS) or "")),
                timestamp=(row.get(_WHEN) or "").strip(),
                path=image_path(row),
            )
    return out


def write_review(
    rows: list[dict],
    fieldnames: list[str],
    decisions: dict[str, Decision],
    spanish_to_latin: dict[str, str],
    campaign_dir: Path,
) -> Path:
    """Normal review: the classifier's rows with every decision applied, written to
    the reviewed CSV. Refuses, writing nothing, if an image already decided in that
    file would be missing from the new one, or if a tag is not allowed on its row."""
    out_path = Path(campaign_dir) / REVIEWED_FILENAME
    keys = {row_key(r) for r in rows}
    lost = sorted(set(read_decisions(campaign_dir)) - keys)
    if lost:
        raise ReviewRecordError(
            f"Saving would drop {len(lost)} image(s) already decided in {out_path.name} "
            f"(e.g. {lost[0]}), because the classifier CSV does not have them. That "
            f"reviewed CSV does not derive from this classifier CSV; to add tags to it, "
            f"use the 'Etiquetas' mode, which never rebuilds the file."
        )

    out_rows = []
    for row in rows:
        row = dict(row)
        d = decisions.get(row_key(row))
        if d is not None:
            row[_SPECIES] = d.species
            row[_LATIN] = spanish_to_latin.get(d.species, "")
            row[_BY] = CLASSIFIED_BY
            row[_WHEN] = d.timestamp
            row[_OUTCOME] = d.outcome
            row[_TAGS] = d.tags
        out_rows.append(row)

    columns = list(fieldnames)
    for col in (_TAGS, _OUTCOME):
        if col not in columns:
            columns.append(col)
    _check_tags([r.get(_TAGS, "") for r in out_rows],
                [r.get(_TYPE, "") for r in out_rows], Path(campaign_dir).name)

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(out_rows)
    return out_path


def write_tags(campaign_dir: Path, tags: dict[str, str]) -> Path:
    """Retro pass: set `observationTags` for the images in `tags` (row_key -> value)
    in the existing reviewed CSV. Every other cell, the row order, the encoding and the
    line endings are written back exactly as read. Refuses, writing nothing, if an
    image is not in the file or a tag is not allowed on its row."""
    path = Path(campaign_dir) / REVIEWED_FILENAME
    if not path.exists():
        raise ReviewRecordError(f"{path} does not exist: there is nothing to tag yet.")
    fmt, header, rows = _read(path)
    if _TAGS not in header:
        raise ReviewRecordError(f"{path.name} has no {_TAGS} column.")
    i_tags, i_type = header.index(_TAGS), header.index(_TYPE)

    found = set()
    for values in rows:
        key = row_key(dict(zip(header, values)))
        if key in tags:
            values[i_tags] = tags[key]
            found.add(key)
    missing = sorted(set(tags) - found)
    if missing:
        raise ReviewRecordError(
            f"{len(missing)} image(s) are not in {path.name} (e.g. {missing[0]}). Nothing saved.")
    _check_tags([v[i_tags] for v in rows], [v[i_type] for v in rows], Path(campaign_dir).name)

    path.write_bytes(_render(fmt, header, rows))
    return path


# ── internals ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Format:
    bom: bool
    newline: str


def _read(path: Path) -> tuple[_Format, list[str], list[list[str]]]:
    raw = path.read_bytes().decode("utf-8")
    bom = raw.startswith(_BOM)
    text = raw[len(_BOM):] if bom else raw
    # The file's own line ending, so a retro save does not turn a whole file over:
    # the app writes CRLF on Windows and git checks it out as LF here.
    newline = "\r\n" if "\r\n" in text else "\n"
    records = list(csv.reader(io.StringIO(text, newline="")))
    return _Format(bom, newline), records[0], records[1:]


def _render(fmt: _Format, header: list[str], rows: list[list[str]]) -> bytes:
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator=fmt.newline)
    writer.writerow(header)
    writer.writerows(rows)
    return ((_BOM if fmt.bom else "") + buf.getvalue()).encode("utf-8")


def _check_tags(tags: list[str], kinds: list[str], campaign: str) -> None:
    """The ingest's own validation, run before writing, so the app cannot save a tag
    that timestamps.py would then refuse."""
    n = len(tags)
    try:
        copresence.resolve(
            pd.Series(tags, dtype="object"), pd.Series(kinds, dtype="object"),
            pd.Series([""] * n, dtype="object"), pd.Series([True] * n), campaign,
        )
    except copresence.InvalidObservationTag as exc:
        raise ReviewRecordError(f"Nothing saved. {exc}") from exc
