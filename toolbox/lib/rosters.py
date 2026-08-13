"""Contact-identity reconciliation across event workbooks.

This module owns one question: *are these two spreadsheet rows the same
person?* That covers which column holds the address, how an address is
canonicalised before comparison, how repeat rows collapse, and what counts
as a conflict versus newly-supplied information.

Nothing outside this module should know what a contact column is called or
how an address is normalised. Callers work with `Roster` objects.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import pandas as pd

# Internal column holding the normalised address. Prefixed so it cannot
# collide with a real header coming out of a workbook.
KEY = "_roster_key"

# Header names seen across FMA event workbooks, accent-stripped and
# lowercased. A column named here outranks one that merely happens to hold
# email-shaped text.
KEY_COLUMN_HINTS = frozenset({
    "email", "e-mail", "emails", "mail", "correo", "correo electronico",
    "correos electronicos", "correo(s) electronico(s)", "email principal",
    "email(s)", "contacto", "contact", "direct email", "direccion de correo",
    "direccion de correo electronico",
})

# Addresses are extracted rather than full-matched: cells are hand-typed and
# routinely arrive as "Juan Pérez <juan@x.cl>" or "a@x.cl; b@y.cl".
_EMAIL_RE = re.compile(r"[^@\s,;<>]+@[^@\s,;<>]+\.[a-zA-Z]{2,}")

# Form exports (Google Forms, SurveyMonkey) often carry a title banner above
# the real header row, so the header is searched for rather than assumed.
_HEADER_SEARCH_DEPTH = 8


def extract_emails(value: object) -> list[str]:
    """Every address in a cell, normalised, in the order written.

    Contact sheets routinely pack a person's addresses into one cell
    ("a@x.cl; b@y.cl"). They are aliases of one person, so identity checks
    must consider all of them — matching on only the first re-adds people
    who are already on the list under their other address.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = unicodedata.normalize("NFC", str(value)).strip()
    seen, found = set(), []
    for match in _EMAIL_RE.findall(text):
        address = match.lower()
        if address not in seen:
            seen.add(address)
            found.append(address)
    return found


def normalize_email(value: object) -> str:
    """The cell's primary address — the first one written — or "".

    Lowercased whole: RFC 5321 permits a case-sensitive local part, but no
    provider FMA corresponds with honours that, and these addresses are typed
    by hand into spreadsheets.
    """
    found = extract_emails(value)
    return found[0] if found else ""


def _norm_name(value: object) -> str:
    """Header name reduced for comparison: accent-free, lowercase, single-spaced."""
    text = unicodedata.normalize("NFKD", str(value)).strip().lower()
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text)


def _cell_text(value: object) -> str:
    """Cell reduced to comparable text. Blank-ish values all become ""."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value)).strip())


def _clean_headers(row: pd.Series) -> list[str]:
    """Header row to unique, non-empty column names."""
    names: list[str] = []
    seen: dict[str, int] = {}
    for position, value in enumerate(row):
        name = _cell_text(value) or f"column_{position + 1}"
        count = seen.get(name, 0)
        seen[name] = count + 1
        names.append(name if count == 0 else f"{name}.{count}")
    return names


def _find_key_column(body: pd.DataFrame, explicit_key: str | None) -> str | None:
    if explicit_key is not None:
        wanted = _norm_name(explicit_key)
        for column in body.columns:
            if _norm_name(column) == wanted:
                return column if (body[column].map(normalize_email) != "").any() else None
        return None

    best, best_score = None, 0
    for column in body.columns:
        score = int((body[column].map(normalize_email) != "").sum())
        if score == 0:
            continue
        if _norm_name(column) in KEY_COLUMN_HINTS:
            score += len(body)  # outrank any column found only by sniffing
        if score > best_score:
            best, best_score = column, score
    return best


def _header_quality(row: pd.Series) -> float:
    """Fraction of cells in `row` that read as column labels.

    Blank cells and email-shaped cells both count against it: the former mark
    a banner line, the latter mark a data row.
    """
    cells = [_cell_text(value) for value in row]
    if not cells:
        return 0.0
    return sum(1 for c in cells if c and not _EMAIL_RE.search(c)) / len(cells)


def _parse_sheet(raw: pd.DataFrame, explicit_key: str | None):
    """Locate the header row and key column. Returns (frame, key_column, offset) or None.

    Every candidate offset is scored and the best one wins, rather than taking
    the first that merely works. A banner row above the real header still has
    the data's emails somewhere beneath it, so "first offset yielding an
    address" picks the banner and silently mislabels every column.
    """
    if raw.empty:
        return None

    best = None
    best_score = float("-inf")
    for offset in range(min(_HEADER_SEARCH_DEPTH, len(raw))):
        body = raw.iloc[offset + 1:].copy()
        if body.empty:
            continue
        body.columns = _clean_headers(raw.iloc[offset])
        key_column = _find_key_column(body, explicit_key)
        if key_column is None:
            continue
        body = body.reset_index(drop=True)
        body[KEY] = body[key_column].map(normalize_email)
        body = body[body[KEY] != ""].reset_index(drop=True)
        if body.empty:
            continue

        # A recognised address-column name is decisive; otherwise how much of
        # the row reads as labels settles it. Earlier offsets break ties.
        score = 100.0 * _header_quality(raw.iloc[offset])
        if _norm_name(key_column) in KEY_COLUMN_HINTS:
            score += 1000.0
        if score > best_score:
            best, best_score = (body, key_column, offset), score

    return best


@dataclass(frozen=True)
class Roster:
    """A workbook's contacts, keyed by normalised address."""

    frame: pd.DataFrame
    key_column: str
    source: Path
    sheet: str
    header_row: int  # 1-based, as Excel shows it

    def __len__(self) -> int:
        return len(self.unique)

    @cached_property
    def unique(self) -> pd.DataFrame:
        """One row per address; the first occurrence wins."""
        return self.frame.drop_duplicates(subset=KEY, keep="first").reset_index(drop=True)

    @cached_property
    def repeated(self) -> pd.DataFrame:
        """Rows whose address appears more than once in the source."""
        counts = self.frame[KEY].value_counts()
        repeats = counts[counts > 1].index
        return self.frame[self.frame[KEY].isin(repeats)].sort_values(KEY)

    def compare(self, other: "Roster") -> "Reconciliation":
        return Reconciliation(left=self, right=other)


def load_roster(
    path: str | Path,
    key: str | None = None,
    sheet: str | int | None = None,
) -> Roster:
    """Read contacts from an Excel workbook.

    The header row, the key column, and the sheet are all detected when not
    given: the first sheet yielding at least one valid address wins. Pass
    `key` to force a column when a file holds several address columns and the
    wrong one is chosen.

    Raises KeyError if no sheet contains a usable address column.
    """
    path = Path(path)
    book = pd.read_excel(path, sheet_name=sheet, header=None, dtype=object)
    sheets = book if isinstance(book, dict) else {sheet if sheet is not None else 0: book}

    for name, raw in sheets.items():
        parsed = _parse_sheet(raw, key)
        if parsed is not None:
            frame, key_column, offset = parsed
            return Roster(
                frame=frame,
                key_column=key_column,
                source=path,
                sheet=str(name),
                header_row=offset + 1,
            )

    hint = f" for key column {key!r}" if key else ""
    raise KeyError(f"No sheet in {path.name} holds a usable email column{hint}.")


@dataclass(frozen=True)
class Reconciliation:
    """The difference between two rosters."""

    left: Roster
    right: Roster

    @cached_property
    def only_in_left(self) -> pd.DataFrame:
        """Contacts present in the left file and absent from the right."""
        missing = ~self.left.unique[KEY].isin(set(self.right.unique[KEY]))
        return self.left.unique[missing].reset_index(drop=True)

    @cached_property
    def only_in_right(self) -> pd.DataFrame:
        missing = ~self.right.unique[KEY].isin(set(self.left.unique[KEY]))
        return self.right.unique[missing].reset_index(drop=True)

    @cached_property
    def in_both(self) -> pd.DataFrame:
        shared = self.left.unique[KEY].isin(set(self.right.unique[KEY]))
        return self.left.unique[shared].reset_index(drop=True)

    @cached_property
    def changes(self) -> pd.DataFrame:
        """Field-level differences for contacts appearing in both files.

        `kind` distinguishes the three cases that matter when merging event
        lists: `new_in_right` (the right file supplies a value the left one
        lacks — the enrichment case), `missing_in_right`, and `conflict`
        (both filled, and they disagree).
        """
        left = self.left.unique.set_index(KEY)
        right = self.right.unique.set_index(KEY)
        keys = left.index.intersection(right.index)
        if len(keys) == 0:
            return pd.DataFrame(columns=["email", "field", "left_value", "right_value", "kind"])

        right_by_name = {_norm_name(c): c for c in right.columns if c != KEY}
        rows = []
        for left_column in left.columns:
            if left_column == KEY:
                continue
            right_column = right_by_name.get(_norm_name(left_column))
            if right_column is None:
                continue
            left_values = left.loc[keys, left_column].map(_cell_text)
            right_values = right.loc[keys, right_column].map(_cell_text)
            for email in keys[(left_values != right_values).to_numpy()]:
                before, after = left_values[email], right_values[email]
                if not before:
                    kind = "new_in_right"
                elif not after:
                    kind = "missing_in_right"
                else:
                    kind = "conflict"
                rows.append({
                    "email": email,
                    "field": left_column,
                    "left_value": before,
                    "right_value": after,
                    "kind": kind,
                })
        return pd.DataFrame(rows, columns=["email", "field", "left_value", "right_value", "kind"])

    @cached_property
    def summary(self) -> pd.DataFrame:
        changes = self.changes
        kinds = changes["kind"].value_counts() if not changes.empty else {}
        return pd.DataFrame(
            [
                ("left file", self.left.source.name),
                ("left sheet", self.left.sheet),
                ("left header row", self.left.header_row),
                ("left key column", self.left.key_column),
                ("left contacts", len(self.left)),
                ("left repeated rows", len(self.left.repeated)),
                ("right file", self.right.source.name),
                ("right sheet", self.right.sheet),
                ("right header row", self.right.header_row),
                ("right key column", self.right.key_column),
                ("right contacts", len(self.right)),
                ("right repeated rows", len(self.right.repeated)),
                ("only in left", len(self.only_in_left)),
                ("only in right", len(self.only_in_right)),
                ("in both", len(self.in_both)),
                ("new values in right", int(kinds.get("new_in_right", 0))),
                ("conflicting values", int(kinds.get("conflict", 0))),
            ],
            columns=["metric", "value"],
        )

    def to_excel(self, path: str | Path) -> Path:
        """Write one sheet per category. Returns the path written."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sheets = {
            "summary": self.summary,
            "only_in_left": self.only_in_left,
            "only_in_right": self.only_in_right,
            "changes": self.changes,
        }
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                frame.drop(columns=[KEY], errors="ignore").to_excel(
                    writer, sheet_name=name, index=False
                )
        return path
