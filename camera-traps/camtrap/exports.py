"""What a Timelapse2 export is, and whether it is fit to diagnose a clock from.

Two exports come out of Timelapse2 per campaign and they answer different questions:

    ImageData_animals.csv   observationType=animal only. Feeds the CLIP classifier.
                            Fine for that; USELESS for clock diagnosis.
    ImageData_total.csv     every image, every category assigned. The only input a
                            clock diagnosis may use.

Why the distinction is enforced in code rather than written in the README: a reset
that happens between two animal photos is invisible in an animal-only export. That
is not hypothetical — it is how otoño 2026 CT_18 was recorded as ONE clock reset
when it had FOUR, and how a single offset came to be applied across all of them,
fabricating dates that reached the pehuen analysis. The README already said to
export all images; the export that reached ingest did not, and nothing noticed.

THE GATE (agreed with Felipe 2026-08-03)

    An export is fit for clock diagnosis only if `person` or `vehicle` appears in
    observationType.

    Presence of categories cannot be the test on its own, because in Timelapse2 as
    Felipe uses it `unclassified` doubles as `empty` — so an export containing only
    {animal, unclassified} looks category-labelled while in fact nothing was ever
    assigned. That is exactly the otoño 2026 file (animal 1,785 / unclassified
    10,283). `person` is the category that proves a real sweep happened, because
    under the field protocol every deployment now begins and ends with a photo of
    the technician: install and retrieval anchors ARE person detections. A campaign
    with no person frames means either the sweep was not done or the protocol was
    not followed — both worth stopping for.

    THE OVERRIDE is for the genuine exception only: a campaign swept in full that
    really contains no person or vehicle. It is a file in the campaign directory,
    not a command-line flag, so the decision is recorded with a name and a date and
    travels with the data:

        data/campaigns/<campaign>/export_gate_override.txt
            verified_by: Felipe Guarda
            date: 2026-08-03
            reason: swept all 12068 images in Timelapse2; the camera was serviced
                    by a technician who never triggered it, so no person frame
                    exists on this card.

    An override CANNOT rescue an export whose categories were never assigned. That
    is not an exception to the rule, it is the absence of the work the rule checks
    for, and no signature turns unswept rows into a sweep.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# The two exports, under the names the README tells Felipe to save them as.
TOTAL_EXPORT_FILENAME  = 'ImageData_total.csv'
ANIMAL_EXPORT_FILENAME = 'ImageData_animals.csv'

OBSERVATION_TYPE_COLUMN = 'observationType'

# A full sweep assigns one of these to every image.
FULL_CATEGORY_TYPES = frozenset({'empty', 'animal', 'person', 'vehicle'})

# `unclassified` means "never looked at" — and doubles as `empty` in Felipe's
# Timelapse2 template, which is why it cannot count as evidence of anything.
UNASSIGNED_TYPES = frozenset({'unclassified', ''})

# The categories whose presence proves a sweep actually happened.
PROOF_OF_SWEEP = frozenset({'person', 'vehicle'})

OVERRIDE_FILENAME = 'export_gate_override.txt'
OVERRIDE_REQUIRED_KEYS = ('verified_by', 'date', 'reason')

# Verdicts. PASS may proceed; NO_PROOF_OF_SWEEP may proceed with an override; the
# other two may never proceed.
PASS               = 'full_category_sweep'
NO_PROOF_OF_SWEEP  = 'no_person_or_vehicle'
NEVER_ASSIGNED     = 'categories_never_assigned'
NO_ROWS            = 'empty_export'

# Only this verdict is an exception a human may sign off on.
OVERRIDABLE_VERDICTS = frozenset({NO_PROOF_OF_SWEEP})


class ExportGateError(ValueError):
    """The export may not be used to diagnose a clock. The message names the fix."""


@dataclass(frozen=True)
class Override:
    verified_by: str
    date: str
    reason: str
    path: Path


@dataclass(frozen=True)
class CategoryAudit:
    """What the export's observationType column holds, and the verdict on it."""
    counts: dict[str, int]
    verdict: str
    n_rows: int = 0
    override: Override | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when the export satisfies the rule on its own merits."""
        return self.verdict == PASS

    @property
    def usable(self) -> bool:
        return self.passed or self.override is not None

    def describe(self) -> str:
        lines = [f'observationType over {self.n_rows} row(s):']
        for cat, n in sorted(self.counts.items(), key=lambda kv: -kv[1]):
            lines.append(f'    {cat or "(blank)":<14} {n:>7}')
        lines.append(f'  verdict: {self.verdict}')
        if self.override is not None:
            lines.append(
                f'  OVERRIDE accepted — {self.override.verified_by} '
                f'({self.override.date}): {self.override.reason}'
            )
        lines.extend(f'  note: {n}' for n in self.notes)
        return '\n'.join(lines)


# =============================================================================
# The rule
# =============================================================================

def audit_categories(observation_type: pd.Series) -> CategoryAudit:
    """Apply the gate to an observationType column. Pure — no file access.

    Order matters: NEVER_ASSIGNED is checked before NO_PROOF_OF_SWEEP because only
    the latter is overridable, and an export of {animal, unclassified} must not be
    able to present itself as the overridable case.
    """
    cats = observation_type.fillna('').astype(str).str.strip().str.lower()
    counts = {k: int(v) for k, v in cats.value_counts().items()}
    n_rows = len(cats)
    present = set(counts)
    notes: list[str] = []

    unknown = present - FULL_CATEGORY_TYPES - UNASSIGNED_TYPES
    if unknown:
        notes.append(
            f'unrecognised observationType value(s) {sorted(unknown)} — not one of '
            f'{sorted(FULL_CATEGORY_TYPES)}; they count as neither assigned nor proof'
        )

    def audit(verdict: str) -> CategoryAudit:
        return CategoryAudit(counts=counts, verdict=verdict, n_rows=n_rows, notes=notes)

    if n_rows == 0:
        return audit(NO_ROWS)

    n_unassigned = sum(counts.get(t, 0) for t in UNASSIGNED_TYPES)
    if n_unassigned:
        notes.append(
            f'{n_unassigned}/{n_rows} row(s) are unclassified or blank. In this '
            f'Timelapse2 template `unclassified` doubles as `empty`, so these rows '
            f'cannot be told apart from images nobody looked at'
        )

    # Nothing but animal + unclassified means the categories were never assigned:
    # `animal` comes for free from the classifier round, and `unclassified` is the
    # default state of every row.
    if not (present - UNASSIGNED_TYPES - {'animal'}):
        return audit(NEVER_ASSIGNED)

    if not (present & PROOF_OF_SWEEP):
        return audit(NO_PROOF_OF_SWEEP)

    return audit(PASS)


# =============================================================================
# The override file
# =============================================================================

def load_override(path: Path) -> Override:
    """Parse export_gate_override.txt. Raises ExportGateError if it is unusable.

    A malformed override is refused rather than ignored: silently falling back to
    "no override" would turn a typo into a rejected campaign with a confusing
    message, and silently accepting it would let an unsigned file open the gate.
    """
    fields: dict[str, str] = {}
    key: str | None = None
    for raw in path.read_text(encoding='utf-8').splitlines():
        if not raw.strip() or raw.lstrip().startswith('#'):
            continue
        if raw[:1].isspace() and key:          # continuation of the previous value
            fields[key] = f'{fields[key]} {raw.strip()}'.strip()
            continue
        head, sep, tail = raw.partition(':')
        if not sep:
            raise ExportGateError(
                f'{path}: line {raw!r} is neither `key: value` nor an indented '
                f'continuation of the line above'
            )
        key = head.strip().lower()
        fields[key] = tail.strip()

    missing = [k for k in OVERRIDE_REQUIRED_KEYS if not fields.get(k)]
    if missing:
        raise ExportGateError(
            f'{path}: missing or empty {missing}. An override must record who '
            f'verified the sweep, when, and why the campaign genuinely contains no '
            f'person or vehicle frame. Required keys: '
            f'{list(OVERRIDE_REQUIRED_KEYS)}'
        )

    return Override(
        verified_by=fields['verified_by'],
        date=fields['date'],
        reason=fields['reason'],
        path=path,
    )


def _gate_message(audit: CategoryAudit, source: str) -> str:
    fix = {
        NO_ROWS: (
            'The export has no rows at all. Re-export from Timelapse2 with no '
            'filter applied.'
        ),
        NEVER_ASSIGNED: (
            'Only `animal` and `unclassified` appear, which is the state of a '
            'project whose categories were never assigned — `animal` comes from the '
            'classifier round and `unclassified` is every row\'s default. Sweep the '
            'campaign in Timelapse2 assigning empty / animal / person / vehicle to '
            'every image, then export ALL images (no filter) as '
            f'{TOTAL_EXPORT_FILENAME}. No override can accept this file.'
        ),
        NO_PROOF_OF_SWEEP: (
            'Neither `person` nor `vehicle` appears. Under the field protocol every '
            'deployment opens and closes with a photo of the technician, so a '
            'person-free campaign means either the sweep is incomplete or the '
            'protocol was not followed at some station. If the campaign genuinely '
            f'contains neither, record that in {OVERRIDE_FILENAME} beside the '
            f'export with keys {list(OVERRIDE_REQUIRED_KEYS)}.'
        ),
    }.get(audit.verdict, 'Re-export all images with every category assigned.')

    return (
        f'{source} cannot be used to diagnose camera clocks '
        f'(verdict: {audit.verdict}).\n'
        f'{audit.describe()}\n'
        f'  FIX: {fix}'
    )


def require_full_category(
    df: pd.DataFrame,
    *,
    source: str,
    override_dir: Path | None = None,
) -> CategoryAudit:
    """Gate a loaded export. Returns the audit, or raises ExportGateError.

    `source` is what to name in the error (a path, usually). `override_dir` is where
    to look for export_gate_override.txt — normally the campaign directory.
    """
    if OBSERVATION_TYPE_COLUMN not in df.columns:
        raise ExportGateError(
            f'{source}: no {OBSERVATION_TYPE_COLUMN!r} column, so this is not a '
            f'Timelapse2 CamtrapDP export'
        )

    audit = audit_categories(df[OBSERVATION_TYPE_COLUMN])
    if audit.passed:
        return audit

    override_path = (override_dir or Path('.')) / OVERRIDE_FILENAME
    if audit.verdict in OVERRIDABLE_VERDICTS and override_path.exists():
        override = load_override(override_path)
        return CategoryAudit(
            counts=audit.counts, verdict=audit.verdict, n_rows=audit.n_rows,
            override=override, notes=audit.notes,
        )

    if audit.verdict not in OVERRIDABLE_VERDICTS and override_path.exists():
        raise ExportGateError(
            f'{_gate_message(audit, source)}\n'
            f'  NOTE: {override_path} exists but cannot apply — {audit.verdict} is '
            f'not an overridable verdict.'
        )

    raise ExportGateError(_gate_message(audit, source))


# =============================================================================
# Reading
# =============================================================================

def read_total_export(campaign_dir: Path, *, filename: str = TOTAL_EXPORT_FILENAME):
    """Load and gate the all-images export. Returns (DataFrame, CategoryAudit).

    Hard-fails when the file is absent instead of falling back to the animal-only
    export. The fallback is the bug: it is what let an animal-only file reach clock
    diagnosis in the first place.
    """
    path = campaign_dir / filename
    if not path.exists():
        raise ExportGateError(
            f'{path} not found. Clock diagnosis requires the all-images export with '
            f'every category assigned (empty / animal / person / vehicle) — see '
            f'README Step 2. {ANIMAL_EXPORT_FILENAME} is NOT a substitute: a reset '
            f'between two animal photos is invisible in it, which is how otoño 2026 '
            f'CT_18 lost four resets.'
        )

    df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    audit = require_full_category(df, source=str(path), override_dir=campaign_dir)
    return df, audit


# =============================================================================
# CLI — check an export the moment it is made, without running ingest
# =============================================================================

def main(argv=None) -> int:
    """`python -m camtrap.exports <campaign_dir|export.csv>`"""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print(
            'usage: python -m camtrap.exports <campaign_dir | export.csv>\n'
            '  Checks whether a Timelapse2 export is fit for clock diagnosis.',
            file=sys.stderr,
        )
        return 2

    target = Path(args[0])
    if target.is_dir():
        campaign_dir, path = target, target / TOTAL_EXPORT_FILENAME
    else:
        campaign_dir, path = target.parent, target

    if not path.exists():
        print(f'ERROR: {path} not found', file=sys.stderr)
        return 2

    df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    try:
        audit = require_full_category(
            df, source=str(path), override_dir=campaign_dir,
        )
    except ExportGateError as exc:
        print(f'REJECTED\n{exc}', file=sys.stderr)
        return 1

    print(f'OK — {path}')
    print(audit.describe())
    return 0


if __name__ == '__main__':
    sys.exit(main())
