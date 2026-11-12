"""camtrap/copresence.py — what else was in the frame.

WHAT THIS OWNS

    The co-presence convention: the tag vocabulary a reviewer writes in Camtrap DP's
    `observationTags`, which campaigns were checked for it, and how a tag — or its
    absence — becomes TRUE, FALSE or unknown in `with_human` / `with_vehicle`.

WHY IT EXISTS

    A frame carries ONE label. Timelapse2's Camtrap DP template has no second
    observation per image (0 multi-row images across all three reviewed exports,
    measured 2026-11-12), so Felipe labels by priority: perro > person > vehicle. A
    frame with the ranger and his dogs is `animal: Canis lupus familiaris`, and the
    ranger is gone from the table. pehuen cannot then tell an accompanied dog from a
    free-ranging one. `observationTags` is the template's own field for this ("pipe
    (|) separated ... key:value pairs"), so recording it there changes neither the
    template nor the standard. Felipe's decision, 2026-11-12.

THE TAGS NAME WHAT RANKS BELOW THE ROW'S OWN LABEL

    The priority means a lower-ranked subject can hide behind a higher one, never the
    reverse: an animal row may carry `with:human` and `with:vehicle`, a human row only
    `with:vehicle`, a vehicle row nothing. A tag that names something the priority
    would have put in the label instead — `with:human` on a vehicle row — means the
    rule was not followed, and it is refused rather than guessed around.

AN EMPTY TAG MEANS "ALONE" ONLY WHERE SOMEONE LOOKED

    Three campaigns were reviewed before this convention existed. An empty tag there
    is not evidence that the dog was alone, so FALSE is written only for rows inside a
    campaign's declared CHECKED scope; outside it the answer is NA. A tag that IS
    present is TRUE wherever it appears — an observed person is evidence whether or
    not the campaign was declared. Fail closed, the same stance as the clock flags.
"""

from __future__ import annotations

import pandas as pd

from camtrap import exports

TAG_HUMAN = 'with:human'
TAG_VEHICLE = 'with:vehicle'
SEPARATOR = '|'

#: Tag -> the Camtrap DP type it says was also in the frame.
TAGS: dict[str, str] = {TAG_HUMAN: exports.TYPE_HUMAN, TAG_VEHICLE: exports.TYPE_VEHICLE}

#: Felipe's labelling priority, highest first. A row may only be tagged with a type
#: that ranks strictly below its own label.
PRIORITY = (exports.TYPE_ANIMAL, exports.TYPE_HUMAN, exports.TYPE_VEHICLE)

#: Output column per tag.
COLUMNS: dict[str, str] = {TAG_HUMAN: 'with_human', TAG_VEHICLE: 'with_vehicle'}

#: Every reviewed row of the campaign was checked for co-presence.
ALL_REVIEWED = 'all'

#: Which reviewed rows were checked, per campaign: ALL_REVIEWED, or the species whose
#: rows were re-checked. A campaign absent here was not checked, so its empty tags are
#: unknown, not "alone". Entries are added as Felipe completes each pass.
CHECKED: dict[str, str | frozenset[str]] = {}


class InvalidObservationTag(ValueError):
    """A tag outside the vocabulary, or one the labelling priority forbids on its row."""


def compose(human: bool, vehicle: bool) -> str:
    """The `observationTags` value for what the reviewer ticked. '' for nothing."""
    chosen = [tag for tag, on in ((TAG_HUMAN, human), (TAG_VEHICLE, vehicle)) if on]
    return SEPARATOR.join(chosen)


def parse(value: str) -> frozenset[str]:
    """One cell -> its tags, lowercased and stripped. Unknown tags are kept, so
    resolve() can name them; an empty cell is the empty set."""
    parts = (p.strip().lower() for p in str(value or '').split(SEPARATOR))
    return frozenset(p for p in parts if p)


def _allowed(kind: str) -> frozenset[str]:
    """Tags a row of this type may carry: those naming a type ranked below it."""
    if kind not in PRIORITY:
        return frozenset()
    below = set(PRIORITY[PRIORITY.index(kind) + 1:])
    return frozenset(tag for tag, t in TAGS.items() if t in below)


def _in_scope(campaign: str, species: pd.Series, reviewed: pd.Series) -> pd.Series:
    scope = CHECKED.get(campaign)
    if scope is None:
        return pd.Series(False, index=species.index)
    if scope == ALL_REVIEWED:
        return reviewed.astype(bool)
    return reviewed.astype(bool) & species.isin(scope)


def resolve(
    tags: pd.Series,
    observation_type: pd.Series,
    species_latin: pd.Series,
    reviewed: pd.Series,
    campaign: str,
) -> pd.DataFrame:
    """`with_human` and `with_vehicle` (nullable boolean), indexed like `tags`.

    TRUE where the tag is present; FALSE where it is absent, the row may carry it, and
    the row is inside CHECKED; NA otherwise. Raises InvalidObservationTag listing every
    offending value, so a typo stops the ingest instead of reading as "alone".
    """
    parsed = tags.fillna('').astype(str).map(parse)
    kind = observation_type.fillna('').astype(str).str.strip()
    species = species_latin.fillna('').astype(str).str.strip()

    problems: dict[str, int] = {}
    for value, k in zip(parsed, kind):
        unknown = value - TAGS.keys()
        forbidden = (value & TAGS.keys()) - _allowed(k)
        for tag in sorted(unknown):
            problems[f'{tag!r} (not a tag)'] = problems.get(f'{tag!r} (not a tag)', 0) + 1
        for tag in sorted(forbidden):
            key = f'{tag!r} on a {k or "(no type)"!r} row'
            problems[key] = problems.get(key, 0) + 1
    if problems:
        listed = '\n'.join(f'    {what:<40} {n:>5} row(s)'
                           for what, n in sorted(problems.items(), key=lambda kv: -kv[1]))
        raise InvalidObservationTag(
            f'{campaign}: observationTags the convention does not allow:\n{listed}\n'
            f'  Valid tags: {", ".join(TAGS)}, separated by {SEPARATOR!r}. A tag may only '
            f'name what ranks below the row\'s own label ({" > ".join(PRIORITY)}). '
            f'See camtrap/copresence.py.'
        )

    in_scope = _in_scope(campaign, species, reviewed)
    out = pd.DataFrame(index=tags.index)
    for tag, column in COLUMNS.items():
        present = parsed.map(lambda s, t=tag: t in s)
        applicable = kind.map(lambda k, t=tag: t in _allowed(k))
        col = pd.Series(pd.NA, index=tags.index, dtype='boolean')
        col[applicable & in_scope] = False
        col[present] = True
        out[column] = col
    return out
