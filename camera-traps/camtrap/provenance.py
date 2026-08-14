"""Whether the frames in a deployment folder are ONE camera's output.

This module owns one question: *how many capture stories does this folder tell?*

It is deliberately not part of `clocks.py`. That module owns how a camera's clock
fails; this one owns whose frames these are.

It imports NOTHING from `clocks`, which the design did not anticipate and which is
the stronger position: `parse_filename` speaks the MMDD grammar, and the entire point
here is the frames that grammar cannot parse. Shapes are grammar-agnostic, so this
module needs no filename knowledge at all and cannot drift from anyone else's. It is
stdlib-only for the same reason.

WHY THIS EXISTS

    Primavera 2025 arrived with station `TC23_M20.2` — 2,460 files — nested inside
    `TC22_M19.2`. Flattening would have attributed every one of them to camera 22, at
    camera 22's coordinates.

    The pipeline already SAW those frames. Pooled into CT22, `establish_order`
    reports `2460 filename(s) do not match the MMDD+counter grammar` and returns
    ordered=False. But it reads that as an ORDERING problem, and per the P1 asymmetry
    (clocks.py) failing to order does not condemn a camera — so the dates stand and
    2,460 frames keep camera 22's identity. The evidence was present and went to the
    wrong question. This module asks the right one.

    `stations.names_a_station()` also catches the TC23 arrangement, and catches it
    earlier and more precisely — it can name the folder to move. But it recognises
    station names by SHAPE, from the three spellings this project has used, so a
    folder called `Camara 23` or `Cam23` walks straight past it. This module never
    enumerates anything: it groups frames by the shape of their own filenames, so a
    camera whose naming convention nobody has seen before forms its own group
    automatically. Narrow-and-precise in front, general-and-vague behind.

THE RULE

    A deployment tells more than one capture story when two or more distinct filename
    shapes each form a COUNTER RUN. A run — not merely a group — because a
    hand-renamed one-off must not read as a second camera: otoño 2026 CT_27 holds
    `01060117_fiscalizador.JPG`, and one frame is not a sequence.

    The shape is the filename stem with every digit run collapsed to `#`, so
    `IMAG0001` and `01120001` are `IMAG#` and `#`. The EXTENSION IS EXCLUDED, which
    matters: these cameras fire three stills and a video, and `01120001.JPG` and
    `01120004.AVI` are one camera telling one story.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

_DIGIT_RUN = re.compile(r'\d+')

# How many names to keep per population, purely so the caller can show the operator
# what it is looking at. Enough to recognise a folder, not enough to fill a terminal.
_SAMPLE = 3


@dataclass
class Population:
    """One filename shape found inside a deployment, and how many frames wear it."""

    shape: str
    n: int
    index_min: int
    index_max: int
    sample: list[str] = field(default_factory=list)

    @property
    def is_run(self) -> bool:
        """Do these frames form a sequence, rather than being incidental?

        Two frames with different indices are a run. One frame is not, and neither
        are several frames that share an index — that is a repeated name, which
        `resolve_dest` handles and which says nothing about provenance.
        """
        return self.n >= 2 and self.index_min != self.index_max


def shape_of(file_name: str) -> str:
    """`IMAG0001.JPG` -> 'IMAG#'.  `01120001.JPG` -> '#'.

    Extension excluded on purpose — see the module docstring: a camera that fires
    three stills and a video is one story, not two.
    """
    stem = file_name.rsplit('.', 1)[0] if '.' in file_name else file_name
    return _DIGIT_RUN.sub('#', stem)


def _index_of(file_name: str) -> int | None:
    """A monotone position within a shape group — the LAST digit run.

    Not the semantic counter: `clocks.parse_filename` owns that, and it only speaks
    the MMDD grammar, whereas the whole point here is to handle names it cannot
    parse. Within one shape group every name has the same digit-run structure, so
    comparing the last run is consistent even when its meaning is unknown.
    """
    runs = _DIGIT_RUN.findall(file_name.rsplit('.', 1)[0] if '.' in file_name else file_name)
    return int(runs[-1]) if runs else None


def _base_shape(shape: str, known: Iterable[str]) -> str:
    """Fold a prefixed shape back onto the shape it was derived from.

    `#EK#_#` and `#` are one story: the first is the second with something glued to
    its front. Cameras do not do that to their own filenames — renaming tools do,
    and ours is one of them (`resolve_dest` prefixes a colliding frame with its DCIM
    folder, `101EK113_06050820.JPG`). Measured, not supposed: without this fold, pv
    2025-2026 CT14's 13 prefixed names formed their own run and read as a second
    camera, the only false positive across all four campaigns.

    Stated as a property of NAMES, not of our script, so it does not import
    flatten's rename rule: a shape that is a strict suffix of another at a separator
    boundary is that other shape wearing a prefix.

    Resolved TRANSITIVELY to the ultimate base — peeling one prefix at a time, taking
    the longest match at each step. A single pass is not enough: with `#`, `X_#` and
    `Y_X_#` present, `Y_X_#` folds onto `X_#` while `X_#` folds onto `#`, leaving two
    groups that are one story. Each step strictly shortens the shape, so this
    terminates.
    """
    seen = {shape}
    while True:
        bases = [k for k in known if k not in seen and shape.endswith(f'_{k}')]
        if not bases:
            return shape
        shape = max(bases, key=len)
        seen.add(shape)


def populations(file_names: Iterable[str]) -> list[Population]:
    """Every filename shape in the folder, largest first. Diagnostic, not a verdict."""
    raw: dict[str, list[str]] = {}
    for name in file_names:
        raw.setdefault(shape_of(name), []).append(name)

    groups: dict[str, list[str]] = {}
    for shape, names in raw.items():
        groups.setdefault(_base_shape(shape, raw), []).extend(names)

    out: list[Population] = []
    for shape, names in groups.items():
        indices = [i for i in (_index_of(n) for n in names) if i is not None]
        out.append(Population(
            shape=shape,
            n=len(names),
            index_min=min(indices) if indices else -1,
            index_max=max(indices) if indices else -1,
            sample=sorted(names)[:_SAMPLE],
        ))
    return sorted(out, key=lambda p: (-p.n, p.shape))


def multiple_capture_stories(file_names: Iterable[str]) -> list[Population]:
    """The competing populations, or [] when the deployment tells one story.

    Empty on agreement rather than raising: the overwhelmingly common case is a
    clean folder, and a caller that has to wrap every deployment in try/except to
    learn "nothing wrong" has been given the wrong interface. The caller decides
    what a conflict costs — flatten refuses; a report might only note it.
    """
    runs = [p for p in populations(file_names) if p.is_run]
    return runs if len(runs) > 1 else []
