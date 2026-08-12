"""The MegaDetector JSON — one place that knows its shape.

MegaDetector reports what it saw, per image, with no knowledge of stations or
campaigns:

    {"images": [{"file": "CT_01/01120011.JPG",
                 "detections": [{"category": "2", "conf": 0.98, "bbox": [...]}]}],
     "detection_categories": {"1": "animal", "2": "person", "3": "vehicle"}}

The category numbers are read from the file rather than hardcoded — AddaxAI has
shipped more than one model and the mapping is per-file metadata, not a constant.

Why this module exists: `person` detections are the raw material of clock anchors.
Every install and retrieval photo of a technician is one, and an anchor buys back a
whole segment of a broken clock — so finding them cheaply is what makes the repair
rule usable rather than theoretical. `anchor_candidates.py` is the consumer.

KNOWN DUPLICATION: `classify_campaign/data_loader.py` decodes this same JSON
independently, animal-only and config-driven, for the CLIP crop pipeline (finding
F002 of the 2026-07-29 review). It should move here. Until it does, a change to the
MegaDetector output format touches two files, and this docstring is the warning.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

MEGADETECTOR_FILENAME = 'timelapse_recognition_file.json'

# Below this, a detection is noise. MegaDetector v5b's own metadata calls 0.2 its
# typical threshold; anchors are reviewed by eye afterwards, so this stays low
# enough to keep a badly-lit technician and high enough to be a short list.
DEFAULT_CONFIDENCE = 0.2

DETECTION_COLUMNS = ['rel_path', 'deployment', 'file_name', 'category', 'conf']

# MegaDetector's OWN category names, as its `detection_categories` map spells them.
# This is not the Camtrap DP vocabulary — that one says `human` where this says
# `person`, and `camtrap/exports.py` owns it. Named here so callers asking for person
# detections do not restate a decision this module owns.
CATEGORY_ANIMAL  = 'animal'
CATEGORY_PERSON  = 'person'
CATEGORY_VEHICLE = 'vehicle'


def read_detections(
    json_path: Path,
    *,
    min_conf: float = DEFAULT_CONFIDENCE,
    categories: set[str] | None = None,
) -> pd.DataFrame:
    """Flatten the JSON to one row per detection.

    `categories` filters by NAME ('person', 'vehicle', 'animal'), resolved through
    the file's own `detection_categories` map. Returns DETECTION_COLUMNS; empty
    frame if nothing matches.

    One row per detection, not per image: an image with two people yields two rows.
    Callers that want images should group — collapsing here would hide the count.
    """
    with open(json_path, encoding='utf-8') as fh:
        data = json.load(fh)

    cat_names = {
        str(k): str(v).strip().lower()
        for k, v in (data.get('detection_categories') or {}).items()
    }
    if not cat_names:
        raise ValueError(
            f'{json_path}: no `detection_categories` map, so detection numbers '
            f'cannot be named. Refusing to guess that 1=animal, 2=person.'
        )

    wanted = None if categories is None else {c.strip().lower() for c in categories}
    unknown = None if wanted is None else wanted - set(cat_names.values())
    if unknown:
        raise ValueError(
            f'{json_path}: no such detection category {sorted(unknown)}; this file '
            f'knows {sorted(set(cat_names.values()))}'
        )

    rows: list[dict] = []
    for img in data.get('images', []):
        rel = str(img.get('file', '')).replace('\\', '/')
        if not rel:
            continue
        parts = rel.split('/')
        for det in img.get('detections') or []:
            name = cat_names.get(str(det.get('category')), '')
            if wanted is not None and name not in wanted:
                continue
            conf = float(det.get('conf') or 0.0)
            if conf < min_conf:
                continue
            rows.append({
                'rel_path': rel,
                'deployment': parts[0] if len(parts) > 1 else '',
                'file_name': parts[-1],
                'category': name,
                'conf': conf,
            })

    return pd.DataFrame(rows, columns=DETECTION_COLUMNS)
