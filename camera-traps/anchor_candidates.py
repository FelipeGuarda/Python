"""
anchor_candidates.py — the short list of frames that could become clock anchors.

An anchor is a frame whose real wall-clock time someone can state, paired with what
the camera's clock said at that moment. Each one buys back a whole segment of a
broken clock, and a mid-deployment anchor is the only thing that can rescue an
interior segment. So the bottleneck in repairing a campaign is never the arithmetic
— it is finding, among 12,000 frames, the handful worth opening.

This report finds them, per station:

    person / vehicle detections   From MegaDetector. Under the field protocol every
                                  deployment opens and closes with a photo of the
                                  technician, so these ARE the install and retrieval
                                  anchors. This is the main event.
    counter-0001 frames           The first frame of an SD-card DCIM folder, i.e.
                                  where a card was swapped or a camera rebooted — the
                                  usual neighbourhood of a reset.
    segment boundaries            The first and last frame of every segment the clock
                                  diagnosis found, so a segment still lacking an
                                  anchor can be inspected directly.

It is deliberately NOT gated on the full-category export rule: a campaign that fails
the gate is exactly the one that needs this list. It reports the gate verdict and
carries on.

USAGE
    python anchor_candidates.py --campaign otono_2026
    python anchor_candidates.py --campaign otono_2026 --unanchored-only

OUTPUT
    data/campaigns/<name>/anchor_candidates.csv   one row per candidate frame
    plus a per-station summary on stdout naming the segments still unrepairable and
    what could rescue each one.

WHAT TO DO WITH IT
    Open the candidate images for a segment that needs an anchor. If the frame shows
    a person at a moment you can date — a visit in the field notebook, a phone photo
    with its own timestamp — add a row to deployment_anchors.csv with the camera's
    stamp as `camera_datetime` and the real time as `real_datetime`. If containment
    cannot place the anchor in the right segment, set `segment_index` explicitly and
    say in `notes` who checked it by eye.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Force UTF-8 on stdout/stderr so accented species names and box-drawing characters
# do not raise UnicodeEncodeError on a default Windows console (cp1252).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from camtrap import anchors, clocks, detections, exports, stations
from camtrap.anchors import ANCHOR_FILENAME, FIELD_NOTES_FILENAME, FieldRecord, load_anchors
from timestamps import diagnose_campaign, prepare_total

OUTPUT_FILENAME = 'anchor_candidates.csv'

# The evidence vocabulary lives in camtrap/anchors.py, which also ranks it: this
# report names candidates, the anchor module decides which name outranks which.
KIND_HUMAN        = anchors.EVIDENCE_HUMAN_LABELLED
KIND_VEH_LABELLED = anchors.EVIDENCE_VEHICLE_LABELLED
KIND_PERSON       = anchors.EVIDENCE_PERSON_DETECTION
KIND_VEHICLE      = anchors.EVIDENCE_VEHICLE_DETECTION
KIND_COUNTER_0001 = anchors.EVIDENCE_COUNTER_0001
KIND_SEGMENT_EDGE = anchors.EVIDENCE_SEGMENT_EDGE

# The swept export's proof-of-sweep categories, mapped to the evidence they provide.
# Keyed on exports' own constants so the Camtrap DP vocabulary is stated once.
LABELLED_KIND = {
    exports.TYPE_HUMAN:   KIND_HUMAN,
    exports.TYPE_VEHICLE: KIND_VEH_LABELLED,
}

OUTPUT_COLUMNS = [
    'station', 'camera_num', 'deployment', 'file_name', 'camera_datetime',
    'candidate_kind', 'md_conf', 'clock_segment', 'segment_needs_anchor',
    'suggested_anchor_type', 'rel_path',
]


EDGE_TOLERANCE = pd.Timedelta(days=1)


def _suggest_anchor_type(camera_dt, ref: tuple | None, trustworthy: bool) -> str:
    """install / retrieval / mid_visit — which field visit this frame is likely from.

    A suggestion only, to point at the right notebook entry. It says nothing about
    validity, and the anchor_type Felipe writes down is the one that counts.

    Measured against the station's OWN range of trustworthy frames, not against the
    anchor window: comparing a camera stamp to a wall-clock window is meaningless for
    the very cameras this report exists for. A frame whose stamp is not trustworthy
    gets 'unknown' rather than a guess — CT_18's 2017 frames would otherwise all be
    labelled `install` for sitting before the deployment began.
    """
    if not trustworthy or ref is None or pd.isna(camera_dt):
        return 'unknown'
    start, end = ref
    if camera_dt <= start + EDGE_TOLERANCE:
        return 'install'
    if camera_dt >= end - EDGE_TOLERANCE:
        return 'retrieval'
    return 'mid_visit'


def build_candidates(
    total: pd.DataFrame,
    diagnoses: dict,
    md_json: Path | None,
    *,
    min_conf: float,
) -> pd.DataFrame:
    """One row per candidate frame, across every station."""
    rows: list[dict] = []

    # ── MegaDetector person / vehicle ─────────────────────────────────────────
    det = pd.DataFrame(columns=detections.DETECTION_COLUMNS)
    if md_json is not None and md_json.exists():
        det = detections.read_detections(
            md_json, min_conf=min_conf,
            categories={detections.CATEGORY_PERSON, detections.CATEGORY_VEHICLE},
        )
        # One row per image, keeping the strongest detection: two people in a frame
        # is still one anchor opportunity.
        det = (
            det.sort_values('conf', ascending=False)
               .drop_duplicates(subset=['rel_path'], keep='first')
        )

    # The detection JSON keys on the pre-flatten relative path, the export on
    # (Deployments, File). Join on the file name within a deployment — the only key
    # both sides carry once the tree is flat.
    det_lookup = {
        (str(r.deployment), str(r.file_name)): (r.category, float(r.conf))
        for r in det.itertuples()
    }

    for camera_num, sd in sorted(diagnoses.items()):
        frames = total[total['_camera_num'] == camera_num].copy()
        seg = clocks.segment_for_rows(sd.diagnosis, frames['_datetime_parsed'])
        frames['_segment'] = seg

        needs_anchor = {
            r.segment_index for r in sd.repairs if not r.valid_date
        }

        # The station's own span of plausible frames — what install/retrieval are
        # measured against. Restricted to the deployment window when one exists, so a
        # camera that jumped to 2017 is bracketed by its real frames, not its bogus
        # ones.
        stamps = frames['_datetime_parsed']
        if sd.window is not None:
            plausible = stamps.between(sd.window[0], sd.window[1])
        else:
            plausible = stamps.notna()
        ref = (stamps[plausible].min(), stamps[plausible].max()) if plausible.any() else None

        def _row(frame, kind: str, conf=None) -> dict:
            seg_val = frame['_segment']
            return {
                'station': sd.station_label,
                'camera_num': camera_num,
                'deployment': frame['Deployments'],
                'file_name': frame['File'],
                'camera_datetime': frame['_datetime_parsed'],
                'candidate_kind': kind,
                'md_conf': conf,
                'clock_segment': seg_val,
                'segment_needs_anchor': (
                    pd.NA if pd.isna(seg_val) else bool(seg_val in needs_anchor)
                ),
                'suggested_anchor_type': _suggest_anchor_type(
                    frame['_datetime_parsed'], ref,
                    bool(plausible.get(frame.name, False)),
                ),
                'rel_path': frame.get('RelativePath', ''),
            }

        for _, frame in frames.iterrows():
            name = str(frame['File']).strip()
            hit = det_lookup.get((str(frame['Deployments']).strip(), name))
            conf = hit[1] if hit is not None else None

            # A `human` label outranks a MegaDetector box on the same frame: someone
            # opened the image and said so, where MegaDetector only guessed. One row
            # per frame, carrying the strongest evidence it has — emitting both would
            # list the same photograph twice as if it were two opportunities.
            labelled = str(frame.get(exports.OBSERVATION_TYPE_COLUMN, '')).strip()
            if labelled in LABELLED_KIND:
                rows.append(_row(frame, LABELLED_KIND[labelled], conf))
                continue

            if hit is not None:
                category, _ = hit
                rows.append(_row(
                    frame,
                    (KIND_PERSON if category == detections.CATEGORY_PERSON
                     else KIND_VEHICLE),
                    conf,
                ))
                continue

            _, counter = clocks.parse_filename(name)
            if counter == 1:
                rows.append(_row(frame, KIND_COUNTER_0001))

        # Segment edges — the frames that bracket each run of the clock. On a camera
        # with several segments these are what someone has to look at to say which
        # reset a segment belongs to.
        if len(sd.diagnosis.segments) > 1:
            for seg_index, grp in frames.dropna(subset=['_segment']).groupby('_segment'):
                grp = grp.sort_values('_datetime_parsed')
                for pos in {0, len(grp) - 1}:
                    rows.append(_row(grp.iloc[pos], KIND_SEGMENT_EDGE))

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=['station', 'file_name', 'candidate_kind'])
    return out[OUTPUT_COLUMNS].sort_values(
        ['camera_num', 'camera_datetime', 'candidate_kind'], na_position='last'
    ).reset_index(drop=True)


def render_summary(candidates: pd.DataFrame, diagnoses: dict) -> str:
    lines: list[str] = []
    total_needing = 0

    for camera_num, sd in sorted(diagnoses.items()):
        unrepaired = [r for r in sd.repairs if not r.valid_date]
        if not unrepaired:
            continue
        total_needing += len(unrepaired)
        mine = candidates[candidates['camera_num'] == camera_num]
        lines.append(
            f'  {sd.station_label} — {len(sd.diagnosis.segments)} segment(s), '
            f'{len(unrepaired)} still unrepairable, {sd.n_anchors} anchor(s) on file'
        )
        for r in unrepaired:
            seg = next(s for s in sd.diagnosis.segments if s.index == r.segment_index)
            in_seg = mine[mine['clock_segment'].eq(r.segment_index)]
            people = in_seg[in_seg['candidate_kind'].isin(
                [KIND_HUMAN, KIND_VEH_LABELLED, KIND_PERSON, KIND_VEHICLE]
            )]
            lines.append(
                f'    [{r.segment_index}] {seg.n_images:>5} frame(s) '
                f'{seg.camera_start:%Y-%m-%d %H:%M} → {seg.camera_end:%Y-%m-%d %H:%M}'
                f'  {r.reason}'
            )
            if not seg.coherent:
                lines.append(
                    '          no anchor can repair this segment: it is incoherent '
                    '(its filenames disagree with their own stamps), so no single '
                    'offset describes it'
                )
                continue
            if len(people):
                lines.append(
                    f'          {len(people)} person/vehicle frame(s) inside it — '
                    f'{", ".join(people["file_name"].head(4))}'
                    + (' …' if len(people) > 4 else '')
                )
            else:
                lines.append(
                    f'          NO person/vehicle frame inside it. '
                    f'{len(in_seg)} other candidate(s); otherwise this segment needs '
                    f'a field-notebook date or an explicit segment_index on an '
                    f'existing anchor'
                )

    if not lines:
        return 'Every segment of every station is already repairable — no anchors needed.'
    return (
        f'Segments still unrepairable: {total_needing}\n'
        + '\n'.join(lines)
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description='List the frames that could become clock anchors for a '
                    'campaign: MegaDetector person/vehicle detections, '
                    'counter-0001 frames, and every segment boundary.',
    )
    ap.add_argument('--campaign', required=True)
    ap.add_argument('--data-root', default='data/campaigns')
    ap.add_argument('--min-conf', type=float, default=detections.DEFAULT_CONFIDENCE,
                    help=f'MegaDetector confidence floor '
                         f'(default {detections.DEFAULT_CONFIDENCE})')
    ap.add_argument('--unanchored-only', action='store_true',
                    help='Write only candidates inside a segment that is still '
                         'unrepairable.')
    args = ap.parse_args(argv)

    campaign_dir = Path(args.data_root) / args.campaign
    if not campaign_dir.is_dir():
        print(f'ERROR: campaign dir not found: {campaign_dir}', file=sys.stderr)
        return 2

    total_csv = campaign_dir / exports.TOTAL_EXPORT_FILENAME
    if not total_csv.exists():
        print(f'ERROR: {total_csv} not found — this report needs the all-images '
              f'export to see the frames between the animal photos.', file=sys.stderr)
        return 2

    print(f'Reading all-images export : {total_csv}')
    total = pd.read_csv(total_csv, dtype=str, keep_default_na=False, low_memory=False)
    audit = exports.audit_categories(total[exports.OBSERVATION_TYPE_COLUMN])
    print(f'  {len(total)} row(s); export gate verdict: {audit.verdict}'
          + ('' if audit.passed else ' (reported, not enforced — this report is a '
                                     'review aid, not an ingest path)'))
    total = prepare_total(total, campaign_dir)

    on_file = load_anchors(campaign_dir / ANCHOR_FILENAME)
    print(f'  {len(on_file)} anchor row(s) already on file')

    field = FieldRecord.load(Path(args.data_root) / FIELD_NOTES_FILENAME)
    print(f'  {len(field)} field visit(s) on file')

    try:
        diagnoses = diagnose_campaign(total, on_file, args.campaign, field)
    except stations.UnknownStation as exc:
        print(f'\nERROR: {exc}', file=sys.stderr)
        return 3

    md_json = campaign_dir / detections.MEGADETECTOR_FILENAME
    if not md_json.exists():
        print(f'  WARNING: {md_json.name} not found — person/vehicle candidates '
              f'cannot be listed, only counter-0001 frames and segment edges')
        md_json = None

    candidates = build_candidates(
        total, diagnoses, md_json, min_conf=args.min_conf,
    )
    if args.unanchored_only:
        candidates = candidates[candidates['segment_needs_anchor'].eq(True)]

    print()
    print(render_summary(candidates, diagnoses))
    print()
    kinds = candidates['candidate_kind'].value_counts().to_dict()
    print(f'{len(candidates)} candidate frame(s): {kinds}')

    out_path = campaign_dir / OUTPUT_FILENAME
    candidates.to_csv(out_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
    print(f'Wrote: {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
