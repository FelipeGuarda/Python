#!/usr/bin/env python3
"""
flatten_for_camtrapdp.py — Flatten camera-trap deployment folders for CamtrapDP.

Moves every image/video file from any depth of subfolder inside each deployment
up into the deployment folder itself, then removes the now-empty subdirectories.

Usage:
    python flatten_for_camtrapdp.py /path/to/DataPackage
    python flatten_for_camtrapdp.py /path/to/DataPackage --dry-run

WHY THIS SCRIPT WRITES A MANIFEST
    An SD card stores roughly 999 images per auto-created DCIM folder
    (`100EK113`, `101EK113`, …) and the 4-digit counter in the filename restarts
    inside each one. Flattening therefore pools many `xxxx0001.JPG` frames into a
    single directory, and Timelapse2's `RelativePath` column retains only the
    deployment name — so after this script runs, `(folder, counter)` is gone and
    capture order can no longer be reconstructed from the flat filenames alone.

    That matters because capture order is the only way to detect a camera-clock
    reset: a reset is a datetime that moves backwards relative to the order the
    frames were actually taken. Otoño 2026 was flattened before anyone realised
    this and has no manifest; its five cameras with >999 images
    (CT_14, CT_20, CT_15, CT_08, CT_23) can never satisfy that precondition.

    So every run now writes `dcim_manifest.csv` (schema owned by
    `camtrap/clocks.py`) recording, per file, which DCIM folder it came from and
    what it was renamed to. Nothing is renamed that was not renamed before — the
    manifest is a sidecar, so existing joins on `file_name` keep working.

    Copy the manifest into `data/campaigns/<campaign>/` alongside the Timelapse2
    export.
"""

import argparse
import csv
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from _fileops import cleanup_empty_dirs, is_target, move_file

# camera-traps repo root — so `camtrap` is importable when run from setup/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camtrap import exports, stations
from camtrap.clocks import (
    DCIM_MANIFEST_COLUMNS,
    DCIM_MANIFEST_FILENAME,
    dcim_folder_key,
)


# ── File helpers ──────────────────────────────────────────────────────────────

def collect_subdir_files(deployment_dir: Path) -> list:
    """
    Recursively collect all target files that live INSIDE a subdirectory of
    deployment_dir (i.e., files already directly in deployment_dir are skipped
    because they are already correctly placed).

    Returns a list of (file_path: Path, rel_parts: list[str]) where rel_parts
    contains the intermediate subfolder names between deployment_dir and the file.
    """
    results = []
    for root_str, dirs, filenames in os.walk(deployment_dir):
        root = Path(root_str)
        # Skip hidden directories and don't recurse into them
        dirs[:] = sorted(d for d in dirs if not d.startswith('.'))

        if root == deployment_dir:
            # Files here are already at the correct level — leave them alone
            continue

        rel_parts = list(root.relative_to(deployment_dir).parts)
        for name in sorted(filenames):
            fp = root / name
            if is_target(fp):
                results.append((fp, rel_parts))

    return results


# ── Destination resolution ────────────────────────────────────────────────────

# Anything outside this becomes '_' in a rename prefix. A prefix is the only way a
# FOLDER name can reach a FILE name, and folder names are typed by hand in the field:
# otoño 2025 arrived with `M 11`, `M 6` and `M17 (TC20)` under the station folders.
# A space or a bracket in a filename survives every later join and has to be quoted by
# every tool that touches it, so it is stripped once, here, at the only point where
# one could be introduced.
_UNSAFE_IN_PREFIX = re.compile(r'[^A-Za-z0-9]+')


def prefix_candidates(rel_parts: list) -> list:
    """Rename prefixes to try, in order, for a file whose flat name is taken.

    The DCIM folder alone comes first. It is the component that actually
    distinguishes two same-named frames — `100EK113` vs `101EK113` — whereas any
    folder above it is constant within the deployment and so adds nothing. Otoño 2026
    had no level above it and produced clean `102EK113_` prefixes; otoño 2025 arrived
    with a grid folder in between, and joining the whole path would have turned
    CT14's 28 collisions into `M 11_101EK113_01160002.JPG` — a space imported into 28
    filenames to disambiguate nothing.

    The full path is kept as a second candidate for the case the first cannot
    resolve: two DCIM folders of the same name under different parents.
    """
    seen, out = set(), []
    for raw in (rel_parts[-1], '_'.join(rel_parts)):
        prefix = _UNSAFE_IN_PREFIX.sub('_', raw).strip('_')
        if prefix and prefix not in seen:
            seen.add(prefix)
            out.append(prefix)
    return out


def resolve_dest(
    deployment_dir: Path,
    rel_parts: list,
    filename: str,
    claimed: set = None,
) -> tuple:
    """
    Determine where src should land inside deployment_dir.

    Strategy:
      1. Try the simple flat name (deployment_dir / filename).
      2. If that name is taken, prefix the DCIM folder the file came from:
         "100EK113_01190313.JPG". See prefix_candidates().
      3. If every prefixed name is also taken, append a numeric counter until a
         free name is found.

    Returns (dest: Path, action: str) where action ∈ {'moved', 'renamed'}.

    NOTHING IS EVER SKIPPED. An earlier version treated same-name + same-size as
    a duplicate and dropped the file. That is exactly the signature a
    reset-clock camera produces: once its RTC reverts to 2017-01-01 it re-emits
    `0101xxxx` filenames in every subsequent DCIM folder, and two such frames can
    easily share a byte size. Otoño 2026 CT_14 carries 24 real collisions that
    survived only because their sizes happened to differ
    (`102EK113_0119xxxx.JPG`); a same-size sibling would have been deleted
    leaving nothing but a log line. A duplicated image is a harmless nuisance —
    `camtrap/observations.py` keys on (campaign, camera_num, file_name), so both
    rows survive and can be compared later. A discarded image is unrecoverable.

    Re-running flatten cannot reintroduce the old duplicate case: `move_file`
    removes the source, so a name clash on a later pass is always a genuinely
    different file.

    `claimed` holds destinations already handed out during this run. Without it a
    --dry-run sees an untouched disk and reports zero renames for a deployment that
    would in fact rename dozens — the opposite of what a dry run is for.
    """
    claimed = claimed if claimed is not None else set()

    def taken(p: Path) -> bool:
        return p.exists() or p in claimed

    simple_dest = deployment_dir / filename
    if not taken(simple_dest):
        return simple_dest, 'moved'

    # Name taken — disambiguate with the DCIM folder the file came from
    prefixed_dest = None
    for prefix in prefix_candidates(rel_parts):
        prefixed_dest = deployment_dir / f'{prefix}_{filename}'
        if not taken(prefixed_dest):
            return prefixed_dest, 'renamed'

    # Every prefixed name taken too — append a counter
    stem, ext = prefixed_dest.stem, prefixed_dest.suffix
    counter = 2
    while True:
        candidate = deployment_dir / f"{stem}_{counter}{ext}"
        if not taken(candidate):
            return candidate, 'renamed'
        counter += 1


def count_flat_files(deployment_dir: Path) -> int:
    """Target files sitting directly in deployment_dir — no recursion.

    Used for the conservation check: after flattening, this must equal what it
    was before plus every file we moved in. See process_deployment().
    """
    return sum(
        1 for p in deployment_dir.iterdir()
        if p.is_file() and is_target(p)
    )


# ── Per-deployment processing ─────────────────────────────────────────────────

def process_deployment(
    deployment_dir: Path,
    files: list,
    dry_run: bool,
    manifest_rows: list,
) -> dict:
    """
    Move all collected files into deployment_dir, resolve conflicts, clean up.

    `files` is the list returned by collect_subdir_files().
    Appends one manifest row per file to manifest_rows in-place — the row is
    recorded BEFORE the move, because src.stat() is unavailable afterwards.
    Returns a summary dict; `lost` is non-zero only if the conservation check
    fails, which means files went missing and the caller must abort.
    """
    moved = renamed = 0
    n_before = count_flat_files(deployment_dir)

    # Files already sitting at deployment level have no DCIM folder to record, but
    # they must still appear: a manifest that silently omits them looks complete,
    # and camtrap/clocks.py would then order a partially-described deployment as if
    # it knew where every frame came from. Recorded with an empty dcim_folder so the
    # gap is explicit and clocks.py can fail closed on it.
    for existing in sorted(deployment_dir.iterdir()):
        if not (existing.is_file() and is_target(existing)):
            continue
        stat = existing.stat()
        manifest_rows.append({
            'deployment': deployment_dir.name,
            'dcim_folder': '',
            'original_name': existing.name,
            'original_relpath': existing.name,
            'flat_name': existing.name,
            'size_bytes': stat.st_size,
            'mtime': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'action': 'already_flat',
        })

    claimed: set = set()
    for src, rel_parts in files:
        dest, action = resolve_dest(deployment_dir, rel_parts, src.name, claimed)
        claimed.add(dest)
        stat = src.stat()

        manifest_rows.append({
            'deployment': deployment_dir.name,
            # The DCIM folder is the whole point of this file: it is the only
            # surviving evidence of capture order once the tree is flat, because
            # the per-folder filename counter wraps at 999 and Timelapse2's
            # RelativePath keeps nothing but the deployment name.
            #
            # Only a CAMERA-created folder counts — see clocks.dcim_folder_key. A
            # folder a person made says nothing about capture order, and recording
            # it here would let clocks.py sort on it. The full path is not lost: it
            # is `original_relpath`, one column across.
            'dcim_folder': dcim_folder_key('/'.join(rel_parts)),
            'original_name': src.name,
            'original_relpath': str(src.relative_to(deployment_dir)),
            'flat_name': dest.name,
            'size_bytes': stat.st_size,
            'mtime': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'action': action,
        })

        if not dry_run:
            move_file(src, dest)

        if action == 'renamed':
            renamed += 1
            print(f"    {'(dry)' if dry_run else 'MOVE'}"
                  f"  [renamed]  {src.relative_to(deployment_dir)}"
                  f" → {dest.name}")
        else:
            moved += 1

    # Cleanup subdirectory tree (skip warnings in dry-run: files are still there
    # only because nothing was moved, not because of a real problem)
    lost = 0
    if not dry_run:
        n_after = count_flat_files(deployment_dir)
        lost = (n_before + len(files)) - n_after

        problems = cleanup_empty_dirs(deployment_dir, dry_run)
        for path, reason in problems:
            print(f"  WARNING: could not remove '{path.relative_to(deployment_dir)}': {reason}")

    return {
        'name': deployment_dir.name,
        'total': len(files),
        'moved': moved,
        'renamed': renamed,
        'lost': lost,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(
        description='Flatten camera-trap deployment folders for CamtrapDP.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Example:\n'
            '  python flatten_for_camtrapdp.py /data/MyProject\n'
            '  python flatten_for_camtrapdp.py /data/MyProject --dry-run\n'
        ),
    )
    parser.add_argument('root', help='Path to the DataPackage root folder')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Preview actions without moving or deleting anything',
    )
    parser.add_argument(
        '--check-stations', action='store_true',
        help=(
            'Accepted and ignored. The station-convention check has been fatal by '
            'default since 2026-08-14; the flag survives so command lines and notes '
            'written before then keep working instead of dying on an unknown option.'
        ),
    )
    parser.add_argument(
        '--check-export', nargs='?', const='auto', default=None, metavar='CSV',
        help=(
            'Validate a Timelapse2 export against the full-category rule and refuse '
            'to proceed if it fails: `human` or `vehicle` must appear (Camtrap DP\'s '
            'vocabulary, not MegaDetector\'s `person`), or the campaign must carry an '
            'export_gate_override.txt. With no value, looks '
            f'for {exports.TOTAL_EXPORT_FILENAME} in the DataPackage root. Note that '
            'at first flatten the export does not exist yet — this is for the '
            're-flatten and re-export pass, and `python -m camtrap.exports <csv>` '
            'runs the same check standalone the moment an export is made.'
        ),
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        sys.exit(f"ERROR: '{root}' is not a directory.")

    # ── Discover deployments ──────────────────────────────────────────────────
    deployments = sorted(
        p for p in root.iterdir()
        if p.is_dir() and not p.name.startswith('.')
    )
    if not deployments:
        sys.exit("No deployment folders found in the specified root.")

    # ── Station-name convention ───────────────────────────────────────────────
    # A folder like `100EK113` (an unrenamed SD-card directory) silently became an
    # unmappable deployment and cost 252 rows of camera 5 in the 2025 annual report.
    # Catching it here — before Timelapse2 ever sees it — is the cheap fix.
    # Fatal by default since 2026-08-14. It used to warn unless --check-stations was
    # passed, which left the WEAKER guard on the older failure: `100EK113` reaching
    # Timelapse2 as a deployment cost 252 rows of camera 5 from the 2025 report for a
    # year. Its sibling — a station folder nested inside another — refuses outright,
    # and both are the same question, "is this folder the camera we think it is?".
    # Two severities for one question is how the cheaper one gets skipped.
    offenders = [d.name for d in deployments if not stations.is_canonical(d.name)]
    if offenders:
        sys.exit(
            f"\nERROR: {len(offenders)} folder(s) do not follow the station "
            f"convention ({stations.CANONICAL_PATTERN}):\n"
            + "\n".join(f"    {name}" for name in offenders)
            + "\n  Rename them to CT01..CT27 before flattening — Timelapse2 takes its "
              "Deployments\n  column straight from these folder names, and a folder "
              "it cannot map becomes\n  a deployment that silently belongs to no camera."
        )

    # ── Export gate ───────────────────────────────────────────────────────────
    # The same rule ingest enforces, available here so a bad export is caught on the
    # Windows box while Timelapse2 is still open, rather than a week later at
    # ingest. Advisory when an export merely happens to be lying in the root;
    # fatal when --check-export asks for it.
    export_path = (
        root / exports.TOTAL_EXPORT_FILENAME if args.check_export in (None, 'auto')
        else Path(args.check_export)
    )
    if args.check_export is not None or export_path.exists():
        fatal = args.check_export is not None
        if not export_path.exists():
            msg = f"no export to check at '{export_path}'"
            if fatal:
                sys.exit(f'ERROR: {msg}')
            print(f'  (skipped export check: {msg})')
        else:
            import pandas as pd   # local: the flatten path itself needs no pandas
            df = pd.read_csv(
                export_path, dtype=str, keep_default_na=False, low_memory=False,
            )
            try:
                audit = exports.require_full_category(
                    df, source=str(export_path), override_dir=export_path.parent,
                )
            except exports.ExportGateError as exc:
                if fatal:
                    sys.exit(f'ERROR: {exc}')
                print(f'WARNING: {exc}\n'
                      f'  (run with --check-export to make this fatal)\n')
            else:
                print(f'Export check OK: {export_path.name} — {audit.verdict}\n')

    # ── Count files per deployment ────────────────────────────────────────────
    deploy_files: dict = {}
    total_files = 0
    for dep in deployments:
        files = collect_subdir_files(dep)
        deploy_files[dep] = files
        total_files += len(files)

    # ── Print discovery summary ───────────────────────────────────────────────
    print(f"\nDataPackage root : {root}")
    print(f"Deployments found: {len(deployments)}")
    print()
    col_w = max(len(dep.name) for dep in deployments)
    print(f"  {'Deployment':<{col_w}}  Files to move")
    print(f"  {'-' * col_w}  -------------")
    for dep in deployments:
        print(f"  {dep.name:<{col_w}}  {len(deploy_files[dep]):>5}")
    print(f"\n  Total files to process: {total_files}")

    if total_files == 0:
        print("\nNothing to do — all files are already at the deployment level.")
        return 0

    # ── Confirm (skip in dry-run) ─────────────────────────────────────────────
    if args.dry_run:
        print("\n[DRY RUN — no files will be moved or deleted]\n")
    else:
        print()
        try:
            answer = input("Proceed? [y/N] ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            sys.exit(0)
        if answer != 'y':
            print("Aborted.")
            sys.exit(0)
        print()

    # ── Process each deployment ───────────────────────────────────────────────
    # The manifest is appended after every deployment rather than written once at
    # the end: if the run dies half-way (or is interrupted), the moves already made
    # must still be described, or their capture order is lost for good.
    manifest_path = root / (
        f'{Path(DCIM_MANIFEST_FILENAME).stem}_dryrun.csv' if args.dry_run
        else DCIM_MANIFEST_FILENAME
    )
    manifest_exists = manifest_path.exists()
    if manifest_exists:
        print(f"Manifest already present — appending → {manifest_path}\n")

    summaries: list = []
    aborted = False

    try:
        with open(manifest_path, 'a', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=DCIM_MANIFEST_COLUMNS)
            if not manifest_exists:
                writer.writeheader()

            for dep in deployments:
                files = deploy_files[dep]
                if not files:
                    continue
                print(f"── {dep.name} ({len(files)} file(s))")

                manifest_rows: list = []
                summary = process_deployment(dep, files, args.dry_run, manifest_rows)
                writer.writerows(manifest_rows)
                fh.flush()

                summaries.append(summary)
                print(f"   moved={summary['moved']}  renamed={summary['renamed']}")

                # Conservation check — the whole reason resolve_dest no longer
                # skips anything. Stop before touching another deployment.
                if summary['lost']:
                    print(
                        f"\nERROR: {dep.name} lost {summary['lost']} file(s): "
                        f"expected {count_flat_files(dep) + summary['lost']} flat "
                        f"files, found {count_flat_files(dep)}.\n"
                        f"  Nothing further will be processed. The manifest up to "
                        f"this point is at {manifest_path}.",
                        file=sys.stderr,
                    )
                    aborted = True
                    break
    except OSError as exc:
        sys.exit(f"ERROR: could not write manifest {manifest_path}: {exc}")

    # ── Print overall summary ─────────────────────────────────────────────────
    if summaries:
        print()
        print("── Summary " + "─" * 65)
        col_w = max(len(s['name']) for s in summaries)
        print(
            f"  {'Deployment':<{col_w}}  "
            f"{'Total':>6}  {'Moved':>6}  {'Renamed':>8}  {'Lost':>6}"
        )
        print(f"  {'-' * col_w}  " + "------  " * 4)
        for s in summaries:
            print(
                f"  {s['name']:<{col_w}}  "
                f"{s['total']:>6}  "
                f"{s['moved']:>6}  "
                f"{s['renamed']:>8}  "
                f"{s['lost']:>6}"
            )
        total_moved   = sum(s['moved']   for s in summaries)
        total_renamed = sum(s['renamed'] for s in summaries)
        total_lost    = sum(s['lost']    for s in summaries)
        print(f"  {'TOTAL':<{col_w}}  "
              f"{sum(s['total'] for s in summaries):>6}  "
              f"{total_moved:>6}  "
              f"{total_renamed:>8}  "
              f"{total_lost:>6}")

    print(f"\nManifest → {manifest_path}")
    if not args.dry_run:
        print(
            "  Copy it into the campaign folder beside the Timelapse2 export "
            "(data/campaigns/<campaign>/) — it is the only record of which DCIM "
            "folder each frame came from, and camtrap/clocks.py needs it to "
            "establish capture order."
        )

    if aborted:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
