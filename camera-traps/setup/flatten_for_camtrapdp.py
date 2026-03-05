#!/usr/bin/env python3
"""
flatten_for_camtrapdp.py — Flatten camera-trap deployment folders for CamtrapDP.

Moves every image/video file from any depth of subfolder inside each deployment
up into the deployment folder itself, then removes the now-empty subdirectories.

Usage:
    python flatten_for_camtrapdp.py /path/to/DataPackage
    python flatten_for_camtrapdp.py /path/to/DataPackage --dry-run
"""

import argparse
import csv
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_EXTENSIONS: frozenset = frozenset({
    '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff',
    '.mp4', '.avi', '.mov', '.mpg', '.mpeg', '.wmv', '.asf', '.mkv',
})

IGNORED_NAMES: frozenset = frozenset({
    'thumbs.db', 'desktop.ini', '.ds_store',
})


# ── File helpers ──────────────────────────────────────────────────────────────

def is_target(path: Path) -> bool:
    """Return True if this file should be moved (correct extension, not hidden/system)."""
    name = path.name
    if name.startswith('.'):
        return False
    if name.lower() in IGNORED_NAMES:
        return False
    return path.suffix.lower() in TARGET_EXTENSIONS


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

def resolve_dest(
    deployment_dir: Path,
    rel_parts: list,
    filename: str,
    src: Path,
) -> tuple:
    """
    Determine where src should land inside deployment_dir.

    Strategy:
      1. Try the simple flat name (deployment_dir / filename).
      2. If that name exists and has the same size → duplicate, skip.
      3. If that name exists with a different size → prepend the subfolder
         path as a prefix: "part1_part2_filename.ext".
      4. If the prefixed name also conflicts, check for duplicate again;
         otherwise append a numeric counter until a free name is found.

    Returns (dest: Path, action: str) where action ∈
        {'moved', 'renamed', 'skipped_duplicate'}.
    """
    src_size = src.stat().st_size
    simple_dest = deployment_dir / filename

    if not simple_dest.exists():
        return simple_dest, 'moved'

    # Name clash at simple destination
    if simple_dest.stat().st_size == src_size:
        return simple_dest, 'skipped_duplicate'

    # Different file — build a prefixed name from the intermediate folder path
    prefix = '_'.join(rel_parts)
    prefixed_name = f"{prefix}_{filename}"
    prefixed_dest = deployment_dir / prefixed_name

    if not prefixed_dest.exists():
        return prefixed_dest, 'renamed'

    if prefixed_dest.stat().st_size == src_size:
        return prefixed_dest, 'skipped_duplicate'

    # Prefixed name also clashes with a different file — add numeric counter
    stem = Path(prefixed_name).stem
    ext = Path(prefixed_name).suffix
    counter = 2
    while True:
        candidate = deployment_dir / f"{stem}_{counter}{ext}"
        if not candidate.exists():
            return candidate, 'renamed'
        if candidate.stat().st_size == src_size:
            return candidate, 'skipped_duplicate'
        counter += 1


# ── Move helper ───────────────────────────────────────────────────────────────

def move_file(src: Path, dest: Path) -> None:
    """Move src → dest, restoring the original access/modification timestamps."""
    st = src.stat()
    shutil.move(str(src), str(dest))
    os.utime(dest, (st.st_atime, st.st_mtime))


# ── Cleanup ───────────────────────────────────────────────────────────────────

def cleanup_empty_dirs(deployment_dir: Path, dry_run: bool) -> list:
    """
    Walk bottom-up through all subdirectories of deployment_dir, removing
    each one that is completely empty.

    Returns a list of (Path, reason: str) for any directory that could not
    be removed, so the caller can warn the user.
    """
    problems = []
    for root_str, _dirs, _files in os.walk(deployment_dir, topdown=False):
        root = Path(root_str)
        if root == deployment_dir:
            continue

        try:
            contents = list(root.iterdir())
        except PermissionError as exc:
            problems.append((root, f"permission error: {exc}"))
            continue

        if not contents:
            if not dry_run:
                try:
                    root.rmdir()
                except OSError as exc:
                    problems.append((root, str(exc)))
            # In dry-run we don't report these as problems; they would be removed
        else:
            leftover_files = [p for p in contents if p.is_file()]
            if leftover_files:
                names = ', '.join(p.name for p in leftover_files[:5])
                more = f" (and {len(leftover_files) - 5} more)" if len(leftover_files) > 5 else ''
                problems.append((root, f"contains non-target file(s): {names}{more}"))

    return problems


# ── Per-deployment processing ─────────────────────────────────────────────────

def process_deployment(
    deployment_dir: Path,
    files: list,
    dry_run: bool,
    log_rows: list,
) -> dict:
    """
    Move all collected files into deployment_dir, resolve conflicts, clean up.

    `files` is the list returned by collect_subdir_files().
    Appends rows to log_rows in-place.
    Returns a summary dict.
    """
    moved = renamed = skipped = 0

    for src, rel_parts in files:
        dest, action = resolve_dest(deployment_dir, rel_parts, src.name, src)

        log_rows.append({
            'deployment': deployment_dir.name,
            'original_path': str(src),
            'new_path': str(dest),
            'action': action,
        })

        if action == 'skipped_duplicate':
            skipped += 1
            print(f"    SKIP  [duplicate]  {src.relative_to(deployment_dir)}")
        elif action == 'renamed':
            if not dry_run:
                move_file(src, dest)
            renamed += 1
            print(f"    {'(dry)' if dry_run else 'MOVE'}"
                  f"  [renamed]  {src.relative_to(deployment_dir)}"
                  f" → {dest.name}")
        else:  # 'moved'
            if not dry_run:
                move_file(src, dest)
            moved += 1

    # Cleanup subdirectory tree (skip warnings in dry-run: files are still there
    # only because nothing was moved, not because of a real problem)
    if not dry_run:
        problems = cleanup_empty_dirs(deployment_dir, dry_run)
        for path, reason in problems:
            print(f"  WARNING: could not remove '{path.relative_to(deployment_dir)}': {reason}")

    return {
        'name': deployment_dir.name,
        'total': len(files),
        'moved': moved,
        'renamed': renamed,
        'skipped': skipped,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
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
        return

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
    log_rows: list = []
    summaries: list = []

    for dep in deployments:
        files = deploy_files[dep]
        if not files:
            continue
        print(f"── {dep.name} ({len(files)} file(s))")
        summary = process_deployment(dep, files, args.dry_run, log_rows)
        summaries.append(summary)
        print(
            f"   moved={summary['moved']}  "
            f"renamed={summary['renamed']}  "
            f"skipped={summary['skipped']}"
        )

    # ── Print overall summary ─────────────────────────────────────────────────
    if summaries:
        print()
        print("── Summary " + "─" * 65)
        col_w = max(len(s['name']) for s in summaries)
        print(
            f"  {'Deployment':<{col_w}}  "
            f"{'Total':>6}  {'Moved':>6}  {'Renamed':>8}  {'Skipped':>8}"
        )
        print(f"  {'-' * col_w}  " + "------  " * 4)
        for s in summaries:
            print(
                f"  {s['name']:<{col_w}}  "
                f"{s['total']:>6}  "
                f"{s['moved']:>6}  "
                f"{s['renamed']:>8}  "
                f"{s['skipped']:>8}"
            )
        total_moved   = sum(s['moved']   for s in summaries)
        total_renamed = sum(s['renamed'] for s in summaries)
        total_skipped = sum(s['skipped'] for s in summaries)
        print(f"  {'TOTAL':<{col_w}}  "
              f"{total_files:>6}  "
              f"{total_moved:>6}  "
              f"{total_renamed:>8}  "
              f"{total_skipped:>8}")

    # ── Write CSV log ─────────────────────────────────────────────────────────
    if log_rows:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        mode_tag = '_dryrun' if args.dry_run else ''
        log_path = root / f"flatten_log_{ts}{mode_tag}.csv"
        try:
            with open(log_path, 'w', newline='', encoding='utf-8') as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=['deployment', 'original_path', 'new_path', 'action'],
                )
                writer.writeheader()
                writer.writerows(log_rows)
            print(f"\nLog written → {log_path}")
        except OSError as exc:
            print(f"\nWARNING: could not write log file: {exc}", file=sys.stderr)


if __name__ == '__main__':
    main()
