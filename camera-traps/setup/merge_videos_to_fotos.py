#!/usr/bin/env python3
"""
merge_videos_to_fotos.py — Flatten video directories and merge into Fotos/.

Two-phase operation on a campaign directory that contains both Fotos/ and Videos/:

  Phase 1 — Flatten: within each Videos/CTxx/, collapse memory-card (Mxx) and
            DCIM-like (100EK113) subdirectories so all files sit directly in
            Videos/CTxx/.

  Phase 2 — Merge:   move every file from Videos/CTxx/ into the matching
            Fotos/CTxx/ directory.

Also cleans up empty subdirectories left behind in both Videos/ and Fotos/.

Usage:
    python merge_videos_to_fotos.py /path/to/campaign
    python merge_videos_to_fotos.py /path/to/campaign --dry-run
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
    name = path.name
    if name.startswith('.'):
        return False
    if name.lower() in IGNORED_NAMES:
        return False
    return path.suffix.lower() in TARGET_EXTENSIONS


def move_file(src: Path, dest: Path) -> None:
    """Move src → dest, preserving original timestamps."""
    st = src.stat()
    shutil.move(str(src), str(dest))
    os.utime(dest, (st.st_atime, st.st_mtime))


def resolve_dest(dest_dir: Path, filename: str, src: Path, prefix_parts: list[str] = None) -> tuple[Path, str]:
    """
    Determine where src should land inside dest_dir.

    Returns (dest_path, action) where action ∈ {'move', 'rename', 'skip_duplicate'}.
    """
    src_size = src.stat().st_size
    simple = dest_dir / filename

    if not simple.exists():
        return simple, 'move'

    if simple.stat().st_size == src_size:
        return simple, 'skip_duplicate'

    # Name clash — build prefixed name
    prefix = '_'.join(prefix_parts) if prefix_parts else 'dup'
    prefixed_name = f"{prefix}_{filename}"
    prefixed = dest_dir / prefixed_name

    if not prefixed.exists():
        return prefixed, 'rename'
    if prefixed.stat().st_size == src_size:
        return prefixed, 'skip_duplicate'

    # Numeric fallback
    stem = Path(prefixed_name).stem
    ext = Path(prefixed_name).suffix
    counter = 2
    while True:
        candidate = dest_dir / f"{stem}_{counter}{ext}"
        if not candidate.exists():
            return candidate, 'rename'
        if candidate.stat().st_size == src_size:
            return candidate, 'skip_duplicate'
        counter += 1


def cleanup_empty_dirs(directory: Path, dry_run: bool) -> list[tuple[Path, str]]:
    """Remove empty subdirectories bottom-up. Returns problems."""
    problems = []
    for root_str, _dirs, _files in os.walk(directory, topdown=False):
        root = Path(root_str)
        if root == directory:
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
        else:
            leftover = [p for p in contents if p.is_file()]
            if leftover:
                names = ', '.join(p.name for p in leftover[:5])
                more = f" (and {len(leftover) - 5} more)" if len(leftover) > 5 else ''
                problems.append((root, f"non-target file(s): {names}{more}"))
    return problems


# ── Phase 1: Flatten Videos ──────────────────────────────────────────────────

def flatten_videos(videos_dir: Path, dry_run: bool, log_rows: list) -> dict:
    """Flatten each Videos/CTxx/ so all files sit directly inside CTxx/."""
    cameras = sorted(
        p for p in videos_dir.iterdir()
        if p.is_dir() and not p.name.startswith('.')
    )

    stats = {'cameras': 0, 'moved': 0, 'renamed': 0, 'skipped': 0, 'already_flat': 0}

    for cam_dir in cameras:
        # Collect files that are in subdirectories (not already flat)
        files_to_flatten = []
        for root_str, dirs, filenames in os.walk(cam_dir):
            root = Path(root_str)
            dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
            if root == cam_dir:
                continue  # files already at camera level
            rel_parts = list(root.relative_to(cam_dir).parts)
            for name in sorted(filenames):
                fp = root / name
                if is_target(fp):
                    files_to_flatten.append((fp, rel_parts))

        if not files_to_flatten:
            stats['already_flat'] += 1
            continue

        stats['cameras'] += 1
        print(f"  {cam_dir.name}: flattening {len(files_to_flatten)} files")

        for src, rel_parts in files_to_flatten:
            dest, action = resolve_dest(cam_dir, src.name, src, prefix_parts=rel_parts)

            log_rows.append({
                'phase': 'flatten',
                'camera': cam_dir.name,
                'original_path': str(src),
                'new_path': str(dest),
                'action': action,
            })

            if action == 'skip_duplicate':
                stats['skipped'] += 1
                if not dry_run:
                    src.unlink()
                print(f"    SKIP [dup] {src.relative_to(cam_dir)}")
            elif action == 'rename':
                if not dry_run:
                    move_file(src, dest)
                stats['renamed'] += 1
                print(f"    {'(dry) ' if dry_run else ''}RENAME {src.relative_to(cam_dir)} → {dest.name}")
            else:
                if not dry_run:
                    move_file(src, dest)
                stats['moved'] += 1

        # Clean up empty subdirs
        if not dry_run:
            problems = cleanup_empty_dirs(cam_dir, dry_run)
            for path, reason in problems:
                print(f"    WARNING: {path.relative_to(cam_dir)}: {reason}")

    return stats


# ── Phase 2: Merge Videos → Fotos ────────────────────────────────────────────

def merge_into_fotos(videos_dir: Path, fotos_dir: Path, dry_run: bool, log_rows: list) -> dict:
    """Move all files from Videos/CTxx/ into Fotos/CTxx/."""
    cameras = sorted(
        p for p in videos_dir.iterdir()
        if p.is_dir() and not p.name.startswith('.')
    )

    stats = {'cameras': 0, 'moved': 0, 'renamed': 0, 'skipped': 0, 'no_files': 0}

    for cam_dir in cameras:
        files = sorted(f for f in cam_dir.iterdir() if f.is_file() and is_target(f))
        if not files:
            stats['no_files'] += 1
            continue

        dest_dir = fotos_dir / cam_dir.name
        if not dest_dir.exists():
            if not dry_run:
                dest_dir.mkdir(parents=True)
            print(f"  {cam_dir.name}: created Fotos/{cam_dir.name}/ (new)")

        stats['cameras'] += 1
        print(f"  {cam_dir.name}: merging {len(files)} files → Fotos/{cam_dir.name}/")

        for src in files:
            dest, action = resolve_dest(dest_dir, src.name, src, prefix_parts=['video'])

            log_rows.append({
                'phase': 'merge',
                'camera': cam_dir.name,
                'original_path': str(src),
                'new_path': str(dest),
                'action': action,
            })

            if action == 'skip_duplicate':
                stats['skipped'] += 1
                print(f"    SKIP [dup] {src.name}")
            elif action == 'rename':
                if not dry_run:
                    move_file(src, dest)
                stats['renamed'] += 1
                print(f"    {'(dry) ' if dry_run else ''}RENAME {src.name} → {dest.name}")
            else:
                if not dry_run:
                    move_file(src, dest)
                stats['moved'] += 1

    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(
        description='Flatten video directories and merge into Fotos/.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Example:\n'
            '  python merge_videos_to_fotos.py "C:\\path\\to\\Otoño 2025"\n'
            '  python merge_videos_to_fotos.py "C:\\path\\to\\Otoño 2025" --dry-run\n'
        ),
    )
    parser.add_argument('campaign', help='Path to the campaign root (must contain Videos/ and Fotos/)')
    parser.add_argument('--dry-run', action='store_true', help='Preview without moving files')
    args = parser.parse_args()

    campaign = Path(args.campaign).resolve()
    videos_dir = campaign / 'Videos'
    fotos_dir = campaign / 'Fotos'

    if not videos_dir.is_dir():
        sys.exit(f"ERROR: Videos/ not found in '{campaign}'")
    if not fotos_dir.is_dir():
        sys.exit(f"ERROR: Fotos/ not found in '{campaign}'")

    dry_tag = " [DRY RUN]" if args.dry_run else ""
    log_rows: list = []

    # ── Count totals ─────────────────────────────────────────────────────────
    total_video_files = sum(
        1 for cam in videos_dir.iterdir() if cam.is_dir()
        for f in cam.rglob('*') if f.is_file() and is_target(f)
    )
    video_cameras = sorted(p.name for p in videos_dir.iterdir() if p.is_dir() and not p.name.startswith('.'))

    print(f"\nCampaign: {campaign}")
    print(f"Videos/: {len(video_cameras)} cameras, {total_video_files} files")
    print(f"Fotos/:  {len(list(fotos_dir.iterdir()))} items")

    if total_video_files == 0:
        print("\nNo video files to process.")
        return

    # ── Confirmation ─────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n{'=' * 60}")
        print(f"DRY RUN — no files will be moved or deleted")
        print(f"{'=' * 60}\n")
    else:
        print(f"\nThis will:")
        print(f"  1. Flatten {total_video_files} video files inside Videos/CTxx/")
        print(f"  2. Move them into the matching Fotos/CTxx/ directories")
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

    # ── Phase 1: Flatten ─────────────────────────────────────────────────────
    print(f"── Phase 1: Flatten Videos/{dry_tag}")
    flatten_stats = flatten_videos(videos_dir, args.dry_run, log_rows)
    print(f"\n  Flatten summary: {flatten_stats['cameras']} cameras processed, "
          f"{flatten_stats['moved']} moved, {flatten_stats['renamed']} renamed, "
          f"{flatten_stats['skipped']} skipped, {flatten_stats['already_flat']} already flat")

    # ── Phase 2: Merge ───────────────────────────────────────────────────────
    print(f"\n── Phase 2: Merge Videos → Fotos{dry_tag}")
    merge_stats = merge_into_fotos(videos_dir, fotos_dir, args.dry_run, log_rows)
    print(f"\n  Merge summary: {merge_stats['cameras']} cameras, "
          f"{merge_stats['moved']} moved, {merge_stats['renamed']} renamed, "
          f"{merge_stats['skipped']} skipped")

    # ── Phase 3: Cleanup empty Fotos subdirs ─────────────────────────────────
    print(f"\n── Phase 3: Cleanup empty subdirectories{dry_tag}")
    fotos_problems = cleanup_empty_dirs(fotos_dir, args.dry_run)
    videos_problems = cleanup_empty_dirs(videos_dir, args.dry_run)
    for path, reason in fotos_problems + videos_problems:
        print(f"  WARNING: {path}: {reason}")
    if not fotos_problems and not videos_problems:
        print("  All empty subdirectories cleaned up.")

    # ── Write CSV log ────────────────────────────────────────────────────────
    if log_rows:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        mode_tag = '_dryrun' if args.dry_run else ''
        log_path = campaign / f"merge_videos_log_{ts}{mode_tag}.csv"
        try:
            with open(log_path, 'w', newline='', encoding='utf-8') as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=['phase', 'camera', 'original_path', 'new_path', 'action'],
                )
                writer.writeheader()
                writer.writerows(log_rows)
            print(f"\nLog written → {log_path}")
        except OSError as exc:
            print(f"\nWARNING: could not write log: {exc}", file=sys.stderr)

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"DONE{dry_tag}")
    grand_total = flatten_stats['moved'] + flatten_stats['renamed'] + merge_stats['moved'] + merge_stats['renamed']
    print(f"  Total file operations: {grand_total}")
    print(f"  Duplicates skipped: {flatten_stats['skipped'] + merge_stats['skipped']}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
