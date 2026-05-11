"""Shared filesystem helpers for the setup/ scripts.

Used by flatten_for_camtrapdp.py and merge_videos_to_fotos.py. resolve_dest()
is intentionally kept per-script because the two consumers have divergent
action-string conventions and prefix logic.
"""

import os
import shutil
from pathlib import Path


TARGET_EXTENSIONS: frozenset = frozenset({
    '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff',
    '.mp4', '.avi', '.mov', '.mpg', '.mpeg', '.wmv', '.asf', '.mkv',
})

IGNORED_NAMES: frozenset = frozenset({
    'thumbs.db', 'desktop.ini', '.ds_store',
})


def is_target(path: Path) -> bool:
    """Return True if this file should be moved (correct extension, not hidden/system)."""
    name = path.name
    if name.startswith('.'):
        return False
    if name.lower() in IGNORED_NAMES:
        return False
    return path.suffix.lower() in TARGET_EXTENSIONS


def move_file(src: Path, dest: Path) -> None:
    """Move src → dest, preserving original access/modification timestamps."""
    st = src.stat()
    shutil.move(str(src), str(dest))
    os.utime(dest, (st.st_atime, st.st_mtime))


def cleanup_empty_dirs(directory: Path, dry_run: bool) -> list[tuple[Path, str]]:
    """Walk bottom-up through subdirectories, removing each empty one.

    Returns (path, reason) for directories that could not be removed —
    typically because they contain non-target leftover files.
    """
    problems: list[tuple[Path, str]] = []
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
