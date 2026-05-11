"""
fix_unicode_filenames.py
Fixes filenames with NFD Unicode encoding (from Synology/Linux/macOS sync)
by renaming them to NFC form (Windows-compatible).

NFD splits accented chars into base + combining accent (invisible),
causing some apps to fail reading those files. NFC merges them back
into a single character — visually identical, universally compatible.

Usage:
    python fix_unicode_filenames.py /path/to/root              # dry run
    python fix_unicode_filenames.py /path/to/root --apply      # actually rename
"""

import argparse
import os
import sys
import unicodedata


def needs_normalization(name: str) -> bool:
    return unicodedata.normalize('NFC', name) != name


def fix_names(root: str, dry_run: bool) -> tuple[int, int]:
    renamed = 0
    errors = 0
    mode = '[DRY RUN] Would rename' if dry_run else 'Renamed'

    # Bottom-up: process files before their parent dirs,
    # so renaming a dir doesn't break paths to files inside it.
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        # Files
        for name in filenames:
            nfc_name = unicodedata.normalize('NFC', name)
            if nfc_name != name:
                old = os.path.join(dirpath, name)
                new = os.path.join(dirpath, nfc_name)
                print(f'{mode}: {old!r}')
                print(f'         -> {nfc_name!r}')
                if not dry_run:
                    try:
                        os.rename(old, new)
                        renamed += 1
                    except Exception as e:
                        print(f'  ERROR: {e}')
                        errors += 1
                else:
                    renamed += 1

        # Directories
        for name in dirnames:
            nfc_name = unicodedata.normalize('NFC', name)
            if nfc_name != name:
                old = os.path.join(dirpath, name)
                new = os.path.join(dirpath, nfc_name)
                print(f'{mode} dir: {old!r}')
                print(f'             -> {nfc_name!r}')
                if not dry_run:
                    try:
                        os.rename(old, new)
                        renamed += 1
                    except Exception as e:
                        print(f'  ERROR: {e}')
                        errors += 1
                else:
                    renamed += 1

    return renamed, errors


def main():
    parser = argparse.ArgumentParser(
        description='Rename files with NFD Unicode encoding to NFC form.',
    )
    parser.add_argument('root', help='Root directory to scan recursively')
    parser.add_argument(
        '--apply', action='store_true',
        help='Actually rename files (default is a dry run that prints what would change)',
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f'ERROR: Directory not found: {args.root}')

    dry_run = not args.apply

    print('=' * 70)
    print('Unicode NFD -> NFC Filename Fixer')
    print('=' * 70)
    print(f'Root:    {args.root}')
    print(f'Mode:    {"DRY RUN (no changes will be made)" if dry_run else "APPLY (files will be renamed)"}')
    print()

    count, errors = fix_names(args.root, dry_run)

    print()
    print('=' * 70)
    if count == 0:
        print('No files with NFD encoding found. Nothing to do.')
    elif dry_run:
        print(f'Found {count} item(s) that need renaming.')
        print()
        print('Run with --apply to actually rename them:')
        print(f'  python fix_unicode_filenames.py {args.root!r} --apply')
    else:
        print(f'Done. Renamed {count} item(s). Errors: {errors}')


if __name__ == '__main__':
    main()
