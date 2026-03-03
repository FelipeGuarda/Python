"""
fix_unicode_filenames.py
Fixes filenames with NFD Unicode encoding (from Synology/Linux/macOS sync)
by renaming them to NFC form (Windows-compatible).

NFD splits accented chars into base + combining accent (invisible),
causing some apps to fail reading those files. NFC merges them back
into a single character — visually identical, universally compatible.

Usage:
    python fix_unicode_filenames.py              # dry run (safe, no changes)
    python fix_unicode_filenames.py --apply      # actually rename files
"""

import os
import sys
import unicodedata

ROOT_DIR = r'C:/Users/USUARIO/SynologyDrive/2. Camaras trampa (SC)/SynologyDrive/DATOS_GRILLA CÁMARAS TRAMPA/2. CAMPAÑAS DE RECOLECCION DE IMAGENES/Primavera 2025'

DRY_RUN = '--apply' not in sys.argv


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
    if not os.path.isdir(ROOT_DIR):
        print(f'ERROR: Directory not found:\n  {ROOT_DIR}')
        sys.exit(1)

    print('=' * 70)
    print('Unicode NFD -> NFC Filename Fixer')
    print('=' * 70)
    print(f'Root:    {ROOT_DIR}')
    print(f'Mode:    {"DRY RUN (no changes will be made)" if DRY_RUN else "APPLY (files will be renamed)"}')
    print()

    count, errors = fix_names(ROOT_DIR, DRY_RUN)

    print()
    print('=' * 70)
    if count == 0:
        print('No files with NFD encoding found. Nothing to do.')
    elif DRY_RUN:
        print(f'Found {count} item(s) that need renaming.')
        print()
        print('Run with --apply to actually rename them:')
        print('  python fix_unicode_filenames.py --apply')
    else:
        print(f'Done. Renamed {count} item(s). Errors: {errors}')


if __name__ == '__main__':
    main()
