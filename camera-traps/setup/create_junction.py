"""
create_junction.py — Create a Windows directory junction (mklink /J).

A junction is a filesystem-level shortcut Windows treats as a real directory.
Useful when a tool can't handle deep paths or non-ASCII characters but the
source data lives somewhere awkward (e.g. Synology Drive shares).

Usage (Windows only):
    python create_junction.py --target "C:\\path\\with\\spaces" --link C:\\ADDAX\\foo
"""

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description='Create a Windows directory junction.')
    parser.add_argument('--target', required=True, help='Existing source directory')
    parser.add_argument('--link', required=True, help='Junction path to create')
    args = parser.parse_args()

    print(f'Target exists: {os.path.isdir(args.target)}')
    print(f'Link path:     {args.link}')
    print()

    if not os.path.isdir(args.target):
        sys.exit(f'ERROR: target directory not found: {args.target}')

    if os.path.exists(args.link):
        print('Junction already exists.')
    else:
        cmd = f'mklink /J "{args.link}" "{args.target}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='oem')
        print('stdout:', result.stdout)
        print('stderr:', result.stderr)
        print('Return code:', result.returncode)

    print()
    print('Verifying...')
    print(f'Junction accessible: {os.path.isdir(args.link)}')
    print(f'File count: {sum(len(f) for _, _, f in os.walk(args.link))}')


if __name__ == '__main__':
    main()
