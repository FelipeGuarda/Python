"""Cross-check two contact workbooks: who is missing, and what is new.

    python scripts/excel_crosscheck.py registrations.xlsx master.xlsx

Writes an .xlsx with four sheets — summary, only_in_left, only_in_right and
changes — next to the first input unless -o says otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.rosters import load_roster  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two Excel contact lists by email address.",
        epilog="Column and header-row detection are automatic; use --key only "
               "when a file has several address columns and the wrong one wins.",
    )
    parser.add_argument("left", type=Path, help="first workbook (e.g. this event's registrations)")
    parser.add_argument("right", type=Path, help="second workbook (e.g. the master list)")
    parser.add_argument("-o", "--output", type=Path, help="output .xlsx (default: alongside LEFT)")
    parser.add_argument("--key", help="force the email column name in both files")
    parser.add_argument("--left-sheet", help="sheet name in LEFT (default: first usable)")
    parser.add_argument("--right-sheet", help="sheet name in RIGHT (default: first usable)")
    args = parser.parse_args()

    for path in (args.left, args.right):
        if not path.exists():
            parser.error(f"file not found: {path}")

    left = load_roster(args.left, key=args.key, sheet=args.left_sheet)
    right = load_roster(args.right, key=args.key, sheet=args.right_sheet)

    print(f"{args.left.name}: {len(left)} contacts "
          f"(sheet {left.sheet!r}, column {left.key_column!r})")
    print(f"{args.right.name}: {len(right)} contacts "
          f"(sheet {right.sheet!r}, column {right.key_column!r})")

    result = left.compare(right)
    output = args.output or args.left.with_name(f"{args.left.stem}_vs_{args.right.stem}.xlsx")
    result.to_excel(output)

    print()
    for _, line in result.summary.iterrows():
        print(f"  {line['metric']:<22} {line['value']}")
    print(f"\nWritten to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
