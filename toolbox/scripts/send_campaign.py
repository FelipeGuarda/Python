"""Send a personalised campaign from a contact workbook.

Preview first — this opens a browser and sends nothing:

    python scripts/send_campaign.py lista.xlsx \
        --template fondo_fma_resultado.html.j2 \
        --subject "Aviso proceso de selección Fondo FMA 2025-2026"

Then add --send to deliver. Without it, the script is always a dry run.
Recipients already in the ledger are skipped, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.mailer import TEMPLATE_DIR, Campaign  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Send templated bulk mail from an Excel list.")
    parser.add_argument("excel", type=Path, help="workbook holding the recipients")
    parser.add_argument("--template", required=True, help=f"template filename inside {TEMPLATE_DIR.name}/")
    parser.add_argument("--subject", required=True, help="subject line; may use {{ }} placeholders")
    parser.add_argument("--attach", type=Path, action="append", default=[], help="file to attach (repeatable)")
    parser.add_argument("--inline-image", type=Path, action="append", default=[],
                        help="image embedded in the body as cid:FILENAME (repeatable)")
    parser.add_argument("--key", help="force the email column name")
    parser.add_argument("--sheet", help="sheet name (default: first usable)")
    parser.add_argument("--preview", type=int, default=3, metavar="N",
                        help="how many messages to render for preview (default: 3)")
    parser.add_argument("--send", action="store_true",
                        help="actually deliver. Omit to dry-run.")
    parser.add_argument("--throttle", type=float, default=2.0,
                        help="seconds between sends (default: 2.0)")
    args = parser.parse_args()

    if not args.excel.exists():
        parser.error(f"file not found: {args.excel}")
    if not (TEMPLATE_DIR / args.template).exists():
        available = ", ".join(sorted(p.name for p in TEMPLATE_DIR.glob("*.j2")))
        parser.error(f"no template {args.template!r}. Available: {available or 'none'}")
    for path in [*args.attach, *args.inline_image]:
        if not path.exists():
            parser.error(f"file not found: {path}")

    campaign = Campaign.from_excel(
        args.excel,
        template=args.template,
        subject=args.subject,
        key=args.key,
        sheet=args.sheet,
        attachments=args.attach,
        inline_images=args.inline_image,
    )
    print(f"{len(campaign.roster)} recipients from {args.excel.name} "
          f"(sheet {campaign.roster.sheet!r}, column {campaign.roster.key_column!r})")

    if not args.send:
        campaign.preview(args.preview)

    campaign.send(confirm=args.send, throttle=args.throttle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
