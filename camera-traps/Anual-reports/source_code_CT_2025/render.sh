#!/usr/bin/env bash
# Render the annual report to DOCX (and optionally PDF) on Linux / macOS.
#
# Usage:
#   bash render.sh           → produces informe_anual_2025.docx
#   bash render.sh pdf       → also produces informe_anual_2025.pdf
#
# Requires pandoc.  Install once with:
#   sudo apt install pandoc                    (Debian / Ubuntu)
#   brew install pandoc                        (macOS, via Homebrew)
#
# Notes:
# - Images referenced as figures/NN_*.png are embedded automatically.
# - For PDF output pandoc needs a LaTeX engine
#   (`sudo apt install texlive-xetex` / `brew install --cask mactex-no-gui`).

set -euo pipefail

cd "$(dirname "$0")"

SRC="informe_anual_2025.md"
OUT_DOCX="informe_anual_2025.docx"
OUT_PDF="informe_anual_2025.pdf"

if ! command -v pandoc >/dev/null 2>&1; then
    echo "pandoc not found. Install it once with: sudo apt install pandoc (Linux) or brew install pandoc (macOS)" >&2
    exit 1
fi

pandoc "$SRC" \
    --from markdown \
    --to docx \
    --output "$OUT_DOCX" \
    --resource-path=.
echo "→ $OUT_DOCX"

if [[ "${1:-}" == "pdf" ]]; then
    pandoc "$SRC" \
        --pdf-engine=xelatex \
        --output "$OUT_PDF" \
        --resource-path=.
    echo "→ $OUT_PDF"
fi
