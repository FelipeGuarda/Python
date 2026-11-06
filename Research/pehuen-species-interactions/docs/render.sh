#!/usr/bin/env bash
# Render the internal status document to Word. Markdown is the source of truth:
# edits made in Word are merged back into the .md by hand, then re-rendered.
# Prerequisite (once): sudo apt install pandoc
set -euo pipefail
cd "$(dirname "$0")"
pandoc estado-analisis-2026-11.md --to docx --resource-path=. \
  -o estado-analisis-2026-11.docx
echo "Wrote docs/estado-analisis-2026-11.docx"
pandoc sintesis-estado-analisis-2026-11.md --to docx --resource-path=. \
  -o sintesis-estado-analisis-2026-11.docx
echo "Wrote docs/sintesis-estado-analisis-2026-11.docx"
