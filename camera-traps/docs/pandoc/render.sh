#!/usr/bin/env bash
# Genera el manual en Word desde docs/MANUAL-SALUD-DATOS.md.
#
# EL MANUAL QUE SE LEE ES EL QUE ESTE SCRIPT ESCRIBE:
#     MANUAL-SALUD-DATOS.docx   (en la raíz del repositorio)
#
# El otro .docx de esta carpeta, `plantilla-estilos-apaisado-NO-ES-EL-MANUAL.docx`,
# NO es un documento legible: es la plantilla de estilos de pandoc y su contenido es
# texto de muestra ("Title / Heading 1 / Body Text..."). Existe por una sola razón:
# lleva la página en horizontal, porque las tablas de vigilancia tienen cinco
# columnas y en vertical quedan ilegibles.
#
#   ./docs/pandoc/render.sh             -> MANUAL-SALUD-DATOS.docx en la raíz
#   ./docs/pandoc/render.sh /otra/ruta  -> ahí en cambio
#
# El .docx generado está en .gitignore: es un artefacto, la fuente es el Markdown.
#
# PDF no está automatizado: esta máquina no tiene motor LaTeX. Abra el Word y
# exporte desde ahí, o instale una distribución TeX y agregue
# `-o salida.pdf --pdf-engine=xelatex -V geometry:landscape`.
#
# NOTA SOBRE EL ÍNDICE: Word inserta el índice como campo. Si aparece vacío al
# abrir, seleccione todo (Ctrl+E) y presione F9 para actualizarlo.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
dest="${1:-$root}"
out="$dest/MANUAL-SALUD-DATOS.docx"

pandoc "$root/docs/MANUAL-SALUD-DATOS.md" \
  --reference-doc="$here/plantilla-estilos-apaisado-NO-ES-EL-MANUAL.docx" \
  --lua-filter="$here/drop-mermaid.lua" \
  --toc --toc-depth=3 \
  --metadata title="Salud de los Datos de Cámaras Trampa" \
  --metadata toc-title="Índice" \
  --metadata lang="es-CL" \
  -o "$out"

echo "escrito: $out"
echo "         (abra ESTE archivo; la plantilla de estilos no es el manual)"
