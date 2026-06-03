# Render the annual report to DOCX (and optionally PDF) on Windows.
#
# Usage (PowerShell, from inside source_code_CT_2025):
#   .\render.ps1           → produces informe_anual_2025.docx
#   .\render.ps1 pdf       → also produces informe_anual_2025.pdf
#
# Requires pandoc.  Install once with:
#   winget install --id JohnMacFarlane.Pandoc -e
#   (or via Chocolatey: choco install pandoc)
#
# Notes:
# - Images referenced as figures/NN_*.png are embedded automatically.
# - For PDF output pandoc needs a LaTeX engine, e.g. MiKTeX:
#   winget install --id MiKTeX.MiKTeX -e
# - If you see "render.ps1 cannot be loaded because running scripts is
#   disabled", run PowerShell once as Administrator and execute:
#     Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

param(
    [string]$Format = ""
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$src     = "informe_anual_2025.md"
$outDocx = "informe_anual_2025.docx"
$outPdf  = "informe_anual_2025.pdf"

if (-not (Get-Command pandoc -ErrorAction SilentlyContinue)) {
    Write-Error "pandoc not found. Install it once with: winget install --id JohnMacFarlane.Pandoc -e"
    exit 1
}

pandoc $src `
    --from markdown `
    --to docx `
    --output $outDocx `
    --resource-path=.
Write-Host "→ $outDocx"

if ($Format -eq "pdf") {
    pandoc $src `
        --pdf-engine=xelatex `
        --output $outPdf `
        --resource-path=.
    Write-Host "→ $outPdf"
}
