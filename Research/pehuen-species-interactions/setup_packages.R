# setup_packages.R
# ─────────────────────────────────────────────────────────────────────────────
# CRAN fallback installer.
# Run this ONLY if `conda env create -f environment.yml` fails for some package.
# Installs the same packages from CRAN instead.
# ─────────────────────────────────────────────────────────────────────────────

pkgs <- c(
  "camtrapR",
  "overlap",
  "activity",
  "circular",
  "ggplot2",
  "dplyr",
  "tidyr",
  "stringr",
  "lubridate",
  "readr",
  "here",
  "patchwork",
  "sf",
  "scales"
)

install.packages(pkgs, repos = "https://cloud.r-project.org")
