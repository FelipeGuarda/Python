# setup_packages.R
# ─────────────────────────────────────────────────────────────────────────────
# CRAN fallback installer.
# Run this ONLY if `conda env create -f environment.yml` fails for some package.
# Installs the same packages from CRAN instead.
# ─────────────────────────────────────────────────────────────────────────────

pkgs <- c(
  "camtrapR",
  # Required: R/01_load_data.R reads camera-traps' observations.parquet.
  # nanoparquet rather than arrow deliberately -- it is tiny and pulls in no
  # Arrow runtime. arrow is accepted if it is already installed.
  "nanoparquet",
  "jsonlite",
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
