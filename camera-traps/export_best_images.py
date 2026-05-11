"""
Export top-N best images per species and per station for all campaigns.

Auto-discovers campaigns: any subfolder of CAMPAIGNS_BASE that contains
new_labeled_data_reviewed.csv (either directly or in a Fotos/ subfolder).

Output structure:
    exports/
      <campaign_name>/
        species/<common_latin_slug>/   ← top N globally per species
        stations/<station_name>/       ← top M per station (any species, for map popups)

Filenames: {station}_{original_filename}.jpg  (fully traceable to source)

Usage:
    conda activate species-classifier
    python export_best_images.py
"""

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from classify_campaign.species import (
    common_names as _load_common_names,
    spanish_to_latin as _load_spanish_to_latin,
)


@dataclass
class AnimalRow:
    """CSV row plus derived fields. Replaces in-place mutation of row dicts."""
    row: dict
    md_conf: float
    src: Path

# ── Config ────────────────────────────────────────────────────────────────────

# Campaigns root: pass --campaigns-base on the CLI or set FMA_CAMPAIGNS_BASE.
# Typical value on the Windows dev box:
#   C:\Users\USUARIO\SynologyDrive\2. Camaras trampa (SC)\SynologyDrive\
#     DATOS_GRILLA CÁMARAS TRAMPA\2. CAMPAÑAS DE RECOLECCION DE IMAGENES

EXPORT_DIR    = Path(__file__).resolve().parent / "exports"
TOP_N_SPECIES = 5   # best images per species (global, for sharing / reports)
TOP_N_STATION = 3   # best images per station (any species, for map popups)
SKIP_SPECIES  = {"", "No reconocible", "No es un animal"}

# Scientific name → Spanish common name. Canonical source: data-pipeline/species.yaml.
# Used to build directory names {common_slug}_{latin_slug}; unknowns get _UNKNOWN_.
SPECIES_COMMON_NAMES: dict[str, str] = _load_common_names()

# Lowercased Spanish common name (or alias) → scientific name. Canonical source:
# data-pipeline/species.yaml. Used to recover scientificName for rows reviewed via
# "Otro (especificar)" where the app couldn't fill in a latin name at review time.
SPANISH_TO_LATIN: dict[str, str] = _load_spanish_to_latin()


# ── Helpers ───────────────────────────────────────────────────────────────────

def slugify(name: str) -> str:
    # NFD-normalise so accented forms slug identically to their stripped variants
    # (Guiña → Guina, Pudú → Pudu). Keeps directory names stable across the
    # accent-consolidation of SPECIES_COMMON_NAMES.
    stripped = "".join(c for c in unicodedata.normalize("NFD", name) if not unicodedata.combining(c))
    return re.sub(r"[^a-zA-Z0-9]+", "_", stripped).strip("_")


def species_dir_name(scientific_name: str) -> str:
    common = SPECIES_COMMON_NAMES.get(scientific_name)
    latin_slug = slugify(scientific_name)
    if common is None:
        return f"_UNKNOWN_{latin_slug}"
    return f"{slugify(common)}_{latin_slug}"


def row_fp(row: dict) -> str:
    """Return the relative file path (forward slashes). Handles empty filePath."""
    fp = row.get("filePath", "").strip()
    if not fp:
        rel   = row.get("RelativePath", "").strip()
        fname = row.get("File", "").strip()
        fp = rel + "/" + fname if rel and fname else ""
    return fp.replace("\\", "/")


def output_filename(row: dict) -> str:
    """Build a traceable output filename: {station}_{original_file} (lowercase ext)."""
    station  = row.get("RelativePath", "").strip().replace("\\", "/")
    fname    = row.get("File", "").strip()
    stem     = Path(fname).stem
    suffix   = Path(fname).suffix.lower() or ".jpg"
    # slugify the station part so dots/spaces don't cause filesystem issues
    station_slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", station)
    return f"{station_slug}_{stem}{suffix}"


def load_md_confidence(md_json_path: Path) -> dict[str, float]:
    """Return {normalised_relative_path → best animal (cat=1) confidence}."""
    with open(md_json_path, encoding="utf-8") as f:
        data = json.load(f)
    best: dict[str, float] = {}
    for img in data["images"]:
        key = img["file"].replace("\\", "/").lower()
        for det in img.get("detections", []):
            if det.get("category") == "1":
                conf = float(det.get("conf", 0.0))
                if conf > best.get(key, 0.0):
                    best[key] = conf
    return best


def find_campaigns(campaigns_base: Path) -> list[dict]:
    """
    Walk campaigns_base and return one entry per campaign that has a
    reviewed CSV and a MegaDetector JSON.

    Checks two locations per season folder:
      1. <season>/new_labeled_data_reviewed.csv          (Primavera layout)
      2. <season>/Fotos/new_labeled_data_reviewed.csv    (Otoño layout)
    """
    campaigns = []
    for season_dir in sorted(campaigns_base.iterdir()):
        if not season_dir.is_dir():
            continue
        for candidate in [season_dir, season_dir / "Fotos"]:
            csv_path = candidate / "new_labeled_data_reviewed.csv"
            json_path = candidate / "timelapse_recognition_file.json"
            if csv_path.exists() and json_path.exists():
                campaigns.append({
                    "label":        season_dir.name,   # e.g. "Otoño 2025"
                    "campaign_dir": candidate,
                    "reviewed_csv": csv_path,
                    "md_json":      json_path,
                })
                break   # don't double-count a season
    return campaigns


# ── Per-campaign export ───────────────────────────────────────────────────────

def export_campaign(campaign: dict) -> tuple[int, int]:
    """Export species/ and stations/ outputs. Returns (species_images, station_images)."""
    label        = campaign["label"]
    campaign_dir = campaign["campaign_dir"]

    print(f"\n{'-'*60}")
    print(f"Campaign : {label}")
    print(f"Dir      : {campaign_dir}")

    # ── Load data ─────────────────────────────────────────────────────────────
    md_conf = load_md_confidence(campaign["md_json"])
    print(f"  {len(md_conf)} images with animal detections in MD JSON")

    with open(campaign["reviewed_csv"], encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"  {len(rows)} total rows in reviewed CSV")

    # Rows written via "Otro (especificar)" have observationComments set but
    # scientificName left empty (no latin name was available at review time).
    # Try to resolve those via a case-insensitive Spanish name lookup.
    resolved = 0
    animal_rows: list[AnimalRow] = []
    for r in rows:
        if r.get("observationType", "").strip() != "animal":
            continue
        sci = r.get("scientificName", "").strip()
        if sci in SKIP_SPECIES:
            comment = r.get("observationComments", "").strip().lower()
            sci = SPANISH_TO_LATIN.get(comment, "")
            if sci:
                r["scientificName"] = sci
                resolved += 1
        if sci and sci not in SKIP_SPECIES:
            fp_key = row_fp(r).lower()
            animal_rows.append(AnimalRow(
                row=r,
                md_conf=md_conf.get(fp_key, 0.0),
                src=campaign_dir / row_fp(r),
            ))

    print(f"  {len(animal_rows)} animal rows with valid species"
          + (f"  ({resolved} resolved from comments)" if resolved else ""))

    no_md = sum(1 for ar in animal_rows if ar.md_conf == 0.0)
    if no_md:
        print(f"  Note: {no_md} rows had no MD detection entry (will sort last)")

    campaign_export = EXPORT_DIR / label
    species_total   = _export_species(animal_rows, campaign_export / "species")
    station_total   = _export_stations(animal_rows, campaign_export / "stations")
    return species_total, station_total


MAX_PX = 1000  # longest side in pixels for exported images


def _copy(src: Path, dst_dir: Path, filename: str) -> bool:
    """Resize src to MAX_PX (longest side) and save to dst_dir/filename. Returns True on success."""
    if not src.exists():
        print(f"    MISSING: {src.name}")
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img.thumbnail((MAX_PX, MAX_PX), Image.LANCZOS)
        img.save(dst_dir / filename, "JPEG", quality=85)
    return True


def _export_species(animal_rows: list[AnimalRow], out_base: Path) -> int:
    """Top N images per species globally, sorted by MD confidence desc."""
    by_species: dict[str, list[AnimalRow]] = defaultdict(list)
    for ar in animal_rows:
        by_species[ar.row["scientificName"].strip()].append(ar)

    total = 0
    for species in sorted(by_species):
        top = sorted(by_species[species], key=lambda ar: ar.md_conf, reverse=True)[:TOP_N_SPECIES]
        out_dir = out_base / species_dir_name(species)
        copied = sum(
            1 for ar in top
            if _copy(ar.src, out_dir, output_filename(ar.row))
        )
        total += copied
        print(f"  [species] {species:<40} {copied}/{len(top)}  "
              f"(best conf: {top[0].md_conf:.3f})")
    return total


def _export_stations(animal_rows: list[AnimalRow], out_base: Path) -> int:
    """Top M images per station (any species), sorted by MD confidence desc."""
    by_station: dict[str, list[AnimalRow]] = defaultdict(list)
    for ar in animal_rows:
        station = ar.row.get("RelativePath", "").strip().replace("\\", "/")
        if station:
            by_station[station].append(ar)

    total = 0
    for station in sorted(by_station):
        top = sorted(by_station[station], key=lambda ar: ar.md_conf, reverse=True)[:TOP_N_STATION]
        station_slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", station)
        out_dir = out_base / station_slug
        copied = sum(
            1 for ar in top
            if _copy(ar.src, out_dir, output_filename(ar.row))
        )
        total += copied
        print(f"  [station] {station:<25} {copied}/{len(top)}  "
              f"(best conf: {top[0].md_conf:.3f})")
    return total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export top-N best images per species and per station for all campaigns.",
    )
    parser.add_argument(
        "--campaigns-base",
        default=os.getenv("FMA_CAMPAIGNS_BASE"),
        help="Root directory containing campaign season folders. "
             "Falls back to FMA_CAMPAIGNS_BASE env var.",
    )
    args = parser.parse_args()

    if not args.campaigns_base:
        sys.exit(
            "ERROR: --campaigns-base or FMA_CAMPAIGNS_BASE env var required."
        )
    campaigns_base = Path(args.campaigns_base)
    if not campaigns_base.is_dir():
        sys.exit(f"ERROR: campaigns base directory not found: {campaigns_base}")

    campaigns = find_campaigns(campaigns_base)
    if not campaigns:
        print(f"No campaigns found in {campaigns_base}")
        print("Each campaign folder must contain new_labeled_data_reviewed.csv "
              "and timelapse_recognition_file.json (at root or in Fotos/).")
        return

    print(f"Found {len(campaigns)} campaign(s): {[c['label'] for c in campaigns]}")
    print(f"Export dir: {EXPORT_DIR}")

    grand_species = grand_stations = 0
    for campaign in campaigns:
        s, st = export_campaign(campaign)
        grand_species  += s
        grand_stations += st

    print(f"\n{'='*60}")
    print(f"Species images exported : {grand_species}")
    print(f"Station images exported : {grand_stations}")
    print(f"Export dir              : {EXPORT_DIR}")


if __name__ == "__main__":
    main()
