"""
Export top-N best images per species per campaign for the platform.

Selects images by MegaDetector confidence (highest conf = clearest animal shot).
Copies to:
    exports/species_images/{latin_name_slug}/{campaign_name}/best_01.jpg …

Usage:
    conda activate species-classifier
    python export_best_images.py
"""

import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

TOP_N = 5
SKIP_SPECIES = {"", "No reconocible"}

SYNOLOGY_BASE = (
    r"C:\Users\USUARIO\SynologyDrive\2. Camaras trampa (SC)"
    r"\SynologyDrive\DATOS_GRILLA CÁMARAS TRAMPA"
    r"\2. CAMPAÑAS DE RECOLECCION DE IMAGENES"
)

CAMPAIGNS = [
    {
        "name": "primavera_2025",
        "campaign_dir": SYNOLOGY_BASE + r"\Primavera 2025",
        "reviewed_csv": "new_labeled_data_reviewed.csv",
        "md_json":      "timelapse_recognition_file.json",
    },
    {
        "name": "otono_2025",
        "campaign_dir": SYNOLOGY_BASE + r"\Otoño 2025\Fotos",
        "reviewed_csv": "new_labeled_data_reviewed.csv",
        "md_json":      "timelapse_recognition_file.json",
    },
]

EXPORT_DIR = Path(__file__).parent / "exports" / "species_images"


# ── Helpers ───────────────────────────────────────────────────────────────────

def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")


def load_md_confidence(md_json_path: Path) -> dict[str, float]:
    """Return {normalised_file_path → best animal (cat=1) confidence}."""
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


# ── Main ──────────────────────────────────────────────────────────────────────

def export_campaign(campaign: dict) -> int:
    campaign_dir = Path(campaign["campaign_dir"])
    name         = campaign["name"]

    reviewed_csv = campaign_dir / campaign["reviewed_csv"]
    md_json      = campaign_dir / campaign["md_json"]

    print(f"\n{'-'*60}")
    print(f"Campaign : {name}")
    print(f"Dir      : {campaign_dir}")

    if not reviewed_csv.exists():
        print(f"  ERROR: reviewed CSV not found: {reviewed_csv}")
        return 0
    if not md_json.exists():
        print(f"  ERROR: MD JSON not found: {md_json}")
        return 0

    # Load MD confidence scores
    print("  Loading MegaDetector confidence scores …")
    md_conf = load_md_confidence(md_json)
    print(f"  {len(md_conf)} images with animal detections in MD JSON")

    # Load reviewed CSV
    with open(reviewed_csv, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"  {len(rows)} total rows in reviewed CSV")

    # Filter: confirmed animal observations with a real species
    animal_rows = [
        r for r in rows
        if r.get("observationType", "").strip() == "animal"
        and r.get("scientificName", "").strip() not in SKIP_SPECIES
    ]
    print(f"  {len(animal_rows)} animal rows with valid species")

    # Attach MD confidence
    for r in animal_rows:
        fp_key = r.get("filePath", "").replace("\\", "/").lower()
        r["_md_conf"] = md_conf.get(fp_key, 0.0)

    no_md = sum(1 for r in animal_rows if r["_md_conf"] == 0.0)
    if no_md:
        print(f"  Note: {no_md} rows had no MD detection entry (will sort last)")

    # Group by species → sort by MD conf desc → take top N
    by_species: dict[str, list] = defaultdict(list)
    for r in animal_rows:
        by_species[r["scientificName"].strip()].append(r)

    total_copied = 0
    for species in sorted(by_species):
        species_rows = sorted(by_species[species], key=lambda r: r["_md_conf"], reverse=True)
        top = species_rows[:TOP_N]

        out_dir = EXPORT_DIR / slugify(species) / name
        out_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for i, r in enumerate(top, 1):
            rel = r.get("filePath", "").replace("\\", "/")
            src = campaign_dir / rel
            suffix = Path(rel).suffix.lower() or ".jpg"
            dst = out_dir / f"best_{i:02d}{suffix}"

            if not src.exists():
                print(f"    MISSING: {src.name}")
                continue

            shutil.copy2(str(src), str(dst))
            copied += 1

        total_copied += copied
        print(f"  {species:<40} {copied}/{len(top)} images  (top MD conf: {top[0]['_md_conf']:.3f})")

    return total_copied


def main():
    print(f"Export dir: {EXPORT_DIR}")
    grand_total = 0
    for campaign in CAMPAIGNS:
        grand_total += export_campaign(campaign)
    print(f"\n{'='*60}")
    print(f"Total images exported: {grand_total}")
    print(f"Export dir : {EXPORT_DIR}")


if __name__ == "__main__":
    main()
