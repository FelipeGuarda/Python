"""
Entry point for zero-shot CLIP species classification of a camera-trap campaign.

Usage (from the camera-traps/ directory):
    conda activate species-classifier
    python run_classification.py [--config config.yaml]

Input:
    <campaign_dir>/ImageData_animals.csv
        — Timelapse2 CamtrapDP export filtered to observationType=animal
    <campaign_dir>/timelapse_recognition_file.json
        — MegaDetector output (provides bounding boxes)

Output:
    <campaign_dir>/ImageData_animals_classified.csv
        — same rows as the input CSV but with classified animals filled in:
          · scientificName            ← Latin binomial
          · observationComments       ← Spanish common name
          · observationType           ← 'animal' (already set, confirmed)
          · classificationMethod      ← 'machine'
          · classifiedBy              ← 'CLIP zero-shot'
          · classificationProbability ← cosine similarity score
          · classificationTimestamp   ← ISO timestamp
"""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import yaml
from tqdm import tqdm

from classify_campaign.clip_classifier import CLIPZeroShotClassifier
from classify_campaign.crop_utils import load_and_crop
from classify_campaign.data_loader import load_animal_images


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    return fieldnames, rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_file_path(row: dict) -> str:
    """Return the normalised relative file path for a CSV row."""
    fp = row.get("filePath", "").strip()
    if not fp:
        rel   = row.get("RelativePath", "").strip()
        fname = row.get("File", "").strip()
        fp = rel + "/" + fname if rel and fname else ""
    return fp.replace("\\", "/").lower()


def main(config_path: str) -> None:
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    campaign_dir = Path(config["campaign_dir"])

    # ── Load image list ───────────────────────────────────────────────────────
    print("Loading data sources …")
    images   = load_animal_images(config)
    n_md     = sum(1 for i in images if i["source"] == "megadetector")
    n_csv    = sum(1 for i in images if i["source"] == "csv_only")
    print(f"  {len(images)} images to classify  "
          f"({n_md} MD detections >={config['animal_confidence_threshold']}, "
          f"{n_csv} CSV-only / no MD bbox)")

    if not images:
        print("Nothing to classify. Check input_csv path and megadetector_json in config.yaml.")
        return

    # ── Load CLIP ─────────────────────────────────────────────────────────────
    print("Loading CLIP model …")
    classifier = CLIPZeroShotClassifier(
        model_name=config["clip_model"],
        species_list=config["species"],
    )

    # ── Classify ──────────────────────────────────────────────────────────────
    timestamp      = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    clip_threshold = float(config.get("clip_confidence_threshold", 0.0))
    # key: normalised relative path  →  classification result
    fp_to_result: dict[str, dict] = {}
    skipped  = 0
    low_conf = 0

    print("Classifying …")
    for img_info in tqdm(images, unit="img"):
        crop = load_and_crop(img_info["full_path"], img_info["bbox"])
        if crop is None:
            skipped += 1
            continue

        norm_key = img_info["file_path"].replace("\\", "/").lower()
        species, score = classifier.classify(crop)

        if score < clip_threshold:
            low_conf += 1
            fp_to_result[norm_key] = {"latin": "", "spanish": "No reconocible", "score": score}
        else:
            fp_to_result[norm_key] = {"latin": species["latin"], "spanish": species["spanish"], "score": score}

    print(f"  Classified: {len(fp_to_result)}  |  "
          f"Low confidence (<{clip_threshold}): {low_conf}  |  "
          f"Skipped (unreadable): {skipped}")

    # ── Update CSV ────────────────────────────────────────────────────────────
    input_csv  = campaign_dir / config["input_csv"]
    output_csv = campaign_dir / config["output_csv"]

    print(f"Writing {output_csv.name} …")
    fieldnames, rows = read_csv(input_csv)
    updated = 0

    for row in rows:
        key = row_file_path(row)
        if key not in fp_to_result:
            continue
        result = fp_to_result[key]
        row["observationType"]           = "animal"
        row["scientificName"]            = result["latin"]
        row["observationComments"]       = result["spanish"]
        row["classificationMethod"]      = config["classification_method"]
        row["classifiedBy"]              = config["classified_by"]
        row["classificationTimestamp"]   = timestamp
        row["classificationProbability"] = str(result["score"])
        updated += 1

    write_csv(output_csv, fieldnames, rows)
    print(f"Done. {updated} rows classified → {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zero-shot CLIP species classifier for camera-trap campaigns."
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml in current directory)"
    )
    args = parser.parse_args()
    main(args.config)
