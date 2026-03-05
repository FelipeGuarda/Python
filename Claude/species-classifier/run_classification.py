"""
Entry point for zero-shot CLIP species classification of a camera-trap campaign.

Usage (from the species-classifier/ directory):
    conda activate species-classifier
    python run_classification.py [--config config.yaml]

Output:
    <campaign_dir>/new_labeled_data_classified.csv
        — same format as new_labeled_data_CamptrapDP.csv but with classified rows:
          · scientificName       ← Latin binomial
          · observationComments  ← Spanish common name
          · observationType      ← 'animal'
          · classificationMethod ← 'machine'
          · classifiedBy         ← 'CLIP zero-shot'
          · classificationProbability ← cosine similarity score
          · classificationTimestamp   ← ISO timestamp
"""

import argparse
import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import yaml
from tqdm import tqdm

from classify_campaign.clip_classifier import CLIPZeroShotClassifier
from classify_campaign.crop_utils import load_and_crop
from classify_campaign.data_loader import load_animal_images


# ── CSV helpers ───────────────────────────────────────────────────────────────

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


# ── DB helper: Id → filePath lookup ──────────────────────────────────────────

def build_id_to_filepath(db_path: Path) -> dict[int, str]:
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute("SELECT Id, filePath FROM DataTable")
    mapping = {row_id: fp for row_id, fp in cur.fetchall()}
    con.close()
    return mapping


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_path: str) -> None:
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    campaign_dir = Path(config["campaign_dir"])

    # ── Load image list ───────────────────────────────────────────────────────
    print("Loading data sources …")
    images = load_animal_images(config)
    n_md  = sum(1 for i in images if i["source"] == "megadetector")
    n_del = sum(1 for i in images if i["source"] == "delete_flag")
    print(f"  {len(images)} images to classify  "
          f"({n_md} MD detections >={config['animal_confidence_threshold']}, "
          f"{n_del} DeleteFlag misses)")

    if not images:
        print("Nothing to classify. Check the JSON path and threshold in config.yaml.")
        return

    # ── Load CLIP ─────────────────────────────────────────────────────────────
    print("Loading CLIP model …")
    classifier = CLIPZeroShotClassifier(
        model_name=config["clip_model"],
        species_list=config["species"],
    )

    # ── Classify ──────────────────────────────────────────────────────────────
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    clip_threshold = float(config.get("clip_confidence_threshold", 0.0))
    id_to_result: dict[int, dict] = {}
    skipped = 0
    low_conf = 0

    print("Classifying …")
    for img_info in tqdm(images, unit="img"):
        crop = load_and_crop(img_info["full_path"], img_info["bbox"])
        if crop is None:
            skipped += 1
            continue

        species, score = classifier.classify(crop)
        if score < clip_threshold:
            low_conf += 1
            id_to_result[img_info["id"]] = {
                "latin":   "",
                "spanish": "No reconocible",
                "score":   score,
            }
        else:
            id_to_result[img_info["id"]] = {
                "latin":   species["latin"],
                "spanish": species["spanish"],
                "score":   score,
            }

    print(f"  Classified: {len(id_to_result)}  |  Low confidence (<{clip_threshold}): {low_conf}  |  Skipped (unreadable): {skipped}")

    # ── Build filePath → result lookup ────────────────────────────────────────
    db_path = campaign_dir / config["database"]
    id_to_fp = build_id_to_filepath(db_path)
    fp_to_result = {
        id_to_fp[row_id]: result
        for row_id, result in id_to_result.items()
        if row_id in id_to_fp
    }

    # ── Update CSV ────────────────────────────────────────────────────────────
    input_csv  = campaign_dir / config["input_csv"]
    output_csv = campaign_dir / config["output_csv"]

    print(f"Writing {output_csv.name} …")
    fieldnames, rows = read_csv(input_csv)
    updated = 0

    for row in rows:
        fp = row.get("filePath", "")
        if fp not in fp_to_result:
            continue
        result = fp_to_result[fp]
        row["observationType"]          = "animal"
        row["scientificName"]           = result["latin"]
        row["observationComments"]      = result["spanish"]
        row["classificationMethod"]     = config["classification_method"]
        row["classifiedBy"]             = config["classified_by"]
        row["classificationTimestamp"]  = timestamp
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
