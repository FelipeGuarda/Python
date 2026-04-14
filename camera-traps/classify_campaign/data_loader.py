"""
Loads ImageData_animals.csv (Timelapse2 CamtrapDP export pre-filtered to
observationType=animal) and returns the list of images to classify, with
bounding boxes from the MegaDetector JSON.

Each entry in the returned list:
    {
        'file_path': str,    # relative path, e.g. "TC10_M3.2/06080001.JPG"
        'full_path': str,    # absolute path to image file
        'bbox':      list|None,  # [x, y, w, h] relative (0-1), or None → full image
        'md_conf':   float,  # best MD confidence; 0.0 for images not in MD JSON
        'source':    str,    # 'megadetector' or 'csv_only'
    }
"""

import csv
import json
from pathlib import Path


def load_animal_images(config: dict) -> list[dict]:
    campaign_dir = Path(config["campaign_dir"])
    animal_cat   = str(config["animal_category"])
    threshold    = float(config["animal_confidence_threshold"])

    # ── 1. MegaDetector JSON → best detection per image ─────────────────────
    json_path = campaign_dir / config["megadetector_json"]
    with open(json_path, encoding="utf-8") as f:
        md_data = json.load(f)

    md_best: dict[str, dict] = {}
    for img in md_data["images"]:
        key = img["file"].replace("\\", "/").lower()
        for det in img.get("detections", []):
            if det.get("category") == animal_cat and det.get("conf", 0) >= threshold:
                if key not in md_best or det["conf"] > md_best[key]["conf"]:
                    md_best[key] = det

    # ── 2. CSV → animal image list ───────────────────────────────────────────
    # The input CSV is exported from Timelapse2 filtered to observationType=animal.
    # filePath is populated in some campaigns; in others only RelativePath + File
    # are present. We handle both.
    csv_path = campaign_dir / config["input_csv"]
    results: list[dict] = []

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("observationType", "").strip().lower() != "animal":
                continue
            if "video" in row.get("fileMediatype", "").lower():
                continue

            fp = row.get("filePath", "").strip()
            if not fp:
                rel   = row.get("RelativePath", "").strip()
                fname = row.get("File", "").strip()
                if not rel or not fname:
                    continue
                fp = rel + "/" + fname

            norm      = fp.replace("\\", "/").lower()
            full_path = str(campaign_dir / fp.replace("\\", "/"))

            if norm in md_best:
                det = md_best[norm]
                results.append({
                    "file_path": fp,
                    "full_path": full_path,
                    "bbox":      det["bbox"],
                    "md_conf":   det["conf"],
                    "source":    "megadetector",
                })
            else:
                # Animal confirmed in Timelapse2 but not caught by MegaDetector
                results.append({
                    "file_path": fp,
                    "full_path": full_path,
                    "bbox":      None,
                    "md_conf":   0.0,
                    "source":    "csv_only",
                })

    return results
