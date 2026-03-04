"""
Loads Megadetector JSON + Timelapse2 .ddb database and returns the list of
images that need to be classified (animal detections + DeleteFlag misses).

Each entry in the returned list:
    {
        'id':        int,    # DataTable row Id
        'file_path': str,    # as stored in DB, e.g. "100EK113\\07110001.JPG"
        'full_path': str,    # absolute path to image file
        'bbox':      list|None,  # [x, y, w, h] relative (0-1), or None → full image
        'md_conf':   float,  # best MD confidence; 0.0 for DeleteFlag-only rows
        'source':    str,    # 'megadetector' or 'delete_flag'
    }
"""

import json
import sqlite3
from pathlib import Path


def load_animal_images(config: dict) -> list[dict]:
    campaign_dir = Path(config["campaign_dir"])
    threshold    = float(config["animal_confidence_threshold"])
    animal_cat   = str(config["animal_category"])

    # ── 1. Megadetector JSON → best detection per image ─────────────────────
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

    # ── 2. DataTable → filePath + DeleteFlag for every image ────────────────
    db_path = campaign_dir / config["database"]
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute("SELECT Id, filePath, DeleteFlag, fileMediatype FROM DataTable")
    db_rows = cur.fetchall()
    con.close()

    # ── 3. Merge ─────────────────────────────────────────────────────────────
    results: list[dict] = []
    for row_id, file_path, delete_flag, media_type in db_rows:
        # Skip videos — CLIP works on still images only
        if media_type and "video" in media_type.lower():
            continue

        norm = file_path.replace("\\", "/").lower()
        full_path = str(campaign_dir / file_path.replace("\\", "/"))

        if norm in md_best:
            det = md_best[norm]
            results.append({
                "id":        row_id,
                "file_path": file_path,
                "full_path": full_path,
                "bbox":      det["bbox"],
                "md_conf":   det["conf"],
                "source":    "megadetector",
            })
        elif delete_flag == "true":
            results.append({
                "id":        row_id,
                "file_path": file_path,
                "full_path": full_path,
                "bbox":      None,
                "md_conf":   0.0,
                "source":    "delete_flag",
            })

    return results
