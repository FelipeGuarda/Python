# run_megadetector_v6.py
import os
import argparse
from pathlib import Path

from wildlife_detector import Detector
from wildlife_detector.render import draw_bounding_boxes
from wildlife_detector.utils import save_timelapse_json


def gather_images(root_dir):
    """Recursively gather all image files under root_dir."""
    exts = {".jpg", ".jpeg", ".png"}
    image_paths = []

    for root, _, files in os.walk(root_dir):
        for f in files:
            if Path(f).suffix.lower() in exts:
                image_paths.append(os.path.join(root, f))

    return sorted(image_paths)


def main():
    parser = argparse.ArgumentParser(description="Run Megadetector v6 on a folder.")
    parser.add_argument("--input_dir", required=True,
                        help="Root directory containing subdirectories with images.")
    parser.add_argument("--output_json", required=True,
                        help="Path to save the Timelapse-compatible JSON.")
    parser.add_argument("--save_annotated", action="store_true",
                        help="Save annotated images.")
    parser.add_argument("--annotated_dir",
                        help="Directory for annotated images.")
    parser.add_argument("--confidence", type=float, default=0.2,
                        help="Confidence threshold.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference.")

    args = parser.parse_args()

    input_dir = args.input_dir
    image_paths = gather_images(input_dir)

    print(f"Found {len(image_paths)} images.")

    # 1. Load MegaDetector v6
    detector = Detector("md_v6a")

    # 2. Run inference
    print("Running inference...")
    results = detector.detect(
        image_paths,
        batch_size=args.batch_size,
        detection_threshold=args.confidence
    )

    # 3. Save Timelapse-compatible JSON
    print(f"Saving Timelapse output to {args.output_json}")
    save_timelapse_json(results, args.output_json)

    # 4. Optional: save annotated images
    if args.save_annotated:
        if not args.annotated_dir:
            raise ValueError("Specify --annotated_dir when using --save_annotated")

        for item in results["images"]:
            src_path = item["file"]
            detections = item["detections"]

            out_path = os.path.join(
                args.annotated_dir,
                os.path.relpath(src_path, input_dir)
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            draw_bounding_boxes(src_path, detections, out_path)

        print("Annotated images saved.")

    print("Done!")


if __name__ == "__main__":
    main()
