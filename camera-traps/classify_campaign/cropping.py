"""
Crops a PIL image to a Megadetector bounding box.

Bbox format: [x, y, width, height] in relative coordinates (0–1),
where (x, y) is the top-left corner.

Falls back to the full image when bbox is None (DeleteFlag cases).
"""

from PIL import Image


PADDING = 0.05  # extra fraction added around the box before cropping


def crop_to_bbox(image: Image.Image, bbox: list | None) -> Image.Image:
    """Return a cropped PIL image. Returns the full image if bbox is None."""
    if bbox is None:
        return image

    iw, ih = image.size
    x, y, bw, bh = bbox

    x1 = max(0.0, x - PADDING)
    y1 = max(0.0, y - PADDING)
    x2 = min(1.0, x + bw + PADDING)
    y2 = min(1.0, y + bh + PADDING)

    return image.crop((
        int(x1 * iw),
        int(y1 * ih),
        int(x2 * iw),
        int(y2 * ih),
    ))


def load_and_crop(image_path: str, bbox: list | None) -> Image.Image | None:
    """Load image from disk and crop to bbox. Returns None on any read error."""
    try:
        img = Image.open(image_path).convert("RGB")
        return crop_to_bbox(img, bbox)
    except Exception as exc:
        print(f"  [warn] could not load {image_path}: {exc}")
        return None
