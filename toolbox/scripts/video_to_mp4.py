"""Convert camera and phone video to H.264 MP4.

    python scripts/video_to_mp4.py 11210204.MOV
    python scripts/video_to_mp4.py "D:/Salidas/Marzo" -o converted/

Accepts a file or a directory. Existing outputs are skipped unless --force.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Formats FMA field cameras and phones actually produce.
SOURCE_SUFFIXES = {".mov", ".avi", ".mts", ".m4v", ".mpg", ".mpeg", ".wmv"}

# libx264 + aac is the pair that plays everywhere without re-encoding on the
# viewer's side — notably in Google Slides and WhatsApp, where the originals
# do not.
VIDEO_CODEC = "libx264"
AUDIO_CODEC = "aac"


def _sources(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*") if p.suffix.lower() in SOURCE_SUFFIXES)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert video files to MP4 (H.264/AAC).")
    parser.add_argument("target", type=Path, help="a video file, or a directory to walk")
    parser.add_argument("-o", "--output-dir", type=Path,
                        help="where to write (default: alongside each source)")
    parser.add_argument("--force", action="store_true", help="re-encode even if the .mp4 exists")
    args = parser.parse_args()

    if not args.target.exists():
        parser.error(f"not found: {args.target}")

    sources = _sources(args.target)
    if not sources:
        print(f"No convertible video under {args.target} "
              f"({', '.join(sorted(SOURCE_SUFFIXES))})")
        return 0

    # Imported here so --help stays instant: moviepy pulls in numpy and
    # probes for an ffmpeg binary at import time.
    from moviepy import VideoFileClip

    converted = skipped = 0
    for index, source in enumerate(sources, start=1):
        destination_dir = args.output_dir or source.parent
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{source.stem}.mp4"

        if destination.exists() and not args.force:
            print(f"[{index}/{len(sources)}] skip (exists): {destination.name}")
            skipped += 1
            continue

        print(f"[{index}/{len(sources)}] {source.name} -> {destination}")
        with VideoFileClip(str(source)) as clip:
            clip.write_videofile(str(destination), codec=VIDEO_CODEC, audio_codec=AUDIO_CODEC)
        converted += 1

    print(f"\nDone — {converted} converted, {skipped} skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
