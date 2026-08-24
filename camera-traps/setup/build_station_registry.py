"""Render the platform's station artifacts from the file that owns station identity.

`data/campaigns/estaciones.csv` is the owner. `plataforma-territorial/data/stations.yaml`
and `.../camera_trap_stations.geojson` are GENERATED FROM IT and must never be hand-edited;
`tests/test_station_registry.py` fails if they drift.

WHY THIS EXISTS. Three files carried station identity and only the CSV had all 27, so
CT27's otono_2026 images ingested with no coordinates -- the same class of defect as the
CT26 coordinate error that reached the platform and came back as a 19 km displacement.
Measured 2026-08-24 before this module was written, the three agreed on every value they
SHARED; the defect was a missing row and the absence of anything to keep it from
recurring. That is the failure this module is built to make impossible: not disagreement
between the files, but the silence when one falls behind.

ONE CANONICAL SPELLING, EVERYWHERE (Felipe, 2026-08-24). A station is `CT01`..`CT27` in
the field, in the pipeline, and on the platform. The artifacts previously spelled it
`TC-01`, which meant the project had two names for one thing and every reader had to know
which dialect it was holding. `camtrap.stations.canonical_id()` is the one place that
spelling is constructed and this module imports it rather than reproducing the format
string.

WHAT IS *NOT* GENERATED. `stations.yaml` also holds `reserve:` and `weather:`. Those are
not camera-trap stations, `estaciones.csv` does not own them, and this module does not
touch them: it rewrites the file from the `camera_traps:` line down and refuses if any
other top-level key follows. Everything above that line -- including the file's history
comments -- is preserved byte for byte.

Usage:
    python setup/build_station_registry.py            # regenerate both artifacts
    python setup/build_station_registry.py --check    # report drift, write nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camtrap import stations  # noqa: E402

# camera-traps/setup/build_station_registry.py -> parents[2] = monorepo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_YAML_REL = Path("plataforma-territorial/data/stations.yaml")
_GEOJSON_REL = Path("plataforma-territorial/data/camera_trap_stations.geojson")

_YAML_SECTION_KEY = "camera_traps:"

#: Emitted for every feature; the artifacts describe camera traps and nothing else.
_STATION_TYPE = "camera_trap"


class RegistryArtifactError(RuntimeError):
    """The artifact on disk is not in a shape this module is willing to rewrite."""


def _camera_num(station_id: str) -> int:
    """'CT07' -> 7. Inverse of `stations.canonical_id`, and it must stay that way."""
    return int(station_id[2:])


def _blank_to_none(value: str) -> str | None:
    """An empty registry cell means NOT RECORDED, which is `null`, never 0 or ''.

    Kept as TEXT rather than parsed to a number so a coordinate round-trips exactly:
    the registry writes `-71.74220` and float() would render it back as `-71.7422`,
    which is a spurious diff on every regeneration.
    """
    value = (value or "").strip()
    return value or None


def _number(value: str | None) -> int | float | None:
    """Registry text -> JSON number, keeping a whole number whole.

    `1263` must not serialise as `1263.0`: it is an elevation in metres read off a GPS,
    and the spurious decimal would rewrite 26 lines of the artifact on a regeneration
    that changed nothing.
    """
    if value is None:
        return None
    f = float(value)
    return int(f) if f.is_integer() else f


def _sorted_registry() -> list[dict[str, str]]:
    """Registry rows in canonical station order. Sorted by NUMBER, not by string.

    `CT10` sorts before `CT9` lexically; the ids are zero-padded so it does not bite
    today, but sorting on the number means it cannot start biting at CT100 either.
    """
    rows = stations.registry()
    if not rows:
        raise RegistryArtifactError(
            f"{stations._REGISTRY_CSV} is missing or empty; refusing to generate "
            f"artifacts that would silently drop every station."
        )
    return [rows[sid] for sid in sorted(rows, key=_camera_num)]


def render_camera_traps() -> list[dict]:
    """The `camera_traps:` section of stations.yaml, as plain data.

    `notes` rides along so the YAML emitter can render it as a comment; it is not a
    YAML key. It is dropped from the GeoJSON, which is machine-read.
    """
    entries = []
    for row in _sorted_registry():
        sid = row["station_id"].strip()
        num = _camera_num(sid)
        entries.append({
            "id": stations.canonical_id(num),
            "tc": num,
            "grid_id": _blank_to_none(row.get("grid_id", "")),
            "name": f"Cámara Trampa {num}",
            "lat": _blank_to_none(row.get("lat", "")),
            "lon": _blank_to_none(row.get("lon", "")),
            "altitude_m": _blank_to_none(row.get("elevation_m", "")),
            "type": _STATION_TYPE,
            "notes": _blank_to_none(row.get("notes", "")),
        })
    return entries


def render_geojson() -> dict:
    """The whole FeatureCollection, ready to serialise.

    Coordinates are [lon, lat] -- GeoJSON order, the reverse of how the registry and
    every human here writes them. That transposition is this function's job precisely
    so no caller has to remember it.
    """
    features = []
    for e in render_camera_traps():
        if e["lat"] is None or e["lon"] is None:
            raise RegistryArtifactError(
                f"station {e['id']} has no coordinates in the registry; a station "
                f"without a position cannot be a GeoJSON feature."
            )
        features.append({
            "type": "Feature",
            "properties": {
                "id": e["id"],
                "tc": e["tc"],
                "grid_id": None if e["grid_id"] is None else int(e["grid_id"]),
                "name": e["name"],
                "type": e["type"],
                "altitude_m": _number(e["altitude_m"]),
            },
            "geometry": {
                "type": "Point",
                "coordinates": [float(e["lon"]), float(e["lat"])],
            },
        })
    return {"type": "FeatureCollection", "features": features}


def _emit_yaml_section(entries: list[dict]) -> str:
    """Hand-emit the section rather than yaml.dump it, for two reasons.

    yaml.dump reformats every float (`-71.74220` -> `-71.7422`) and cannot write the
    `notes` comments at all. The structure here is a flat list of scalar maps, so
    emitting it directly is a smaller risk than the churn -- and the round-trip
    assertion in `write_artifacts` proves the result parses back to what we meant.
    """
    lines = [_YAML_SECTION_KEY]
    for i, e in enumerate(entries):
        if i:
            lines.append("")
        lines.append(f"  - id: {e['id']}")
        lines.append(f"    tc: {e['tc']}")
        lines.append(f"    grid_id: {e['grid_id'] if e['grid_id'] is not None else 'null'}")
        lines.append(f"    name: {e['name']}")
        lines.append(f"    lat: {e['lat']}")
        lines.append(f"    lon: {e['lon']}")
        lines.append(
            f"    altitude_m: {e['altitude_m'] if e['altitude_m'] is not None else 'null'}"
        )
        lines.append(f"    type: {e['type']}")
        if e["notes"]:
            for note_line in _wrap_comment(e["notes"]):
                lines.append(f"    # {note_line}")
    return "\n".join(lines) + "\n"


def _wrap_comment(text: str, width: int = 92) -> list[str]:
    """Wrap a registry note into YAML comment lines without breaking words."""
    words, out, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        out.append(cur)
    return out


def _spliced_yaml(existing: str, entries: list[dict]) -> str:
    """Replace the camera_traps section, preserve everything above it verbatim."""
    lines = existing.splitlines(keepends=True)
    starts = [i for i, ln in enumerate(lines) if ln.startswith(_YAML_SECTION_KEY)]
    if len(starts) != 1:
        raise RegistryArtifactError(
            f"expected exactly one top-level {_YAML_SECTION_KEY!r} line, found "
            f"{len(starts)}; refusing to guess where the generated section begins."
        )
    start = starts[0]
    trailing = [
        ln for ln in lines[start + 1:]
        if ln.strip() and not ln[0].isspace() and not ln.lstrip().startswith("#")
    ]
    if trailing:
        raise RegistryArtifactError(
            f"top-level key(s) appear AFTER {_YAML_SECTION_KEY!r}: "
            f"{[ln.split(':')[0] for ln in trailing]}. This module rewrites the file "
            f"from that line down and would delete them. Move them above it."
        )
    return "".join(lines[:start]) + _emit_yaml_section(entries)


def write_artifacts(root: Path | None = None, check: bool = False) -> list[Path]:
    """Regenerate both artifacts. Returns the paths whose content CHANGED.

    With `check=True` nothing is written and the same list is returned, which is what
    the test and a CI step both want: drift reported, disk untouched.
    """
    import yaml

    root = root or _REPO_ROOT
    yaml_path, geojson_path = root / _YAML_REL, root / _GEOJSON_REL
    entries = render_camera_traps()

    new_yaml = _spliced_yaml(yaml_path.read_text(encoding="utf-8"), entries)

    # The emitter writes YAML by hand, so prove it parses back to what was intended
    # before it reaches disk. A malformed splice must not be a committed artifact.
    parsed = yaml.safe_load(new_yaml)["camera_traps"]
    if [c["id"] for c in parsed] != [e["id"] for e in entries]:
        raise RegistryArtifactError(
            "the emitted YAML does not parse back to the stations it was built from."
        )

    new_geojson = json.dumps(render_geojson(), indent=2, ensure_ascii=False) + "\n"

    changed = []
    for path, content in ((yaml_path, new_yaml), (geojson_path, new_geojson)):
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current != content:
            changed.append(path)
            if not check:
                path.write_text(content, encoding="utf-8")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="report drift and write nothing (exit 1 if any)")
    args = ap.parse_args()

    changed = write_artifacts(check=args.check)
    n = len(render_camera_traps())
    if args.check:
        if changed:
            print(f"DRIFT: {len(changed)} artifact(s) differ from the registry:")
            for p in changed:
                print(f"  {p.relative_to(_REPO_ROOT)}")
            print("Run: python setup/build_station_registry.py")
            return 1
        print(f"Both artifacts match the registry ({n} stations).")
        return 0

    if changed:
        for p in changed:
            print(f"wrote {p.relative_to(_REPO_ROOT)}")
    else:
        print("no change")
    print(f"{n} stations from {stations._REGISTRY_CSV.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
