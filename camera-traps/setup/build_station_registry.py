"""Render the station registry's GeoJSON from the file that owns station identity.

`data/campaigns/estaciones.csv` is the owner. `data/campaigns/estaciones.geojson` is
GENERATED FROM IT and must never be hand-edited; `tests/test_station_registry.py` fails if
it drifts.

WHY THIS EXISTS. Three files once carried station identity and only the CSV had all 27, so
CT27's otono_2026 images ingested with no coordinates -- the same class of defect as the
CT26 coordinate error that reached a downstream copy and came back as a 19 km displacement.
Measured 2026-08-24 before this module was written, the three agreed on every value they
SHARED; the defect was a missing row and the absence of anything to keep it from
recurring. That is the failure this module is built to make impossible: not disagreement
between the files, but the silence when one falls behind.

WHO READS THE OUTPUT. Nobody this module knows about. The registry and its GeoJSON are
published here, next to each other, and any consumer reads them from here. Until
2026-09-03 this module also spliced a `camera_traps:` section into a downstream project's
config file, which meant the producer knew a consumer's path and layout; that direction of
knowledge is the one the data-health manual forbids (§10F.2), so the splice is gone and
that consumer reads the registry directly.

ONE CANONICAL SPELLING, EVERYWHERE (agreed 2026-08-24). A station is `CT01`..`CT27` in the
field, in the pipeline, and downstream. The artifacts previously spelled it `TC-01`, which
meant the project had two names for one thing and every reader had to know which dialect
it was holding. `camtrap.stations.canonical_id()` is the one place that spelling is
constructed and this module imports it rather than reproducing the format string.

Usage:
    python setup/build_station_registry.py            # regenerate the GeoJSON
    python setup/build_station_registry.py --check    # report drift, write nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camtrap import stations  # noqa: E402

# camera-traps/setup/build_station_registry.py -> parents[1] = camera-traps/
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_GEOJSON_REL = Path("data/campaigns/estaciones.geojson")

#: Emitted for every feature; the artifact describes camera traps and nothing else.
_STATION_TYPE = "camera_trap"


class RegistryArtifactError(RuntimeError):
    """The registry is not in a shape this module is willing to render."""


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
            f"an artifact that would silently drop every station."
        )
    return [rows[sid] for sid in sorted(rows, key=_camera_num)]


def render_camera_traps() -> list[dict]:
    """The registry as plain data, one entry per station in canonical order."""
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


def write_artifacts(root: Path | None = None, check: bool = False) -> list[Path]:
    """Regenerate the GeoJSON. Returns the paths whose content CHANGED.

    With `check=True` nothing is written and the same list is returned, which is what
    the test and a CI step both want: drift reported, disk untouched.
    """
    root = root or _PROJECT_ROOT
    geojson_path = root / _GEOJSON_REL
    new_geojson = json.dumps(render_geojson(), indent=2, ensure_ascii=False) + "\n"

    current = geojson_path.read_text(encoding="utf-8") if geojson_path.exists() else None
    if current == new_geojson:
        return []
    if not check:
        geojson_path.write_text(new_geojson, encoding="utf-8")
    return [geojson_path]


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
                print(f"  {p.relative_to(_PROJECT_ROOT)}")
            print("Run: python setup/build_station_registry.py")
            return 1
        print(f"The GeoJSON matches the registry ({n} stations).")
        return 0

    if changed:
        for p in changed:
            print(f"wrote {p.relative_to(_PROJECT_ROOT)}")
    else:
        print("no change")
    print(f"{n} stations from {stations._REGISTRY_CSV.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
