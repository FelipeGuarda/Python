"""What must stay true of station identity, in the registry and in both artifacts.

The check that earns this file is `test_committed_artifacts_match_the_registry`: the
platform's two station files must equal a fresh render of `estaciones.csv`. V2-REVIEW 1.6
asked instead for a test that the three files "agree on station count and coordinates to 5
decimal places", and that is the weaker check -- it restates the projection in a second
place, and it passes vacuously on any field it does not happen to enumerate. `sd_card`
lived in the artifacts and in no test for five months precisely that way.

The defect being locked out: on 2026-08-24 `stations.yaml` held 26 stations and the other
two held 27, so CT27's otono_2026 images ingested with no coordinates. Nothing failed.
That is the same shape as the CT26 coordinate error which reached the platform and came
back as a 19 km displacement.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camtrap import stations
from setup.build_station_registry import (
    RegistryArtifactError,
    _GEOJSON_REL,
    _REPO_ROOT,
    _YAML_REL,
    _spliced_yaml,
    render_camera_traps,
    render_geojson,
    write_artifacts,
)

YAML_PATH = _REPO_ROOT / _YAML_REL
GEOJSON_PATH = _REPO_ROOT / _GEOJSON_REL


@contextmanager
def _registry_csv(text: str):
    """Point the registry loader at a throwaway CSV for one test.

    `stations.registry` is lru_cached, so the cache is cleared on the way in AND on the
    way out -- a stub left in the cache would poison every later test in the process.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "estaciones.csv"
        path.write_text(text, encoding="utf-8")
        original = stations._REGISTRY_CSV
        stations._REGISTRY_CSV = path
        stations.registry.cache_clear()
        try:
            yield path
        finally:
            stations._REGISTRY_CSV = original
            stations.registry.cache_clear()


class TestArtifactsMatchRegistry(unittest.TestCase):

    def test_committed_artifacts_match_the_registry(self):
        """The one check. If this fails, someone hand-edited a generated file."""
        drifted = write_artifacts(check=True)
        self.assertEqual(
            drifted, [],
            f"{[str(p.relative_to(_REPO_ROOT)) for p in drifted]} differ from "
            f"{stations._REGISTRY_CSV.name}. Do not edit them by hand -- edit the "
            f"registry and run: python setup/build_station_registry.py",
        )

    def test_all_three_files_hold_the_same_stations(self):
        registry_ids = set(stations.registry())
        yaml_ids = {c["id"] for c in yaml.safe_load(YAML_PATH.read_text("utf-8"))["camera_traps"]}
        geojson_ids = {
            f["properties"]["id"]
            for f in json.loads(GEOJSON_PATH.read_text("utf-8"))["features"]
        }
        self.assertEqual(registry_ids, yaml_ids)
        self.assertEqual(registry_ids, geojson_ids)

    def test_ct27_is_present_everywhere(self):
        """The station whose absence started this. Named, so the regression is legible."""
        self.assertIn("CT27", stations.registry())
        ids = {c["id"] for c in yaml.safe_load(YAML_PATH.read_text("utf-8"))["camera_traps"]}
        self.assertIn("CT27", ids)


class TestCanonicalSpelling(unittest.TestCase):
    """One spelling everywhere (Felipe, 2026-08-24). The artifacts used to say `TC-01`."""

    def test_every_id_is_canonical(self):
        for entry in render_camera_traps():
            self.assertTrue(
                stations.is_canonical(entry["id"]),
                f"{entry['id']!r} does not match {stations.CANONICAL_PATTERN}",
            )

    def test_id_and_number_agree(self):
        for entry in render_camera_traps():
            self.assertEqual(entry["id"], stations.canonical_id(entry["tc"]))

    def test_no_artifact_still_spells_a_station_tc_dash(self):
        for path in (YAML_PATH, GEOJSON_PATH):
            self.assertNotIn("TC-0", path.read_text("utf-8"))
            self.assertNotIn("TC-1", path.read_text("utf-8"))
            self.assertNotIn("TC-2", path.read_text("utf-8"))


class TestGeoJSON(unittest.TestCase):

    def test_coordinates_are_lon_lat(self):
        """Bosque Pehuen is near -39.4 lat / -71.7 lon. Transposed, lat would be out of range."""
        for feature in render_geojson()["features"]:
            lon, lat = feature["geometry"]["coordinates"]
            self.assertTrue(-72.0 < lon < -71.0, f"{feature['properties']['id']} lon={lon}")
            self.assertTrue(-40.0 < lat < -39.0, f"{feature['properties']['id']} lat={lat}")

    def test_whole_elevations_stay_whole(self):
        """1263 metres, never 1263.0 -- a spurious decimal rewrites the artifact for nothing."""
        alts = [f["properties"]["altitude_m"] for f in render_geojson()["features"]]
        self.assertIn(1263, alts)
        self.assertNotIn(1263.0, [a for a in alts if isinstance(a, float)])

    def test_a_station_without_coordinates_is_refused(self):
        """A position is what a station IS here; a blank one must stop the render, not
        become a feature at (0, 0) off the coast of Africa."""
        with _registry_csv(
            "station_id,grid_id,lat,lon,elevation_m,notes\n"
            "CT01,1,,,,\n"
        ):
            with self.assertRaises(RegistryArtifactError):
                render_geojson()

    def test_an_empty_registry_is_refused(self):
        """Otherwise a truncated file regenerates both artifacts with zero stations."""
        with _registry_csv("station_id,grid_id,lat,lon,elevation_m,notes\n"):
            with self.assertRaises(RegistryArtifactError):
                render_camera_traps()


class TestYamlSplice(unittest.TestCase):
    """The splice rewrites from `camera_traps:` down, so what sits above it must survive."""

    def test_reserve_and_weather_survive(self):
        doc = yaml.safe_load(YAML_PATH.read_text("utf-8"))
        self.assertIn("reserve", doc)
        self.assertIn("weather", doc)
        self.assertEqual(doc["weather"][0]["id"], "WS-01")
        self.assertEqual(doc["reserve"]["timezone"], "America/Santiago")

    def test_header_comments_survive(self):
        self.assertIn("Fuente: CT ID and coordinates.xlsx", YAML_PATH.read_text("utf-8"))

    def test_registry_notes_render_as_comments(self):
        """CT26's coordinate-error note is provenance; it must not be lost on a rebuild."""
        self.assertIn("19 km fuera de la reserva", YAML_PATH.read_text("utf-8"))

    def test_a_top_level_key_after_the_section_is_refused(self):
        """Otherwise the splice would silently delete it."""
        existing = "reserve:\n  zoom: 14\n\ncamera_traps:\n  - id: CT01\n\nfootpaths:\n  - a\n"
        with self.assertRaises(RegistryArtifactError) as cm:
            _spliced_yaml(existing, render_camera_traps())
        self.assertIn("footpaths", str(cm.exception))

    def test_a_missing_section_is_refused(self):
        with self.assertRaises(RegistryArtifactError):
            _spliced_yaml("reserve:\n  zoom: 14\n", render_camera_traps())


class TestRoundTrip(unittest.TestCase):

    def test_regeneration_is_idempotent(self):
        """Running the generator twice must not produce a third state."""
        self.assertEqual(write_artifacts(check=True), [])

    def test_emitted_yaml_parses_back_to_the_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / _YAML_REL.parent).mkdir(parents=True)
            (root / _YAML_REL).write_text(YAML_PATH.read_text("utf-8"), encoding="utf-8")
            (root / _GEOJSON_REL).write_text("{}", encoding="utf-8")
            write_artifacts(root=root)

            doc = yaml.safe_load((root / _YAML_REL).read_text("utf-8"))
            self.assertEqual(
                [c["id"] for c in doc["camera_traps"]],
                [e["id"] for e in render_camera_traps()],
            )
            for c in doc["camera_traps"]:
                row = stations.registry()[c["id"]]
                self.assertAlmostEqual(float(row["lat"]), float(c["lat"]), places=5)
                self.assertAlmostEqual(float(row["lon"]), float(c["lon"]), places=5)


if __name__ == "__main__":
    unittest.main()
