"""What must stay true of station identity, in the registry and in its GeoJSON.

The check that earns this file is `test_committed_artifact_matches_the_registry`: the
published GeoJSON must equal a fresh render of `estaciones.csv`. V2-REVIEW 1.6 asked
instead for a test that the files "agree on station count and coordinates to 5 decimal
places", and that is the weaker check -- it restates the projection in a second place, and
it passes vacuously on any field it does not happen to enumerate. `sd_card` lived in the
artifacts and in no test for five months precisely that way.

The defect being locked out: on 2026-08-24 one downstream copy held 26 stations and the
other two files held 27, so CT27's otono_2026 images ingested with no coordinates. Nothing
failed. That is the same shape as the CT26 coordinate error which reached a downstream copy
and came back as a 19 km displacement.

Since 2026-09-03 the producer publishes only its own GeoJSON; the downstream copy that used
to be spliced from here no longer exists, and its reader takes the registry directly.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camtrap import stations
from setup.build_station_registry import (
    RegistryArtifactError,
    _GEOJSON_REL,
    _PROJECT_ROOT,
    render_camera_traps,
    render_geojson,
    write_artifacts,
)

GEOJSON_PATH = _PROJECT_ROOT / _GEOJSON_REL


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

    def test_committed_artifact_matches_the_registry(self):
        """The one check. If this fails, someone hand-edited a generated file."""
        drifted = write_artifacts(check=True)
        self.assertEqual(
            drifted, [],
            f"{[str(p.relative_to(_PROJECT_ROOT)) for p in drifted]} differ from "
            f"{stations._REGISTRY_CSV.name}. Do not edit them by hand -- edit the "
            f"registry and run: python setup/build_station_registry.py",
        )

    def test_the_geojson_holds_the_same_stations_as_the_registry(self):
        registry_ids = set(stations.registry())
        geojson_ids = {
            f["properties"]["id"]
            for f in json.loads(GEOJSON_PATH.read_text("utf-8"))["features"]
        }
        self.assertEqual(registry_ids, geojson_ids)

    def test_ct27_is_present_everywhere(self):
        """The station whose absence started this. Named, so the regression is legible."""
        self.assertIn("CT27", stations.registry())
        ids = {f["properties"]["id"]
               for f in json.loads(GEOJSON_PATH.read_text("utf-8"))["features"]}
        self.assertIn("CT27", ids)


class TestCanonicalSpelling(unittest.TestCase):
    """One spelling everywhere (agreed 2026-08-24). The artifacts used to say `TC-01`."""

    def test_every_id_is_canonical(self):
        for entry in render_camera_traps():
            self.assertTrue(
                stations.is_canonical(entry["id"]),
                f"{entry['id']!r} does not match {stations.CANONICAL_PATTERN}",
            )

    def test_id_and_number_agree(self):
        for entry in render_camera_traps():
            self.assertEqual(entry["id"], stations.canonical_id(entry["tc"]))

    def test_the_artifact_does_not_spell_a_station_tc_dash(self):
        text = GEOJSON_PATH.read_text("utf-8")
        self.assertNotIn("TC-0", text)
        self.assertNotIn("TC-1", text)
        self.assertNotIn("TC-2", text)


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
        """Otherwise a truncated file regenerates the artifact with zero stations."""
        with _registry_csv("station_id,grid_id,lat,lon,elevation_m,notes\n"):
            with self.assertRaises(RegistryArtifactError):
                render_camera_traps()


class TestRoundTrip(unittest.TestCase):

    def test_regeneration_is_idempotent(self):
        """Running the generator twice must not produce a third state."""
        self.assertEqual(write_artifacts(check=True), [])

    def test_a_fresh_render_parses_back_to_the_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / _GEOJSON_REL.parent).mkdir(parents=True)
            write_artifacts(root=root)
            doc = json.loads((root / _GEOJSON_REL).read_text("utf-8"))
            self.assertEqual(
                [f["properties"]["id"] for f in doc["features"]],
                [e["id"] for e in render_camera_traps()],
            )
            for f in doc["features"]:
                row = stations.registry()[f["properties"]["id"]]
                lon, lat = f["geometry"]["coordinates"]
                self.assertAlmostEqual(float(row["lat"]), lat, places=5)
                self.assertAlmostEqual(float(row["lon"]), lon, places=5)


if __name__ == "__main__":
    unittest.main()
