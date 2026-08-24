"""The deployment window is a denominator, so its failure modes are silent.

Every test here guards a way the number could be wrong while still looking like a
number: padded by a tolerance that belongs to anchor validation, truncated by an
assumed visit hour, invented from a half-open window, or quietly absent.
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camtrap import deployments
from camtrap.anchors import FIELD_NOTES_FILENAME, FieldRecord
from camtrap.canonical_state import PUBLISHED_CAMPAIGNS
from camtrap.observations import CAMPAIGNS_ROOT, CANONICAL_FILENAME

FIELD_NOTES = CAMPAIGNS_ROOT / FIELD_NOTES_FILENAME


class TestWindowIsTheFieldWindow(unittest.TestCase):

    def test_the_anchor_tolerance_is_not_applied(self):
        """FieldRecord.window() pads by +/-3 d so a clock anchor can be checked against
        a window it may sit just outside of. Applied to effort that adds six days to
        every camera in the reserve. This is the single most likely way for this module
        to go wrong, because window() is the obvious method to reach for."""
        field = FieldRecord.load(FIELD_NOTES)
        frame = deployments.build("otono_2025")
        row = frame[frame["station_id"] == "CT03"].iloc[0]

        opening = field.opening("CT03", "otono_2025")
        padded_start, padded_end = field.window("CT03", "otono_2025")

        self.assertEqual(row["field_start"], opening.visit_date.date().isoformat())
        self.assertGreater(row["field_start"], padded_start.date().isoformat())
        self.assertLess(row["field_end"], padded_end.date().isoformat())

    def test_days_are_date_scale_not_datetime_scale(self):
        """CT01's install carries a recorded time (15:13) and its retrieval does not, so
        the retrieval is stamped at ASSUMED_VISIT_HOUR. Subtracting the two datetimes
        truncates to 168 days; the deployment is 169. An assumed hour must never reach
        a camera-day count."""
        frame = deployments.build("otono_2025")
        row = frame[frame["station_id"] == "CT01"].iloc[0]
        self.assertEqual(row["field_start"], "2024-12-10")
        self.assertEqual(row["field_end"], "2025-05-28")
        self.assertEqual(row["field_days"], 169)

    def test_both_ends_are_required(self):
        """A half-open window would invent an end date, and an invented denominator is
        worse than a missing one: it cannot be spotted downstream."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with open(root / FIELD_NOTES_FILENAME, "w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["campaign_closed", "campaign_opened", "station_id",
                            "visit_type", "visit_date", "visit_time"])
                # Opens otono_2025 and is never closed.
                w.writerow(["", "otono_2025", "CT99", "install", "2025-01-01", ""])
            frame = deployments.build("otono_2025", root=root)
        self.assertEqual(len(frame), 0, "a station with one dated end is not a deployment")


class TestPublishedFiles(unittest.TestCase):

    def test_committed_files_equal_a_fresh_build(self):
        """The CSVs are generated, not maintained. A hand edit is a defect."""
        for campaign in PUBLISHED_CAMPAIGNS:
            with self.subTest(campaign=campaign):
                path = CAMPAIGNS_ROOT / campaign / deployments.DEPLOYMENTS_FILENAME
                self.assertTrue(path.exists(), f"{path} missing; run python -m camtrap.deployments --publish")
                committed = pd.read_csv(path, keep_default_na=False)
                fresh = deployments.build(campaign).fillna("")
                fresh["field_days"] = fresh["field_days"].apply(
                    lambda v: "" if v == "" else v)
                self.assertEqual(
                    committed.to_csv(index=False),
                    fresh.to_csv(index=False),
                    f"{path} differs from a fresh build",
                )

    def test_columns_are_the_published_contract(self):
        for campaign in PUBLISHED_CAMPAIGNS:
            path = CAMPAIGNS_ROOT / campaign / deployments.DEPLOYMENTS_FILENAME
            with open(path, encoding="utf-8") as fh:
                header = next(csv.reader(fh))
            self.assertEqual(tuple(header), deployments.COLUMNS, campaign)


class TestAgreementWithTheCanonicalTable(unittest.TestCase):

    def test_has_media_matches_the_canonical_table(self):
        for campaign in PUBLISHED_CAMPAIGNS:
            with self.subTest(campaign=campaign):
                df = pd.read_parquet(CAMPAIGNS_ROOT / campaign / CANONICAL_FILENAME,
                                     columns=["station_canonical"])
                expected = set(df["station_canonical"].dropna().unique())
                frame = deployments.build(campaign)
                actual = set(frame.loc[frame["has_media"], "station_id"])
                self.assertEqual(actual, expected)

    def test_every_station_with_images_has_a_window(self):
        """The property that makes effort computable at all. It is true for 74 of 74
        deployments as of 2026-08-24, and only after CT27's two field dates were
        reconstructed. If this fails, some camera has images and no denominator."""
        for campaign in PUBLISHED_CAMPAIGNS:
            with self.subTest(campaign=campaign):
                frame = deployments.build(campaign)
                orphans = frame[frame["has_media"] & frame["field_days"].isna()]
                self.assertEqual(
                    list(orphans["station_id"]), [],
                    "station(s) with images and no field window",
                )

    def test_ct27_carries_both_reconstructed_dates(self):
        """CT27 never appeared on an install sheet and was omitted from the May 2026
        retrieval sheet, so both ends were reconstructed on 2026-08-24 -- the opening
        from a resolved day/month transposition, the closing from retrieval-trip order.
        A revert of either would silently return CT27 to an observed-media window."""
        frame = deployments.build("otono_2026")
        row = frame[frame["station_id"] == "CT27"].iloc[0]
        self.assertEqual(row["field_start"], "2025-12-11")
        self.assertEqual(row["field_end"], "2026-05-14")
        self.assertTrue(row["has_media"])

    def test_stations_deployed_without_images_are_published(self):
        """otono_2025 records five cameras installed in February 2025 and collected in
        June that appear in no image data at all. Dropping them would erase ~620
        camera-days and make them read as stations that never existed; publishing them
        with has_media=False keeps the discrepancy visible while it is resolved."""
        frame = deployments.build("otono_2025")
        empty = frame[~frame["has_media"]]
        self.assertEqual(sorted(empty["station_id"]),
                         ["CT21", "CT22", "CT24", "CT25", "CT26"])
        self.assertTrue((empty["field_days"] > 0).all())
        self.assertTrue((empty["note"] != "").all())


if __name__ == "__main__":
    unittest.main()
