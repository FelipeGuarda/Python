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
        with has_media=False keeps the discrepancy visible."""
        frame = deployments.build("otono_2025")
        empty = frame[~frame["has_media"]]
        self.assertEqual(sorted(empty["station_id"]),
                         ["CT21", "CT22", "CT24", "CT25", "CT26"])
        self.assertTrue((empty["field_days"] > 0).all())
        self.assertTrue((empty["note"] != "").all())


class TestMediaStatusIsAReasonNotAMeasurement(unittest.TestCase):
    """`has_media` says whether stills are here; `media_status` says WHY not.

    These were one column until 2026-08-25, and conflating them published a false
    sentence for four station-campaigns: "no images in the canonical table" read as
    "the camera saw nothing" when the cameras had been recording video the whole
    time, stored outside this pipeline. The distinction decides a DENOMINATOR, so a
    regression here silently rescales every otono_2025 rate.
    """

    def test_the_four_video_only_stations_are_named_as_such(self):
        frame = deployments.build("otono_2025")
        video = frame[frame["media_status"] == "video_only_offline"]
        self.assertEqual(sorted(video["station_id"]),
                         ["CT22", "CT24", "CT25", "CT26"])
        # They were sampling, so their days are real effort -- the note must say so.
        self.assertTrue(video["note"].str.contains("EXCLUDE").all())

    def test_ct21_recorded_nothing_and_is_not_lumped_with_them(self):
        """CT21's own field note says "SD vacia" -- the camera was dead, not filming.
        It contributes no effort to any question, so it must not share a verdict with
        the four that do."""
        frame = deployments.build("otono_2025")
        row = frame[frame["station_id"] == "CT21"].iloc[0]
        self.assertEqual(row["media_status"], "card_failure")

    def test_stations_with_stills_are_in_canonical(self):
        for campaign in PUBLISHED_CAMPAIGNS:
            frame = deployments.build(campaign)
            with_media = frame[frame["has_media"]]
            self.assertTrue(
                (with_media["media_status"] == deployments.STATUS_IN_CANONICAL).all(),
                f"{campaign}: a station with stills carries a non-canonical status",
            )

    def test_an_undeclared_gap_reports_unexplained_rather_than_nothing(self):
        """The failure this guards is silence. If a future campaign has a station with
        no stills and nobody writes down why, it must NOT inherit a reassuring note --
        an unexplained gap is a question that has not been asked, and absorbing it
        into an effort figure is how a wrong denominator looks right."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "otono_2025").mkdir()
            # Field record dates both ends for a station with no canonical table at all.
            with (root / FIELD_NOTES_FILENAME).open("w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["station_id", "visit_type", "visit_date", "visit_time",
                            "campaign_opened", "campaign_closed"])
                w.writerow(["CT99", "install", "2025-02-04", "", "otono_2025", ""])
                w.writerow(["CT99", "revision", "2025-06-05", "", "", "otono_2025"])
            frame = deployments.build("otono_2025", root=root)
            row = frame[frame["station_id"] == "CT99"].iloc[0]
            self.assertEqual(row["media_status"], deployments.STATUS_UNEXPLAINED)
            self.assertIn("media_absence.csv", row["note"])

    def test_a_misspelled_reason_is_refused_not_ignored(self):
        """A typo in the reason column must not read as a licence to count the
        camera-days. Fail-closed, same posture as the flatten preconditions."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "otono_2025").mkdir()
            with (root / FIELD_NOTES_FILENAME).open("w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["station_id", "visit_type", "visit_date", "visit_time",
                            "campaign_opened", "campaign_closed"])
                w.writerow(["CT99", "install", "2025-02-04", "", "otono_2025", ""])
                w.writerow(["CT99", "revision", "2025-06-05", "", "", "otono_2025"])
            with (root / deployments.MEDIA_ABSENCE_FILENAME).open(
                    "w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["campaign", "station_id", "reason"])
                w.writerow(["otono_2025", "CT99", "video_only_ofline"])   # typo
            with self.assertRaises(ValueError) as cm:
                deployments.build("otono_2025", root=root)
            self.assertIn("video_only_ofline", str(cm.exception))

    def test_the_declared_absences_match_the_committed_file(self):
        """media_absence.csv is a data file, so it can drift from the stations that
        actually have no stills. Every declared row must correspond to a real gap --
        a leftover declaration would silently license effort for a station that has
        since been ingested."""
        declared = pd.read_csv(CAMPAIGNS_ROOT / deployments.MEDIA_ABSENCE_FILENAME,
                               dtype=str, keep_default_na=False)
        for campaign, group in declared.groupby("campaign"):
            frame = deployments.build(campaign)
            without = set(frame.loc[~frame["has_media"], "station_id"])
            self.assertEqual(set(group["station_id"]), without,
                             f"{campaign}: declared absences != stations without stills")


if __name__ == "__main__":
    unittest.main()
