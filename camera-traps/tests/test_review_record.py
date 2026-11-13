"""How review decisions reach the reviewed CSV — `phase1_labeling/review_record.py`.

Run:  python -m unittest tests.test_review_record -v

The tests that matter most are the two "untouched" ones: the retro tag pass runs on
campaigns whose reviewed CSV is a record of months of work, and the only change it is
allowed to make is the `observationTags` cell. Measured 2026-11-13: a rebuild through
the normal export would have erased every otoño 2025 decision.
"""

import csv
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camtrap import copresence
from phase1_labeling import review_record as rr
from phase1_labeling.review_record import Decision

CAMPAIGNS = Path(__file__).resolve().parents[1] / "data" / "campaigns"
PUBLISHED = ("otono_2025", "primavera_2025", "otono_2026")

HEADER = ["filePath", "RelativePath", "File", "observationType", "scientificName",
          "classifiedBy", "classificationTimestamp", "observationTags",
          "observationComments", "reviewOutcome"]
OLD_DATE = "2025-07-01T10:00:00"
TAGS = copresence.join([copresence.TAG_HUMAN])


def _row(path, species="", outcome="", tags="", when="", kind="animal"):
    return {"filePath": path, "RelativePath": "", "File": "", "observationType": kind,
            "scientificName": "", "classifiedBy": "", "classificationTimestamp": when,
            "observationTags": tags, "observationComments": species,
            "reviewOutcome": outcome}


def _write(path: Path, rows, newline="\r\n"):
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=HEADER, lineterminator=newline)
    w.writeheader()
    w.writerows(rows)
    path.write_bytes(("﻿" + buf.getvalue()).encode("utf-8"))


def _read(path: Path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


class _TmpCampaign(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.reviewed = self.dir / rr.REVIEWED_FILENAME

    def tearDown(self):
        self._tmp.cleanup()


class TestTheCommittedRecordsSurviveUntouched(unittest.TestCase):
    def test_a_save_with_no_tags_reproduces_each_reviewed_csv_byte_for_byte(self):
        for campaign in PUBLISHED:
            src = CAMPAIGNS / campaign / rr.REVIEWED_FILENAME
            with self.subTest(campaign=campaign), tempfile.TemporaryDirectory() as tmp:
                shutil.copy(src, tmp)
                rr.write_tags(Path(tmp), {})
                self.assertEqual((Path(tmp) / rr.REVIEWED_FILENAME).read_bytes(),
                                 src.read_bytes())

    def test_otono_2025_backslash_paths_are_found(self):
        """Its reviewed CSV writes CT09\\x.JPG; the classifier CSV writes CT09/x.JPG."""
        decisions = rr.read_decisions(CAMPAIGNS / "otono_2025")
        self.assertEqual(len(decisions), 830)
        self.assertTrue(all("\\" not in k for k in decisions))


class TestWriteTags(_TmpCampaign):
    def setUp(self):
        super().setUp()
        _write(self.reviewed, [
            _row("CT09\\A.JPG", "Perro", "confirmed", when=OLD_DATE),
            _row("CT09\\B.JPG", "Perro", "corrected", when=OLD_DATE),
            _row("CT10/C.JPG", "Puma", "confirmed", when=OLD_DATE),
        ])

    def test_a_tag_changes_exactly_one_cell(self):
        before = _read(self.reviewed)
        rr.write_tags(self.dir, {"ct09/b.jpg": TAGS})
        after = _read(self.reviewed)
        changed = [(i, k) for i, (a, b) in enumerate(zip(before, after))
                   for k in a if a[k] != b[k]]
        self.assertEqual(changed, [(1, "observationTags")])
        self.assertEqual(after[1]["observationTags"], TAGS)

    def test_line_endings_and_bom_are_kept(self):
        rr.write_tags(self.dir, {"ct09/a.jpg": TAGS})
        raw = self.reviewed.read_bytes()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
        self.assertEqual(raw.count(b"\r\n"), 4)

    def test_a_tag_the_priority_forbids_saves_nothing(self):
        """with:human on a vehicle row: the ingest would refuse it, so the app must."""
        _write(self.reviewed, [_row("CT01/V.JPG", "", "confirmed", kind="vehicle")])
        before = self.reviewed.read_bytes()
        with self.assertRaises(rr.ReviewRecordError):
            rr.write_tags(self.dir, {"ct01/v.jpg": TAGS})
        self.assertEqual(self.reviewed.read_bytes(), before)

    def test_an_image_not_in_the_file_saves_nothing(self):
        before = self.reviewed.read_bytes()
        with self.assertRaises(rr.ReviewRecordError):
            rr.write_tags(self.dir, {"ct09/a.jpg": TAGS, "ct99/zzz.jpg": TAGS})
        self.assertEqual(self.reviewed.read_bytes(), before)


class TestReadDecisions(_TmpCampaign):
    def test_a_decision_comes_back_whole(self):
        _write(self.reviewed, [
            _row("CT09\\A.JPG", "Perro", "corrected", "WITH:HUMAN", OLD_DATE),
            _row("CT09/U.JPG"),   # in the file, never decided
        ])
        got = rr.read_decisions(self.dir)
        self.assertEqual(set(got), {"ct09/a.jpg"})
        self.assertEqual(got["ct09/a.jpg"],
                         Decision("Perro", "corrected", TAGS, OLD_DATE, "CT09/A.JPG"))

    def test_no_file_means_no_decisions(self):
        self.assertEqual(rr.read_decisions(self.dir), {})

    def test_backslash_and_slash_paths_are_one_image(self):
        self.assertEqual(rr.row_key({"filePath": "CT09\\A.JPG"}),
                         rr.row_key({"filePath": "ct09/a.jpg"}))
        self.assertEqual(rr.row_key({"filePath": "", "RelativePath": "CT09", "File": "A.JPG"}),
                         "ct09/a.jpg")


class TestWriteReview(_TmpCampaign):
    def setUp(self):
        super().setUp()
        self.classified = [_row("CT09/A.JPG"), _row("CT09/B.JPG"), _row("CT10/C.JPG")]

    def test_past_review_dates_are_kept_and_new_ones_set(self):
        _write(self.reviewed, [_row("CT09\\A.JPG", "Perro", "confirmed", when=OLD_DATE),
                               _row("CT09\\B.JPG"), _row("CT10\\C.JPG")])
        decisions = rr.read_decisions(self.dir)
        decisions["ct10/c.jpg"] = Decision("Puma", "confirmed", TAGS, "2026-11-13T12:00:00")
        rr.write_review(self.classified, HEADER, decisions, {"Puma": "Puma concolor"}, self.dir)
        out = {r["filePath"]: r for r in _read(self.reviewed)}
        self.assertEqual(out["CT09/A.JPG"]["classificationTimestamp"], OLD_DATE)
        self.assertEqual(out["CT10/C.JPG"]["classificationTimestamp"], "2026-11-13T12:00:00")
        self.assertEqual(out["CT10/C.JPG"]["observationTags"], TAGS)
        self.assertEqual(out["CT10/C.JPG"]["scientificName"], "Puma concolor")
        self.assertEqual(out["CT09/B.JPG"]["reviewOutcome"], "")

    def test_a_save_that_would_drop_decided_images_is_refused(self):
        """otoño 2025 has 12 reviewed images the classifier CSV does not."""
        _write(self.reviewed, [_row("CT09\\A.JPG", "Perro", "confirmed", when=OLD_DATE),
                               _row("CT99\\GONE.JPG", "Puma", "confirmed", when=OLD_DATE)])
        before = self.reviewed.read_bytes()
        with self.assertRaises(rr.ReviewRecordError) as ctx:
            rr.write_review(self.classified, HEADER, rr.read_decisions(self.dir), {}, self.dir)
        self.assertIn("Etiquetas", str(ctx.exception))
        self.assertEqual(self.reviewed.read_bytes(), before)

    def test_a_forbidden_tag_saves_nothing(self):
        rows = [_row("CT01/V.JPG", kind="vehicle")]
        with self.assertRaises(rr.ReviewRecordError):
            rr.write_review(rows, HEADER, {"ct01/v.jpg": Decision("x", "confirmed", TAGS)},
                            {}, self.dir)
        self.assertFalse(self.reviewed.exists())


class TestJoin(unittest.TestCase):
    def test_order_is_the_vocabulary_order_whatever_the_ticks(self):
        both = [copresence.TAG_VEHICLE, copresence.TAG_HUMAN]
        self.assertEqual(copresence.join(both), copresence.compose(True, True))
        self.assertEqual(copresence.join([]), "")


if __name__ == "__main__":
    unittest.main()
