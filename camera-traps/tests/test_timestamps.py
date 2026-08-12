"""Fixtures for timestamps.py — the wiring between clocks.py and the output rows.

Run:  python3 -m unittest discover -s tests -v

test_clocks.py asserts the RULE. This file asserts the PLUMBING: that the verdict
reached per segment actually lands on the right rows, that two segments of one
camera get two different offsets, and that the export gate stops ingest.

The single most important assertion here is
`test_two_segments_get_two_different_offsets`. Until 2026-07-31 this pipeline
applied one offset per station, and CT_18 — four resets — got one. Everything else
in the repair is worthless if that regresses.
"""

import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import timestamps
from camtrap import anchors, exports

STATION = 'CT01'          # canonical, so no station alias is needed
CAMPAIGN = 'unit_test'    # matches the fixture directory name

# The clock is right for the first run, then reverts to the 2017 epoch. The offset
# that repairs the second run is +8 years and change — and must NOT be applied to
# the first, whose stamps are already true.
REAL_START  = datetime(2025, 11, 20, 10, 0)
BOGUS_START = datetime(2017, 1, 1, 0, 0)


def _rows(start: datetime, n: int, counter0: int) -> list[dict]:
    return [
        {
            'Deployments': STATION,
            'RelativePath': STATION,
            'File': f'{(start + timedelta(hours=6 * i)):%m%d}{counter0 + i:04d}.JPG',
            'DateTime': f'{start + timedelta(hours=6 * i):%Y-%m-%d %H:%M:%S}',
        }
        for i in range(n)
    ]


def write_campaign(
    root: Path,
    *,
    campaign: str = 'unit_test',
    total_types: list[str] | None = None,
    anchors: list[dict] | None = None,
    reviewed_files: list[str] | None = None,
) -> Path:
    """A minimal campaign directory: total export, reviewed CSV, anchors."""
    campaign_dir = root / campaign
    campaign_dir.mkdir(parents=True, exist_ok=True)

    frames = _rows(REAL_START, 4, 1) + _rows(BOGUS_START, 4, 5)
    total = pd.DataFrame(frames)
    # Default: a swept export whose human frame proves the sweep happened.
    total[exports.OBSERVATION_TYPE_COLUMN] = (
        total_types if total_types is not None
        else ['human'] + ['blank'] * 3 + ['animal'] * 4
    )
    total.to_csv(campaign_dir / exports.TOTAL_EXPORT_FILENAME, index=False)

    keep = total if reviewed_files is None else total[total['File'].isin(reviewed_files)]
    reviewed = keep.copy()
    reviewed['scientificName'] = 'Puma concolor'
    reviewed['observationComments'] = ''
    reviewed['reviewOutcome'] = 'confirmed'
    reviewed['filePath'] = ''
    reviewed.to_csv(campaign_dir / 'new_labeled_data_reviewed.csv', index=False)

    anchor_rows = anchors if anchors is not None else []
    pd.DataFrame(anchor_rows, columns=[
        'station_id', 'anchor_type', 'real_datetime', 'camera_datetime',
        'source', 'notes', 'segment_index',
    ]).to_csv(campaign_dir / 'deployment_anchors.csv', index=False)

    return campaign_dir


def run(campaign_dir: Path, campaign: str = 'unit_test'):
    """Diagnose + repair, as main() does. Returns (corrected, report)."""
    total, _ = timestamps.load_total(campaign_dir)
    photos = timestamps.load_reviewed(campaign_dir / 'new_labeled_data_reviewed.csv')
    anchors = timestamps.load_anchors(campaign_dir / 'deployment_anchors.csv')
    diagnoses = timestamps.diagnose_campaign(total, anchors, campaign)
    return timestamps.repair_campaign(photos, total, diagnoses, campaign)


class _TmpCampaign(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class TestPerSegmentOffsets(_TmpCampaign):

    def test_two_segments_get_two_different_offsets(self):
        """THE regression test. One offset per station is the bug this replaced."""
        campaign_dir = write_campaign(self.root, anchors=[
            # Segment 0's clock was already right: offset zero.
            {'station_id': STATION, 'anchor_type': 'install',
             'real_datetime': '2025-11-20 10:00:00',
             'camera_datetime': '2025-11-20 10:00:00',
             'source': 'field_notebook', 'notes': '', 'segment_index': ''},
            # Segment 1 reverted to 2017; a mid-visit anchor dates its first frame.
            {'station_id': STATION, 'anchor_type': 'mid_visit',
             'real_datetime': '2025-12-01 00:00:00',
             'camera_datetime': '2017-01-01 00:00:00',
             'source': 'field_notebook', 'notes': '', 'segment_index': ''},
        ])
        corrected, _ = run(campaign_dir)

        seg0 = corrected[corrected['clock_segment'] == 0]
        seg1 = corrected[corrected['clock_segment'] == 1]
        self.assertEqual(len(seg0), 4)
        self.assertEqual(len(seg1), 4)

        # Segment 0 is untouched...
        self.assertEqual(
            pd.Timestamp(seg0['datetime_corrected'].min()), pd.Timestamp(REAL_START),
        )
        # ...and segment 1 is moved onto its anchor, not onto segment 0's offset.
        self.assertEqual(
            pd.Timestamp(seg1['datetime_corrected'].min()),
            pd.Timestamp('2025-12-01 00:00:00'),
        )
        self.assertTrue(corrected['valid_date'].all())
        self.assertTrue(corrected['valid_effort'].all())

    def test_a_segment_without_an_anchor_is_refused_and_the_other_is_not(self):
        campaign_dir = write_campaign(self.root, anchors=[
            {'station_id': STATION, 'anchor_type': 'install',
             'real_datetime': '2025-11-20 10:00:00',
             'camera_datetime': '2025-11-20 10:00:00',
             'source': 'field_notebook', 'notes': '', 'segment_index': ''},
        ])
        corrected, report = run(campaign_dir)

        seg0 = corrected[corrected['clock_segment'] == 0]
        seg1 = corrected[corrected['clock_segment'] == 1]
        self.assertTrue(seg0['valid_date'].all())
        self.assertFalse(seg1['valid_date'].any())
        self.assertTrue(seg1['datetime_corrected'].isna().all())
        self.assertEqual(
            set(seg1['repair_method']), {'no_anchor_in_segment'},
        )

    def test_valid_effort_is_station_level_not_per_segment(self):
        """One unknown segment takes the whole station out of the denominator —
        including the rows whose own date is fine."""
        campaign_dir = write_campaign(self.root, anchors=[
            {'station_id': STATION, 'anchor_type': 'install',
             'real_datetime': '2025-11-20 10:00:00',
             'camera_datetime': '2025-11-20 10:00:00',
             'source': 'field_notebook', 'notes': '', 'segment_index': ''},
        ])
        corrected, report = run(campaign_dir)
        self.assertFalse(corrected['valid_effort'].any())
        self.assertEqual(report.n_stations_no_effort, 1)

    def test_segment_index_on_an_anchor_rescues_an_unreachable_segment(self):
        """The escape hatch: an anchor containment cannot place, named explicitly."""
        campaign_dir = write_campaign(self.root, anchors=[
            {'station_id': STATION, 'anchor_type': 'install',
             'real_datetime': '2025-11-20 10:00:00',
             'camera_datetime': '2025-11-20 10:00:00',
             'source': 'field_notebook', 'notes': '', 'segment_index': ''},
            # camera_datetime is inside NO segment; segment_index says which one.
            {'station_id': STATION, 'anchor_type': 'retrieval',
             'real_datetime': '2025-12-05 00:00:00',
             'camera_datetime': '2019-06-06 00:00:00',
             'source': 'field_notebook', 'notes': 'checked by eye',
             'segment_index': '1'},
        ])
        corrected, _ = run(campaign_dir)
        seg1 = corrected[corrected['clock_segment'] == 1]
        self.assertTrue(seg1['valid_date'].all())
        self.assertEqual(set(seg1['repair_method']), {'offset_from_retrieval'})
        self.assertEqual(set(seg1['repair_anchor_source']), {'field_notebook'})

    def test_a_clean_camera_needs_no_anchor(self):
        campaign_dir = write_campaign(self.root)
        # Only the in-window run, so there is exactly one segment.
        total = pd.read_csv(campaign_dir / exports.TOTAL_EXPORT_FILENAME)
        total = total[total['DateTime'].str.startswith('2025')]
        total.to_csv(campaign_dir / exports.TOTAL_EXPORT_FILENAME, index=False)
        total.assign(
            scientificName='Puma concolor', observationComments='',
            reviewOutcome='confirmed', filePath='',
        ).to_csv(campaign_dir / 'new_labeled_data_reviewed.csv', index=False)

        corrected, _ = run(campaign_dir)
        self.assertEqual(set(corrected['repair_method']), {'clock_clean'})
        self.assertTrue(corrected['valid_date'].all())
        self.assertTrue(corrected['valid_time_of_day'].all())
        self.assertTrue(corrected['valid_effort'].all())


class TestApproximateAnchor(_TmpCampaign):

    def test_last_real_proxy_keeps_the_date_and_drops_the_time_of_day(self):
        campaign_dir = write_campaign(self.root, anchors=[
            {'station_id': STATION, 'anchor_type': 'install',
             'real_datetime': '2025-11-20 10:00:00',
             'camera_datetime': '2025-11-20 10:00:00',
             'source': 'field_notebook', 'notes': '', 'segment_index': ''},
            {'station_id': STATION, 'anchor_type': 'last_real_proxy',
             'real_datetime': '2025-12-10 09:00:00',
             'camera_datetime': '2017-01-01 18:00:00',
             'source': 'last_bogus_exif', 'notes': '', 'segment_index': ''},
        ])
        corrected, _ = run(campaign_dir)
        seg1 = corrected[corrected['clock_segment'] == 1]
        self.assertTrue(seg1['valid_date'].all())
        self.assertFalse(seg1['valid_time_of_day'].any())


class TestTheGateStopsIngest(_TmpCampaign):

    def test_animal_only_export_cannot_reach_diagnosis(self):
        campaign_dir = write_campaign(self.root, total_types=['animal'] * 8)
        with self.assertRaises(exports.ExportGateError):
            run(campaign_dir)

    def test_missing_total_export_is_fatal(self):
        campaign_dir = write_campaign(self.root)
        (campaign_dir / exports.TOTAL_EXPORT_FILENAME).unlink()
        with self.assertRaises(exports.ExportGateError):
            run(campaign_dir)

    def test_cli_returns_nonzero_on_a_rejected_export(self):
        write_campaign(self.root, total_types=['animal'] * 8)
        rc = timestamps.main([
            '--campaign', 'unit_test', '--data-root', str(self.root), '--dry-run',
        ])
        self.assertEqual(rc, 3)


class TestReviewedRowsMustBeCovered(_TmpCampaign):

    def test_a_reviewed_row_missing_from_the_total_export_aborts(self):
        """If the total export does not cover the reviewed rows it is not, in fact,
        all the images — and the diagnosis ran on a different set of frames."""
        campaign_dir = write_campaign(self.root)
        total = pd.read_csv(campaign_dir / exports.TOTAL_EXPORT_FILENAME)
        # Drop from the END: row 0 is the person frame the gate needs to see, and
        # losing it would fail this test for the wrong reason.
        total.iloc[:-1].to_csv(campaign_dir / exports.TOTAL_EXPORT_FILENAME, index=False)
        with self.assertRaises(ValueError) as cm:
            run(campaign_dir)
        self.assertIn('do not appear in', str(cm.exception))

    def test_allow_unmatched_marks_them_unusable_instead(self):
        campaign_dir = write_campaign(self.root)
        total = pd.read_csv(campaign_dir / exports.TOTAL_EXPORT_FILENAME)
        dropped = total.iloc[-1]['File']
        total.iloc[:-1].to_csv(campaign_dir / exports.TOTAL_EXPORT_FILENAME, index=False)

        loaded, _ = timestamps.load_total(campaign_dir)
        photos = timestamps.load_reviewed(campaign_dir / 'new_labeled_data_reviewed.csv')
        anchors = timestamps.load_anchors(campaign_dir / 'deployment_anchors.csv')
        diagnoses = timestamps.diagnose_campaign(loaded, anchors, 'unit_test')
        corrected, report = timestamps.repair_campaign(
            photos, loaded, diagnoses, 'unit_test', allow_unmatched=True,
        )
        row = corrected[corrected['File'] == dropped].iloc[0]
        self.assertEqual(row['repair_method'], timestamps.METHOD_NOT_IN_TOTAL)
        self.assertFalse(row['valid_date'])
        self.assertTrue(any('do not appear in' in w for w in report.warnings))


class TestAnchorSchema(_TmpCampaign):

    def test_segment_index_is_optional(self):
        campaign_dir = write_campaign(self.root)
        path = campaign_dir / 'deployment_anchors.csv'
        pd.DataFrame([{
            'station_id': STATION, 'anchor_type': 'install',
            'real_datetime': '2025-11-20 10:00:00',
            'camera_datetime': '2025-11-20 10:00:00',
            'source': 'field_notebook', 'notes': '',
        }]).to_csv(path, index=False)
        anchors = timestamps.load_anchors(path)
        self.assertIsNone(anchors[0].segment_index)

    def test_non_integer_segment_index_is_refused(self):
        campaign_dir = write_campaign(self.root, anchors=[{
            'station_id': STATION, 'anchor_type': 'install',
            'real_datetime': '2025-11-20 10:00:00',
            'camera_datetime': '2025-11-20 10:00:00',
            'source': 'field_notebook', 'notes': '', 'segment_index': 'first',
        }])
        with self.assertRaises(ValueError) as cm:
            timestamps.load_anchors(campaign_dir / 'deployment_anchors.csv')
        self.assertIn('segment_index', str(cm.exception))

    def test_window_needs_two_distinct_anchor_times(self):
        """With no field record, the anchors are the only source of a window."""
        one = [timestamps.Anchor(
            station_id=STATION, anchor_type='install',
            real_datetime=REAL_START, camera_datetime=REAL_START, source='t',
        )]
        self.assertIsNone(
            anchors.deployment_window(STATION, CAMPAIGN, one, None))
        two = one + [timestamps.Anchor(
            station_id=STATION, anchor_type='retrieval',
            real_datetime=datetime(2026, 5, 15, 12, 10),
            camera_datetime=datetime(2026, 5, 15, 12, 10), source='t',
        )]
        window = anchors.deployment_window(STATION, CAMPAIGN, two, None)
        self.assertIsNotNone(window)
        self.assertLess(window[0], REAL_START)      # tolerance widens it


if __name__ == '__main__':
    unittest.main()
