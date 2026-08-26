"""Fixtures for camtrap.anchors — what the field record may and may not assert.

Run:  python3 -m unittest discover -s tests -v

stdlib unittest on purpose: these must run on the office Windows box and on the
Linux laptop without either growing a test dependency.

Two assertions here carry the weight, and both encode a mistake this module made
before it was finished:

  `test_clean_clock_gets_no_anchor` — CT01. The notebook says the deployment ran
  2025-11-24 → 2026-05-13 while the frames run 2025-11-26 → 2026-05-14 across one
  coherent segment. Turning that visit into an anchor applies a two-day offset to a
  clock that was never wrong.

  `test_counter_0001_cannot_witness_a_visit` — CT18 segment 0. Its only candidate is
  a counter-0001 frame five days after the install visit. Pairing them yields a -5 day
  offset applied to ten frames whose clock was correct; the camera simply did not
  trigger for five days. A frame must SHOW someone at the camera to date a visit.
"""

import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camtrap import anchors, clocks

CAMPAIGN = 'otono_2026'
STATION = 'CT18'

INSTALL = datetime(2025, 11, 14)
RETRIEVAL = datetime(2026, 5, 15, 12, 10)


# ── fixtures ──────────────────────────────────────────────────────────────────

def frames(start, n, counter0, *, step_hours=6):
    return [
        {
            'file_name': f'{(start + timedelta(hours=step_hours * i)):%m%d}'
                         f'{counter0 + i:04d}.JPG',
            'camera_datetime': start + timedelta(hours=step_hours * i),
        }
        for i in range(n)
    ]


def diagnose(rows, window=None, station=STATION):
    return clocks.diagnose(pd.DataFrame(rows), station, window=window)


def candidates(*rows) -> pd.DataFrame:
    cols = ['station', 'camera_num', 'file_name', 'camera_datetime',
            'candidate_kind', 'clock_segment']
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(list(rows))
    df['camera_datetime'] = pd.to_datetime(df['camera_datetime'])
    return df


def candidate(when, kind, segment, name='X.JPG'):
    return {'station': STATION, 'camera_num': 18, 'file_name': name,
            'camera_datetime': when, 'candidate_kind': kind,
            'clock_segment': segment}


def field_csv(rows) -> anchors.FieldRecord:
    """Build a FieldRecord through the CSV, so the parsing is under test too."""
    tmp = Path(tempfile.mkdtemp()) / anchors.FIELD_NOTES_FILENAME
    pd.DataFrame(rows).to_csv(tmp, index=False)
    return anchors.FieldRecord.load(tmp)


def visit(station=STATION, *, opened='', date='', time='',
          flags='', kind='revision'):
    """There is no `closed=`: which campaign a visit closes is derived from the
    station's own sequence, so a closing is expressed by putting an opener before it."""
    return {'station_id': station, 'visit_type': kind,
            'campaign_opened': opened,
            'visit_date': date, 'visit_time': time, 'data_flags': flags,
            'source_sheet': 'test'}


def both_ends(open_date='2025-11-14', close_date='2026-05-15',
              open_time='', close_time='12:10:00', flags=''):
    return field_csv([
        visit(opened=CAMPAIGN, date=open_date, time=open_time, flags=flags),
        visit(date=close_date, time=close_time),
    ])


# ── the deployment window ─────────────────────────────────────────────────────

class TestTheClosingCampaignIsDerived(unittest.TestCase):
    """Since 2026-08-26 `campaign_closed` is not recorded. Measured against the 107
    legacy rows the derivation reproduced 105 of 106 dated values; the exception was
    an assertion the rest of the project already contradicted."""

    def test_a_revision_closes_what_the_previous_visit_opened(self):
        record = field_csv([
            visit(kind='instalacion', opened='otono_2025', date='2025-02-04'),
            visit(kind='revision', opened='primavera_2025', date='2025-06-05'),
        ])
        self.assertIsNotNone(record.closing(STATION, 'otono_2025'))
        self.assertIsNotNone(record.opening(STATION, 'primavera_2025'))

    def test_an_installation_closes_nothing(self):
        """There was no card in the ground to close."""
        record = field_csv([visit(kind='instalacion', opened='otono_2025',
                                 date='2025-02-04')])
        self.assertIsNone(record.closing(STATION, 'otono_2025'))

    def test_a_mantencion_closes_nothing(self):
        """The card is not touched, so the campaign it belongs to is still open. This
        is why the derivation reads the visit type and not `campaign_opened` alone."""
        record = field_csv([
            visit(kind='instalacion', opened='otono_2025', date='2025-02-04'),
            visit(kind='mantencion', opened='otono_2025', date='2025-04-01'),
            visit(kind='revision', opened='primavera_2025', date='2025-06-05'),
        ])
        closing = record.closing(STATION, 'otono_2025')
        self.assertIsNotNone(closing)
        self.assertEqual(closing.visit_date.date().isoformat(), '2025-06-05')

    def test_a_retiro_leaves_nothing_in_the_ground(self):
        """Without the reset, a station lifted and later reinstalled would derive its
        reinstall as closing the campaign that ended before the gap. No `retiro` row
        exists in the record yet, so this branch is held here rather than by data."""
        record = field_csv([
            visit(kind='instalacion', opened='otono_2025', date='2025-02-04'),
            visit(kind='retiro', date='2025-06-05'),
            visit(kind='instalacion', opened='otono_2026', date='2025-11-24'),
            visit(kind='revision', opened='primavera_2026', date='2026-05-14'),
        ])
        self.assertIsNotNone(record.closing(STATION, 'otono_2025'))
        self.assertEqual(record.closing(STATION, 'otono_2025')
                         .visit_date.date().isoformat(), '2025-06-05')
        self.assertEqual(record.closing(STATION, 'otono_2026')
                         .visit_date.date().isoformat(), '2026-05-14')

    def test_the_sequence_is_the_dates_not_the_file_order(self):
        """A workbook transcribed out of order must derive the same closings."""
        record = field_csv([
            visit(kind='revision', opened='primavera_2025', date='2025-06-05'),
            visit(kind='instalacion', opened='otono_2025', date='2025-02-04'),
        ])
        self.assertIsNotNone(record.closing(STATION, 'otono_2025'))

    def test_an_undated_visit_closes_nothing(self):
        """CT27's placeholder row: the station exists, the install was never written
        down. It cannot be placed in the sequence, so it closes nothing and does not
        advance the carry."""
        record = field_csv([
            visit(kind='unrecorded'),
            visit(kind='revision', opened='otono_2026', date='2025-12-11'),
        ])
        self.assertIsNone(record.closing(STATION, 'otono_2025'))
        self.assertIsNone(record.closing(STATION, 'otono_2026'))

    def test_stations_do_not_leak_into_each_other(self):
        record = field_csv([
            visit(station='CT01', kind='instalacion', opened='otono_2025',
                  date='2025-02-04'),
            visit(station='CT02', kind='revision', opened='primavera_2025',
                  date='2025-06-05'),
        ])
        self.assertIsNone(record.closing('CT02', 'otono_2025'))

    def test_a_file_that_still_records_the_closing_is_refused(self):
        """A pre-reshape copy, or a `build_field_notes.py` run that reverted the
        curated rows. Reading it while ignoring the column would reinterpret it."""
        tmp = Path(tempfile.mkdtemp()) / anchors.FIELD_NOTES_FILENAME
        pd.DataFrame([{'station_id': STATION, 'visit_type': 'revision',
                       'campaign_closed': 'otono_2025', 'campaign_opened': '',
                       'visit_date': '2025-06-05'}]).to_csv(tmp, index=False)
        with self.assertRaises(ValueError) as cm:
            anchors.FieldRecord.load(tmp)
        self.assertIn(anchors.RECORDED_CLOSE_COLUMN, str(cm.exception))


class TestDeploymentWindow(unittest.TestCase):

    def test_window_brackets_the_visits_with_tolerance(self):
        w = both_ends().window(STATION, CAMPAIGN)
        self.assertIsNotNone(w)
        self.assertEqual(w[0], datetime(2025, 11, 14, 12, 0)
                         - anchors.VISIT_WINDOW_TOLERANCE)
        self.assertEqual(w[1], datetime(2026, 5, 15, 12, 10)
                         + anchors.VISIT_WINDOW_TOLERANCE)

    def test_both_ends_are_required(self):
        """A half-open window tests one edge and passes everything past the other,
        which reads as a check while being none."""
        only_open = field_csv([visit(opened=CAMPAIGN, date='2025-11-14')])
        self.assertIsNone(only_open.window(STATION, CAMPAIGN))

    def test_a_closing_cannot_exist_without_an_opening(self):
        """The other half-open shape stopped being reachable on 2026-08-26. A closing
        is derived from the carry left by an earlier visit to the same station, so a
        lone revision closes nothing rather than closing a campaign nobody opened."""
        lone = field_csv([visit(date='2026-05-15')])
        self.assertIsNone(lone.closing(STATION, CAMPAIGN))
        self.assertIsNone(lone.window(STATION, CAMPAIGN))

    def test_field_record_preferred_over_anchors(self):
        two = [
            clocks.Anchor(STATION, 'install', datetime(2020, 1, 1),
                          datetime(2020, 1, 1)),
            clocks.Anchor(STATION, 'retrieval', datetime(2020, 6, 1),
                          datetime(2020, 6, 1)),
        ]
        w = anchors.deployment_window(STATION, CAMPAIGN, two, both_ends())
        self.assertEqual(w[0].year, 2025)      # the visits won, not the anchors

    def test_falls_back_to_anchors_when_no_field_record(self):
        two = [
            clocks.Anchor(STATION, 'install', INSTALL, INSTALL),
            clocks.Anchor(STATION, 'retrieval', RETRIEVAL, RETRIEVAL),
        ]
        empty = field_csv([visit(station='CT99', opened=CAMPAIGN, date='2025-01-01')])
        w = anchors.deployment_window(STATION, CAMPAIGN, two, empty)
        self.assertIsNotNone(w)
        self.assertEqual(w[0], INSTALL - anchors.WINDOW_TOLERANCE)

    def test_ct01_stays_inside_its_visit_window(self):
        """The regression that set VISIT_WINDOW_TOLERANCE. CT01's frames run two days
        past the notebook at one end and one day past at the other; under the 1 h
        anchor tolerance that would read as a clock failure."""
        rows = frames(datetime(2025, 11, 26, 13, 39), 12, 1, step_hours=24 * 13)
        record = both_ends(open_date='2025-11-24', close_date='2026-05-13',
                           close_time='12:00:00')
        d = diagnose(rows, window=record.window(STATION, CAMPAIGN))
        self.assertFalse(d.has_clock_failure)


# ── evidence ──────────────────────────────────────────────────────────────────

class TestEvidence(unittest.TestCase):

    def test_labelled_human_outranks_megadetector(self):
        """The sweep is only worth doing if confirmed evidence beats a guess."""
        self.assertLess(
            anchors.EVIDENCE_RANK.index(anchors.EVIDENCE_HUMAN_LABELLED),
            anchors.EVIDENCE_RANK.index(anchors.EVIDENCE_PERSON_DETECTION),
        )

    def test_navigational_evidence_is_not_witness_evidence(self):
        self.assertNotIn(anchors.EVIDENCE_COUNTER_0001, anchors.EVIDENCE_WITNESS)
        self.assertNotIn(anchors.EVIDENCE_SEGMENT_EDGE, anchors.EVIDENCE_WITNESS)
        self.assertIn(anchors.EVIDENCE_HUMAN_LABELLED, anchors.EVIDENCE_WITNESS)


# ── proposals ─────────────────────────────────────────────────────────────────

class TestPropose(unittest.TestCase):

    def _propose(self, d, record, cands, existing=()):
        return anchors.propose(d, CAMPAIGN, record, cands, list(existing))

    def test_clean_clock_gets_no_anchor(self):
        """CT01. A visit dates a VISIT, not a clock; forcing it on a coherent camera
        applies the notebook's imprecision as an offset to correct data."""
        record = both_ends(open_date='2025-11-24', close_date='2026-05-13',
                           close_time='12:00:00')
        rows = frames(datetime(2025, 11, 26, 13, 39), 12, 1, step_hours=24 * 13)
        d = diagnose(rows, window=record.window(STATION, CAMPAIGN))
        out = self._propose(d, record, candidates(
            candidate(datetime(2025, 11, 26, 14, 5),
                      anchors.EVIDENCE_HUMAN_LABELLED, 0)))
        self.assertEqual([p.status for p in out], [anchors.NOT_NEEDED])
        self.assertEqual(out[0].anchor_type, '')
        self.assertIsNone(out[0].real_datetime)

    def test_clean_but_unwindowed_is_reported_as_unverified(self):
        """CT27. No install record, so the in-window test never ran — a forward jump
        would have been invisible. That is an absence of evidence, not a verdict."""
        rows = frames(datetime(2026, 1, 5), 8, 1, step_hours=24)
        d = diagnose(rows, window=None)
        record = field_csv([visit(station='CT99', opened=CAMPAIGN, date='2025-11-14')])
        out = self._propose(d, record, candidates())
        self.assertEqual(out[0].status, anchors.NOT_NEEDED)
        self.assertEqual(out[0].evidence, anchors.UNVERIFIED)

    def test_counter_0001_cannot_witness_a_visit(self):
        """CT18 segment 0: the only candidate is a counter-0001 frame five days after
        the install. It marks where to look, not when it was taken."""
        rows = (frames(datetime(2025, 11, 19, 6, 41), 10, 1)
                + frames(datetime(2017, 1, 1), 12, 101))
        d = diagnose(rows)
        out = self._propose(d, both_ends(), candidates(
            candidate(datetime(2025, 11, 19, 6, 41),
                      anchors.EVIDENCE_COUNTER_0001, 0, name='11190001.JPG')))
        seg0 = [p for p in out if p.segment_index == 0]
        self.assertEqual(len(seg0), 1)
        self.assertEqual(seg0[0].status, anchors.NEEDS_REVIEW)
        self.assertEqual(seg0[0].anchor_type, 'unrepairable_pending')
        self.assertIsNone(seg0[0].real_datetime)
        self.assertIn('WITNESSING', seg0[0].why)

    def test_witness_frame_and_timed_visit_is_ready_and_exact(self):
        rows = (frames(datetime(2025, 11, 19, 6, 41), 10, 1)
                + frames(datetime(2017, 1, 1), 12, 101))
        d = diagnose(rows)
        last = len(d.segments) - 1
        seen = datetime(2017, 1, 3, 5, 0)
        out = self._propose(d, both_ends(), candidates(
            candidate(seen, anchors.EVIDENCE_HUMAN_LABELLED, last, name='RET.JPG')))
        got = [p for p in out if p.segment_index == last]
        self.assertEqual(got[0].status, anchors.READY)
        self.assertEqual(got[0].anchor_type, 'retrieval')
        self.assertIn(got[0].anchor_type, clocks.ANCHOR_TYPES_EXACT)
        self.assertEqual(got[0].camera_datetime, seen)
        self.assertEqual(got[0].real_datetime, RETRIEVAL)

    def test_date_only_visit_yields_an_approximate_anchor(self):
        """Every otoño 2026 opening visit is date-only. Asserting an hour nobody wrote
        down is how CT18's install anchor came to claim 14:00 against a bare date."""
        rows = (frames(datetime(2025, 11, 19, 6, 41), 10, 1)
                + frames(datetime(2017, 1, 1), 12, 101))
        d = diagnose(rows)
        out = self._propose(d, both_ends(), candidates(
            candidate(datetime(2025, 11, 19, 8, 0),
                      anchors.EVIDENCE_HUMAN_LABELLED, 0, name='INS.JPG')))
        seg0 = [p for p in out if p.segment_index == 0][0]
        self.assertEqual(seg0.anchor_type, 'visit_date_only')
        self.assertIn(seg0.anchor_type, clocks.ANCHOR_TYPES_APPROXIMATE)
        self.assertEqual(seg0.status, anchors.NEEDS_REVIEW)
        self.assertEqual(seg0.real_datetime.hour, anchors.ASSUMED_VISIT_HOUR.hour)

    def test_approximate_anchor_never_claims_a_time_of_day(self):
        """The reason the type matters: repair_plan must not let activity analysis
        see a segment whose hour was assumed."""
        a = clocks.Anchor(STATION, 'visit_date_only', datetime(2025, 11, 14, 12),
                          datetime(2025, 11, 19, 6, 41))
        self.assertFalse(a.exact)

    def test_interior_segment_is_refused_not_guessed(self):
        # Each run must step BACKWARDS from the previous one to read as a reset; a
        # later start date would just continue the same chronology.
        rows = (frames(datetime(2025, 11, 19, 6, 41), 10, 1)
                + frames(datetime(2017, 3, 1), 12, 101)
                + frames(datetime(2017, 1, 1), 12, 201))
        d = diagnose(rows)
        self.assertGreaterEqual(len(d.segments), 3)
        out = self._propose(d, both_ends(), candidates(
            candidate(datetime(2017, 1, 2), anchors.EVIDENCE_HUMAN_LABELLED, 1)))
        interior = [p for p in out if p.segment_index == 1][0]
        self.assertEqual(interior.anchor_type, 'unrepairable_pending')
        self.assertIn('interior', interior.why)

    def test_ambiguous_visit_date_may_not_anchor(self):
        """CT27's only visit is 2025-11-12 or 2025-12-11, a month apart. Choosing one
        would be a coin flip recorded as a fact."""
        rows = (frames(datetime(2025, 11, 19, 6, 41), 10, 1)
                + frames(datetime(2017, 1, 1), 12, 101))
        d = diagnose(rows)
        record = both_ends(flags='date_ambiguous: Excel stored 2025-11-12; '
                                 'day/month swap 2025-12-11 is equally plausible')
        out = self._propose(d, record, candidates(
            candidate(datetime(2025, 11, 19, 8, 0),
                      anchors.EVIDENCE_HUMAN_LABELLED, 0)))
        seg0 = [p for p in out if p.segment_index == 0][0]
        self.assertEqual(seg0.anchor_type, 'unrepairable_pending')
        self.assertIn('unsettled', seg0.why)

    def test_segment_already_anchored_is_not_reproposed(self):
        """Existing anchors mostly leave segment_index blank and find their segment by
        containment, so the check must go through assign_anchors."""
        rows = (frames(datetime(2025, 11, 19, 6, 41), 10, 1)
                + frames(datetime(2017, 1, 1), 12, 101))
        d = diagnose(rows)
        existing = [clocks.Anchor(STATION, 'install', datetime(2025, 11, 19, 6, 41),
                                  datetime(2025, 11, 19, 6, 41))]
        out = self._propose(d, both_ends(), candidates(
            candidate(datetime(2025, 11, 19, 8, 0),
                      anchors.EVIDENCE_HUMAN_LABELLED, 0)), existing)
        self.assertEqual([p for p in out if p.segment_index == 0], [])

    def test_unordered_but_clean_station_needs_nothing(self):
        """The P1 asymmetry, and it must match repair_plan: an in-window sequence that
        never contradicts itself demonstrably never reset, so failing the ordering
        precondition does not condemn it. This is what keeps otoño 2026's five
        flattened wrap cameras usable."""
        rows = [{'file_name': 'IMG.JPG', 'camera_datetime': datetime(2026, 1, 5)},
                {'file_name': 'IMG2.JPG', 'camera_datetime': datetime(2026, 2, 5)}]
        d = diagnose(rows, window=both_ends().window(STATION, CAMPAIGN))
        self.assertFalse(d.ordered)
        self.assertFalse(d.has_clock_failure)
        out = self._propose(d, both_ends(), candidates())
        self.assertEqual(out[0].status, anchors.NOT_NEEDED)

    def test_unordered_with_a_failure_is_refused_whatever_the_notebook_says(self):
        """Once something IS wrong, no anchor can be placed without capture order —
        there is no way to say which run of the clock a frame belongs to."""
        rows = [{'file_name': 'IMG.JPG', 'camera_datetime': datetime(2026, 1, 5)},
                {'file_name': 'IMG2.JPG', 'camera_datetime': datetime(2017, 1, 1)}]
        d = diagnose(rows, window=both_ends().window(STATION, CAMPAIGN))
        self.assertFalse(d.ordered)
        self.assertTrue(d.has_clock_failure)
        out = self._propose(d, both_ends(), candidates())
        self.assertEqual(out[0].anchor_type, 'unrepairable_pending')
        self.assertIn('capture order', out[0].why)

    def test_refusals_are_written_down_not_omitted(self):
        """A station absent from the anchor file and one known to be unanchorable look
        identical downstream, and only one of them is a decision."""
        rows = (frames(datetime(2025, 11, 19, 6, 41), 10, 1)
                + frames(datetime(2017, 1, 1), 12, 101))
        d = diagnose(rows)
        out = self._propose(d, both_ends(), candidates())
        df = anchors.to_frame(out)
        self.assertTrue(len(df))
        self.assertTrue((df['anchor_type'] == 'unrepairable_pending').all())
        self.assertEqual(list(df.columns), anchors.PROPOSAL_COLUMNS)


if __name__ == '__main__':
    unittest.main()
