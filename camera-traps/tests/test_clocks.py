"""Fixtures for camtrap.clocks — Felipe's scenario taxonomy A–G plus the cases that
produced wrong answers during the 2026-07-30 analysis.

Run:  python3 -m unittest discover -s tests -v

stdlib unittest on purpose: these must run on the office Windows box and on the
Linux laptop without either growing a test dependency.

Scenarios A–E are Felipe's; F (zero anchors — the state of the legacy archive) and
G (>1 split AND dead at retrieval — otoño 2026 CT_18's actual case) were the two
cells missing from his table. They are encoded as FIXTURES, not as code branches:
the module implements one rule and these assert it produces the right verdict for
each shape.
"""

import unittest
from datetime import datetime, timedelta

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camtrap import clocks, exports


# Real deployment window, matching otoño 2026's shape.
WINDOW = (datetime(2025, 11, 14, 14, 0), datetime(2026, 5, 15, 12, 10))

STATION = 'CT18'


def frame(dt, counter, *, mmdd=None, ext='JPG', dcim=None):
    """One row. The filename encodes MMDD+counter, coherent unless overridden."""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    stamp = mmdd if mmdd is not None else f'{dt:%m%d}'
    row = {'file_name': f'{stamp}{counter:04d}.{ext}', 'camera_datetime': dt}
    if dcim is not None:
        row['dcim_folder'] = dcim
    return row


def run_of(start, n, counter_start, *, step_hours=6, mmdd=None, dcim=None):
    """A coherent run of n frames, 6 h apart, with consecutive counters."""
    if isinstance(start, str):
        start = datetime.fromisoformat(start)
    return [
        frame(start + timedelta(hours=step_hours * i), counter_start + i,
              mmdd=mmdd, dcim=dcim)
        for i in range(n)
    ]


def anchor(kind, real, camera, **kw):
    if isinstance(real, str):
        real = datetime.fromisoformat(real)
    if isinstance(camera, str):
        camera = datetime.fromisoformat(camera)
    return clocks.Anchor(
        station_id=STATION, anchor_type=kind,
        real_datetime=real, camera_datetime=camera,
        source='test', **kw,
    )


def diagnose_and_plan(rows, anchors, window=WINDOW):
    d = clocks.diagnose(pd.DataFrame(rows), STATION, window=window)
    repairs, notes = clocks.repair_plan(d, anchors)
    return d, repairs, notes


# =============================================================================
# Filename grammar
# =============================================================================

class TestFilenameGrammar(unittest.TestCase):

    def test_plain_name(self):
        self.assertEqual(clocks.parse_filename('01190313.JPG'), ('0119', 313))

    def test_collision_prefixed_name(self):
        # 24 real files in otoño 2026 CT_14 look like this.
        self.assertEqual(
            clocks.parse_filename('102EK113_01190313.JPG'), ('0119', 313)
        )

    def test_hand_renamed_name_does_not_parse(self):
        # otoño 2026 CT_27. Not a convention, so it must not be guessed at.
        self.assertEqual(
            clocks.parse_filename('01060117_fiscalizador.JPG'), (None, None)
        )

    def test_stills_and_videos(self):
        self.assertTrue(exports.is_still('01190313.JPG'))
        self.assertFalse(exports.is_still('01190313.MP4'))
        self.assertFalse(exports.is_still('01190313.MOV'))


# =============================================================================
# Felipe's scenarios
# =============================================================================

class TestScenarioA(unittest.TestCase):
    """Clock correct, no split. Install anchor with offset zero. All repairable.

    Camera-off-at-retrieval is irrelevant here: there is no error to propagate.
    """

    def test_clean_clock_needs_no_anchor(self):
        rows = run_of('2025-11-20T10:00:00', 20, 1)
        d, repairs, _ = diagnose_and_plan(
            rows, [anchor('install', '2025-11-20T10:00:00', '2025-11-20T10:00:00')]
        )
        self.assertEqual(len(d.segments), 1)
        self.assertFalse(d.has_clock_failure)
        self.assertEqual([r.reason for r in repairs], ['clock_clean'])
        self.assertTrue(all(r.valid_date and r.valid_time_of_day and r.valid_effort
                            for r in repairs))
        self.assertIsNone(repairs[0].offset)


class TestScenarioB(unittest.TestCase):
    """Clock misconfigured at install, no split. One constant offset repairs all."""

    def test_constant_offset_from_install(self):
        rows = run_of('2017-01-01T00:00:00', 20, 1)
        a = anchor('install', '2025-11-20T10:00:00', '2017-01-01T00:00:00')
        d, repairs, _ = diagnose_and_plan(rows, [a])

        self.assertEqual(len(d.segments), 1)
        self.assertTrue(d.has_clock_failure)          # out of window
        self.assertTrue(d.segments[0].coherent)
        self.assertEqual([r.reason for r in repairs], ['offset_from_install'])
        self.assertEqual(repairs[0].offset, a.offset)
        self.assertTrue(repairs[0].valid_date)
        self.assertTrue(repairs[0].valid_time_of_day)
        self.assertTrue(repairs[0].valid_effort)


class TestScenarioC(unittest.TestCase):
    """1 split, retrieval photo + real datetime. Both segments repairable."""

    def test_install_plus_retrieval_covers_both(self):
        rows = (
            run_of('2025-11-20T10:00:00', 10, 1)
            + run_of('2017-01-01T00:00:00', 30, 11)
        )
        anchors = [
            anchor('install',   '2025-11-20T10:00:00', '2025-11-20T10:00:00'),
            anchor('retrieval', '2026-05-15T12:10:00', '2017-01-08T06:00:00'),
        ]
        d, repairs, _ = diagnose_and_plan(rows, anchors)

        self.assertEqual(len(d.segments), 2)
        self.assertEqual(
            [r.reason for r in repairs],
            ['offset_from_install', 'offset_from_retrieval'],
        )
        self.assertTrue(all(r.valid_date and r.valid_time_of_day for r in repairs))
        self.assertTrue(all(r.valid_effort for r in repairs))


class TestScenarioD(unittest.TestCase):
    """>1 split, install + retrieval. First and last repairable; middles are not.

    The middle segment becomes presence-only and, because its dates are unknown,
    the whole station drops out of the effort denominator.
    """

    def test_middle_segment_has_no_anchor(self):
        rows = (
            run_of('2025-11-20T10:00:00', 10, 1)          # seg 0, in window
            + run_of('2017-01-01T00:00:00', 8, 11)        # seg 1, ends 01-02
            + run_of('2017-01-01T00:00:00', 40, 19)       # seg 2, runs to 01-10
        )
        anchors = [
            anchor('install',   '2025-11-20T10:00:00', '2025-11-20T10:00:00'),
            anchor('retrieval', '2026-05-15T12:10:00', '2017-01-10T18:00:00'),
        ]
        d, repairs, _ = diagnose_and_plan(rows, anchors)

        self.assertEqual(len(d.segments), 3)
        self.assertEqual(
            [r.reason for r in repairs],
            ['offset_from_install', 'no_anchor_in_segment', 'offset_from_retrieval'],
        )
        self.assertEqual([r.valid_date for r in repairs], [True, False, True])
        # Station-level: one unknown segment removes the station from effort entirely.
        self.assertTrue(all(not r.valid_effort for r in repairs))


class TestScenarioE(unittest.TestCase):
    """1 split, camera dead at retrieval. Install only ⇒ first segment only."""

    def test_only_first_segment_repairable(self):
        rows = (
            run_of('2025-11-20T10:00:00', 10, 1)
            + run_of('2017-01-01T00:00:00', 30, 11)
        )
        d, repairs, _ = diagnose_and_plan(
            rows, [anchor('install', '2025-11-20T10:00:00', '2025-11-20T10:00:00')]
        )
        self.assertEqual(
            [r.reason for r in repairs],
            ['offset_from_install', 'no_anchor_in_segment'],
        )
        self.assertEqual([r.valid_date for r in repairs], [True, False])
        self.assertTrue(all(not r.valid_effort for r in repairs))


class TestScenarioF(unittest.TestCase):
    """Zero anchors — the status of the legacy archive. Nothing is repairable."""

    def test_no_anchors_repairs_nothing(self):
        rows = (
            run_of('2025-11-20T10:00:00', 10, 1)
            + run_of('2017-01-01T00:00:00', 30, 11)
        )
        d, repairs, _ = diagnose_and_plan(rows, [])
        self.assertEqual(len(repairs), 2)
        self.assertTrue(all(r.reason == 'no_anchor_in_segment' for r in repairs))
        self.assertTrue(all(not r.valid_date and not r.valid_effort for r in repairs))


class TestScenarioG(unittest.TestCase):
    """>1 split AND dead at retrieval — otoño 2026 CT_18's actual case.

    This is the shape the shipped single-offset repair fabricated datetimes for.
    """

    def test_only_first_segment_survives(self):
        rows = (
            run_of('2025-11-19T06:41:00', 10, 1)          # the 9.4 real days
            + run_of('2017-01-01T00:00:00', 32, 11)
            + run_of('2017-01-01T00:00:00', 40, 43)
            + run_of('2017-01-01T00:00:00', 227, 83)
        )
        d, repairs, _ = diagnose_and_plan(
            rows, [anchor('install', '2025-11-19T06:41:00', '2025-11-19T06:41:00')]
        )
        self.assertEqual(len(d.segments), 4)
        self.assertEqual([r.valid_date for r in repairs], [True, False, False, False])
        self.assertTrue(all(not r.valid_effort for r in repairs))

    def test_last_real_proxy_cannot_rescue_overlapping_segments(self):
        """The bug, stated as a test.

        A single `last_real_proxy` anchored to the last bogus stamp must NOT
        validate the earlier bogus segments. They all restart at 2017-01-01, so
        the anchor's stamp is ambiguous or simply outside them — either way it
        buys exactly one segment, not four.
        """
        rows = (
            run_of('2025-11-19T06:41:00', 10, 1)
            + run_of('2017-01-01T00:00:00', 32, 11)       # ends 2017-01-08
            + run_of('2017-01-01T00:00:00', 40, 43)       # ends 2017-01-10
            + run_of('2017-01-01T00:00:00', 227, 83)      # runs far past both
        )
        proxy = anchor('last_real_proxy', '2026-05-15T12:10:00', '2017-02-25T12:00:00')
        d, repairs, _ = diagnose_and_plan(rows, [proxy])

        self.assertEqual(len(d.segments), 4)
        # Only the segment actually containing the proxy stamp is repaired,
        # and only its DATE — a proxy rotates the time-of-day.
        self.assertEqual(
            [r.reason for r in repairs],
            ['no_anchor_in_segment', 'no_anchor_in_segment',
             'no_anchor_in_segment', 'offset_from_last_real_proxy'],
        )
        self.assertEqual([r.valid_date for r in repairs], [False, False, False, True])
        self.assertTrue(all(not r.valid_time_of_day for r in repairs))
        self.assertTrue(all(not r.valid_effort for r in repairs))


# =============================================================================
# Preconditions
# =============================================================================

class TestPrecondition1Ordering(unittest.TestCase):
    """P1 — capture order must be provable."""

    def test_colliding_counters_with_clock_failure_fail_closed(self):
        rows = (
            run_of('2025-11-20T10:00:00', 10, 1)
            + run_of('2017-01-01T00:00:00', 10, 1)       # counter wrap: 1..10 again
        )
        d, repairs, _ = diagnose_and_plan(
            rows, [anchor('install', '2025-11-20T10:00:00', '2025-11-20T10:00:00')]
        )
        self.assertFalse(d.ordered)
        self.assertEqual(d.order_evidence, clocks.ORDER_COUNTER)
        self.assertTrue(all(r.reason == 'ordering_unrecoverable' for r in repairs))
        self.assertTrue(all(not r.valid_date for r in repairs))

    def test_healthy_camera_survives_unrecoverable_order(self):
        """The otoño 2026 wrap cameras: >999 images, no manifest, sane clock.

        CT_14, CT_20, CT_15, CT_08 and CT_23 can never satisfy P1 because the
        campaign was flattened before the manifest existed. They must stay fully
        valid — failing P1 does not condemn a camera that never reset.
        """
        rows = (
            run_of('2025-11-20T10:00:00', 10, 1)
            + run_of('2025-12-20T10:00:00', 10, 1)      # colliding counters, sane dates
        )
        d, repairs, _ = diagnose_and_plan(rows, [])
        self.assertFalse(d.ordered)
        self.assertEqual([r.reason for r in repairs], ['clock_clean'])
        self.assertTrue(all(r.valid_date and r.valid_time_of_day and r.valid_effort
                            for r in repairs))

    def test_dcim_manifest_restores_order(self):
        """The manifest is what makes a wrapped counter orderable again."""
        rows = (
            run_of('2025-11-20T10:00:00', 10, 1, dcim='100EK113')
            + run_of('2017-01-01T00:00:00', 10, 1, dcim='101EK113')
        )
        d, repairs, _ = diagnose_and_plan(
            rows, [anchor('install', '2025-11-20T10:00:00', '2025-11-20T10:00:00')]
        )
        self.assertTrue(d.ordered)
        self.assertEqual(d.order_evidence, clocks.ORDER_MANIFEST)
        self.assertEqual(len(d.segments), 2)
        self.assertEqual(
            [r.reason for r in repairs],
            ['offset_from_install', 'no_anchor_in_segment'],
        )

    def test_partial_manifest_fails_closed(self):
        """A manifest covering only some frames is worse than no manifest.

        The described frames would sort under their folder and the rest would pool
        together, giving a confident but wrong order. Happens when a deployment was
        partially flattened before the manifest existed.
        """
        rows = (
            run_of('2025-11-20T10:00:00', 5, 1, dcim='100EK113')
            + run_of('2025-11-25T10:00:00', 5, 1)          # no dcim_folder
        )
        d, _, _ = diagnose_and_plan(rows, [])
        self.assertFalse(d.ordered)
        self.assertEqual(d.order_evidence, clocks.ORDER_MANIFEST)
        self.assertTrue(any('partially' in n for n in d.notes))

    def test_unparseable_counter_blocks_ordering(self):
        rows = run_of('2025-11-20T10:00:00', 5, 1)
        rows.append({'file_name': 'IMG_weird.JPG',
                     'camera_datetime': datetime(2017, 1, 1)})
        d, _, _ = diagnose_and_plan(rows, [])
        self.assertFalse(d.ordered)


class TestPrecondition2Coherence(unittest.TestCase):
    """P2 — inside a segment the filename MMDD must agree with its DateTime."""

    def test_register_corruption_is_unrepairable_even_with_an_anchor(self):
        """CT_18's second, independent failure: 166 of 312 frames disagree.

        An offset cannot fix a camera whose month/day registers are corrupt, so
        no number of anchors makes it repairable. This is what distinguishes it
        from otoño 2025 CT15's clean +8 year error, where the filename still
        encodes the true month and day.
        """
        rows = run_of('2025-11-20T10:00:00', 10, 1, mmdd='0008')   # impossible month
        d, repairs, _ = diagnose_and_plan(
            rows, [anchor('install', '2025-11-20T10:00:00', '2025-11-20T10:00:00')]
        )
        self.assertFalse(d.segments[0].coherent)
        self.assertTrue(repairs[0].reason.startswith('segment_incoherent'))
        self.assertFalse(repairs[0].valid_date)
        self.assertFalse(repairs[0].valid_effort)

    def test_year_only_error_stays_coherent(self):
        """A pure year error preserves MM-DD and time-of-day, so it IS repairable.

        otoño 2025 CT15 is exactly this: "+8 yr, filename codifica 09-10".
        """
        rows = run_of('2017-09-10T10:00:00', 10, 1)
        a = anchor('install', '2025-09-10T10:00:00', '2017-09-10T10:00:00')
        d, repairs, _ = diagnose_and_plan(rows, [a])
        self.assertTrue(d.segments[0].coherent)
        self.assertEqual(repairs[0].offset, a.offset)
        self.assertTrue(repairs[0].valid_time_of_day)


# =============================================================================
# Detection cases the old `year < 2024` rule could not see
# =============================================================================

class TestForwardJump(unittest.TestCase):

    def test_forward_jump_out_of_window_is_a_split(self):
        """A clock set forward to 2030 keeps every delta positive.

        Backwards-step detection alone is blind to it, and `year < 2024` cannot
        see it at all. The deployment window is what catches it.
        """
        rows = (
            run_of('2025-11-20T10:00:00', 10, 1)
            + run_of('2030-06-01T10:00:00', 10, 11)
        )
        d, repairs, _ = diagnose_and_plan(
            rows, [anchor('install', '2025-11-20T10:00:00', '2025-11-20T10:00:00')]
        )
        self.assertEqual(len(d.segments), 2)
        self.assertTrue(d.has_clock_failure)
        self.assertIs(d.segments[0].in_window, True)
        self.assertIs(d.segments[1].in_window, False)
        self.assertEqual([r.valid_date for r in repairs], [True, False])


class TestVideosExcluded(unittest.TestCase):

    def test_video_rows_never_create_segments(self):
        """Trap 1: video DateTimes are stamped +1 h from the paired JPG and some
        carry the file-copy date. Including them gave CT_18 61 phantom resets."""
        rows = run_of('2025-11-20T10:00:00', 10, 1)
        rows += [
            frame('2026-06-10T00:00:00', 5, ext='MP4'),
            frame('2026-06-10T00:00:00', 6, ext='MOV'),
        ]
        d, repairs, _ = diagnose_and_plan(rows, [])
        self.assertEqual(d.n_videos_excluded, 2)
        self.assertEqual(d.n_stills, 10)
        self.assertEqual([r.reason for r in repairs], ['clock_clean'])


class TestUnrepairablePending(unittest.TestCase):

    def test_pending_anchor_outranks_a_clean_looking_sequence(self):
        rows = run_of('2025-11-20T10:00:00', 10, 1)
        pending = clocks.Anchor(
            station_id=STATION, anchor_type='unrepairable_pending',
            real_datetime=None, camera_datetime=None,
            source='pending_field_info', notes='awaiting notebook',
        )
        _, repairs, notes = diagnose_and_plan(rows, [pending])
        self.assertTrue(all(r.reason == 'unrepairable_pending_anchor' for r in repairs))
        self.assertTrue(all(not r.valid_date for r in repairs))
        self.assertTrue(any('awaiting' in n for n in notes))


class TestAmbiguousAnchor(unittest.TestCase):

    def test_anchor_inside_two_overlapping_segments_repairs_nothing(self):
        rows = (
            run_of('2017-01-01T00:00:00', 20, 1)         # 2017-01-01 → 01-05
            + run_of('2017-01-01T00:00:00', 20, 21)      # same range again
        )
        a = anchor('retrieval', '2026-05-15T12:10:00', '2017-01-03T00:00:00')
        d, repairs, notes = diagnose_and_plan(rows, [a])

        self.assertEqual(len(d.segments), 2)
        self.assertTrue(all(r.reason == 'no_anchor_in_segment' for r in repairs))
        self.assertTrue(any('ambiguous' in n for n in notes))

    def test_explicit_segment_index_resolves_the_ambiguity(self):
        rows = (
            run_of('2017-01-01T00:00:00', 20, 1)
            + run_of('2017-01-01T00:00:00', 20, 21)
        )
        a = anchor('retrieval', '2026-05-15T12:10:00', '2017-01-03T00:00:00',
                   segment_index=1)
        _, repairs, _ = diagnose_and_plan(rows, [a])
        self.assertEqual(
            [r.reason for r in repairs],
            ['no_anchor_in_segment', 'offset_from_retrieval'],
        )


class TestSegmentForRows(unittest.TestCase):
    """Which segment each row belongs to — including the rows diagnosis excluded.

    This is what a caller needs to apply a per-segment offset. Getting it wrong is
    the whole bug: one offset across four resets.
    """

    def test_stills_keep_the_segment_they_were_split_into(self):
        rows = (
            run_of('2025-11-20T10:00:00', 4, 1)
            + run_of('2017-01-01T00:00:00', 3, 5)
        )
        df = pd.DataFrame(rows)
        d = clocks.diagnose(df, STATION, window=WINDOW)
        seg = clocks.segment_for_rows(d, df['camera_datetime'])
        self.assertEqual(list(seg), [0, 0, 0, 0, 1, 1, 1])

    def test_a_video_on_a_healthy_camera_joins_its_only_segment(self):
        """One segment means the camera never reset, so there is nothing to
        attribute and a video cannot be placed wrongly."""
        rows = run_of('2025-11-20T10:00:00', 3, 1)
        rows.append(frame('2026-05-15T12:00:00', 99, ext='MP4'))   # after every still
        df = pd.DataFrame(rows)
        d = clocks.diagnose(df, STATION, window=WINDOW)
        self.assertEqual(len(d.segments), 1)
        seg = clocks.segment_for_rows(d, df['camera_datetime'])
        self.assertEqual(list(seg), [0, 0, 0, 0])

    def test_a_video_on_a_reset_camera_is_placed_by_containment(self):
        rows = (
            run_of('2025-11-20T10:00:00', 4, 1)
            + run_of('2017-01-01T00:00:00', 4, 5)
        )
        rows.append(frame('2017-01-01T06:00:00', 99, ext='MOV'))
        df = pd.DataFrame(rows)
        d = clocks.diagnose(df, STATION, window=WINDOW)
        seg = clocks.segment_for_rows(d, df['camera_datetime'])
        self.assertEqual(seg.iloc[-1], 1)

    def test_an_unplaceable_row_stays_unassigned_rather_than_guessed(self):
        """Two segments overlapping in camera time cannot claim a video between
        them; NA is the honest answer and the caller must refuse the row."""
        rows = (
            run_of('2017-01-01T00:00:00', 4, 1, step_hours=1)
            + run_of('2017-01-01T00:00:00', 4, 5, step_hours=1)
        )
        rows.append(frame('2020-06-01T00:00:00', 99, ext='MP4'))
        df = pd.DataFrame(rows)
        d = clocks.diagnose(df, STATION, window=WINDOW)
        seg = clocks.segment_for_rows(d, df['camera_datetime'])
        self.assertTrue(pd.isna(seg.iloc[-1]))


class TestUnaccountedDaysIsDiagnosticOnly(unittest.TestCase):

    def test_unaccounted_days_reported_but_does_not_change_verdicts(self):
        """§5.6 — the rejected "slack" heuristic. Reported, never a criterion."""
        rows = (
            run_of('2025-11-20T10:00:00', 10, 1)
            + run_of('2017-01-01T00:00:00', 30, 11)
        )
        anchors = [
            anchor('install',   '2025-11-20T10:00:00', '2025-11-20T10:00:00'),
            anchor('retrieval', '2026-05-15T12:10:00', '2017-01-08T06:00:00'),
        ]
        d, repairs, _ = diagnose_and_plan(rows, anchors)
        self.assertIsNotNone(d.unaccounted_days)
        self.assertGreater(d.unaccounted_days, 100)      # most of the window is a gap
        # ...and yet both segments are repairable, because each holds an anchor.
        self.assertTrue(all(r.valid_date for r in repairs))


class TestDcimFolderKey(unittest.TestCase):
    """Condition 1 of the manifest's ordering claim: the group must be a folder the
    CAMERA created. A folder a person made says nothing about capture order.

    Otoño 2025 CT04 is the case that forced this: 723 loose frames under `M5`, beside
    `M5/100EK113` and `M5/101EK113`. Recording the whole path made `M5` sort FIRST,
    asserting its January frames preceded the October ones — a backwards step in
    capture order, which reads as a clock reset on 2,097 frames.
    """

    def test_camera_folders_are_kept(self):
        for raw in ('100EK113', 'M5/100EK113', 'M 11/101EK113',
                    'M17 (TC20)/102EK113', '100CANON'):
            with self.subTest(raw=raw):
                self.assertRegex(clocks.dcim_folder_key(raw), r'^\d{3}[A-Za-z0-9]{3,}$')

    def test_only_the_last_component_is_used(self):
        self.assertEqual(clocks.dcim_folder_key('M5/100EK113'), '100EK113')

    def test_windows_separators_are_handled(self):
        """The manifest is written on Windows; original_relpath uses backslashes."""
        self.assertEqual(clocks.dcim_folder_key('M5\\100EK113'), '100EK113')

    def test_hand_made_folders_yield_no_key(self):
        """Every one of these was a real otoño 2025 folder name."""
        for raw in ('M5', 'M 6', 'M 11', 'M17', 'M17 (TC20)',
                    'M18 (vacía, TC mala)', '', None):
            with self.subTest(raw=raw):
                self.assertEqual(clocks.dcim_folder_key(raw), '')

    def test_loose_group_makes_the_deployment_refuse_to_order(self):
        """The two conditions composing: a hand-made group gets no key (condition 1),
        which makes the deployment partially described (condition 2), which
        establish_order already refuses. CT04 ends up unordered rather than wrong."""
        rows = run_of('2024-10-11T10:00:00', 4, 1, dcim='100EK113')
        rows += run_of('2025-01-09T10:00:00', 4, 1,
                       dcim=clocks.dcim_folder_key('M5'))     # the loose group
        df = pd.DataFrame(rows)
        df['_mmdd'] = [f'{d:%m%d}' for d in df.camera_datetime]
        df['_counter'] = [1, 2, 3, 4, 1, 2, 3, 4]
        _, ordered, evidence, notes = clocks.establish_order(df)
        self.assertFalse(ordered)
        self.assertEqual(evidence, clocks.ORDER_MANIFEST)
        self.assertTrue(any('only partially' in n for n in notes), notes)

    def test_all_camera_folders_still_order(self):
        """CT14 and CT20's shape must keep full manifest ordering."""
        rows = run_of('2025-01-16T10:00:00', 4, 1, dcim='100EK113')
        rows += run_of('2025-01-16T20:00:00', 4, 1, dcim='101EK113')
        df = pd.DataFrame(rows)
        df['_mmdd'] = [f'{d:%m%d}' for d in df.camera_datetime]
        df['_counter'] = [1, 2, 3, 4, 1, 2, 3, 4]
        ordered_df, ordered, evidence, _ = clocks.establish_order(df)
        self.assertTrue(ordered)
        self.assertEqual(evidence, clocks.ORDER_MANIFEST)
        self.assertEqual(list(ordered_df.dcim_folder)[:4], ['100EK113'] * 4)


if __name__ == '__main__':
    unittest.main()
