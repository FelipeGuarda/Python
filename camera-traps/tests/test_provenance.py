"""Fixtures for camtrap/provenance.py — one deployment, one capture story.

Run:  python3 -m unittest discover -s tests -v

stdlib unittest on purpose, like its siblings: these must run on the office Windows
box and on the Linux laptop without either growing a test dependency.

The rule this file locks down is the GENERAL one. `names_a_station` recognises the
three station spellings we have used; this recognises a second camera from its frames
and enumerates nothing, so the cases below are chosen to prove it never consults a
list — an invented naming convention must fire just as a historical one does.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camtrap.provenance import (
    Population,
    multiple_capture_stories,
    populations,
    shape_of,
)


def _run(prefix: str, lo: int, hi: int, ext: str = '.JPG') -> list:
    return [f'{prefix}{i:04d}{ext}' for i in range(lo, hi + 1)]


class TestShapeOf(unittest.TestCase):

    def test_digit_runs_collapse(self):
        self.assertEqual(shape_of('IMAG0001.JPG'), 'IMAG#')
        self.assertEqual(shape_of('01120001.JPG'), '#')

    def test_extension_is_excluded(self):
        """These cameras fire three stills and a video. `01120001.JPG` and
        `01120004.AVI` are one camera telling one story, and a shape that included
        the extension would split every deployment that holds video."""
        self.assertEqual(shape_of('01120001.JPG'), shape_of('01120004.AVI'))

    def test_a_counter_wrapping_does_not_change_the_shape(self):
        """0999 -> 1000 must not look like a new camera."""
        self.assertEqual(shape_of('IMAG0999.JPG'), shape_of('IMAG1000.JPG'))


class TestMultipleCaptureStories(unittest.TestCase):

    def test_the_tc23_case(self):
        """Primavera 2025: 2,460 `IMAG####` frames pooled with CT22's `MMDDnnnn`.

        The pipeline already saw these — `establish_order` reports them as
        unparseable filenames and returns ordered=False. But that routes the evidence
        to the ORDERING question, and failing to order does not condemn a camera, so
        the frames kept camera 22's identity. This asks the identity question.
        """
        found = multiple_capture_stories(_run('0512', 1, 50) + _run('IMAG', 1, 2460))
        self.assertEqual({(p.shape, p.n) for p in found}, {('#', 50), ('IMAG#', 2460)})

    def test_a_naming_convention_we_have_never_seen_fires_too(self):
        """The whole point of the general rule. `names_a_station` would miss a folder
        called `Camara 23`; this does not care what the folder is called."""
        found = multiple_capture_stories(_run('0512', 1, 40) + _run('WSCF', 1, 900))
        self.assertEqual({p.shape for p in found}, {'#', 'WSCF#'})

    def test_a_clean_deployment_is_one_story(self):
        self.assertEqual(multiple_capture_stories(_run('0512', 1, 500)), [])

    def test_stills_and_videos_are_one_story(self):
        """The 3-stills-plus-1-video trigger pattern, which every station shows."""
        names = _run('0512', 1, 300) + _run('0512', 301, 400, ext='.AVI')
        self.assertEqual(multiple_capture_stories(names), [])

    def test_a_hand_renamed_one_off_is_not_a_second_camera(self):
        """Otoño 2026 CT_27 holds `01060117_fiscalizador.JPG`. One frame is not a
        sequence, and a rule that fired on it would be unusable."""
        names = _run('0106', 1, 200) + ['01060117_fiscalizador.JPG']
        self.assertEqual(multiple_capture_stories(names), [])

    def test_our_own_rename_prefix_is_folded_back(self):
        """MEASURED, not supposed: without the fold, pv 2025-2026 CT14's 13
        `101EK113_`-prefixed names formed their own run and read as a second camera —
        the only false positive across all four campaigns. `resolve_dest` writes that
        prefix, so the frames are ours and the story is one."""
        names = _run('0605', 1, 51) + [f'101EK113_0605{i:04d}.JPG' for i in range(1, 14)]
        self.assertEqual(multiple_capture_stories(names), [])
        self.assertEqual([p.n for p in populations(names)], [64])

    def test_the_longest_base_wins_when_prefixes_nest(self):
        """`a_b_#` folds onto `b_#`, not straight onto `#` — otherwise a genuine
        second camera could be absorbed by an unrelated shorter shape."""
        names = _run('0605', 1, 20) + [f'X_0605{i:04d}.JPG' for i in range(1, 20)] \
            + [f'Y_X_0605{i:04d}.JPG' for i in range(1, 20)]
        self.assertEqual(multiple_capture_stories(names), [])

    def test_repeated_names_are_not_a_run(self):
        """A repeated filename is what a reset camera emits — `resolve_dest` handles
        it and it says nothing about which camera took the frame."""
        names = _run('0512', 1, 200) + ['ALT0001.JPG'] * 5
        self.assertEqual(multiple_capture_stories(names), [])

    def test_an_impossible_clock_is_not_a_provenance_problem(self):
        """Primavera CT16 emits month 00 and month 16 — `00300001.JPG`,
        `16300071.JPG`. Same shape, one camera, a broken RTC. That belongs to
        camtrap/clocks.py, and this module must stay out of it."""
        names = _run('0030', 1, 100) + _run('1630', 1, 71)
        self.assertEqual(multiple_capture_stories(names), [])

    def test_empty_input(self):
        self.assertEqual(multiple_capture_stories([]), [])


class TestPopulationIsRun(unittest.TestCase):

    def test_two_frames_at_different_indices_are_a_run(self):
        self.assertTrue(Population('#', 2, 1, 2).is_run)

    def test_one_frame_is_not(self):
        self.assertFalse(Population('#', 1, 7, 7).is_run)

    def test_several_frames_at_one_index_are_not(self):
        self.assertFalse(Population('#', 5, 3, 3).is_run)


if __name__ == '__main__':
    unittest.main()
