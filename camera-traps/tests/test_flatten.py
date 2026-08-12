"""Fixtures for setup/flatten_for_camtrapdp.py — the naming decisions.

Run:  python3 -m unittest discover -s tests -v

stdlib unittest on purpose: these must run on the office Windows box and on the
Linux laptop without either growing a test dependency.

Only the destination-naming rule is covered here. The moving itself is I/O and is
exercised by the conservation check the script runs on every deployment; what needs
locking down is which NAME a colliding frame ends up with, because that name reaches
`file_name` in every export and every downstream join.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'setup'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flatten_for_camtrapdp import prefix_candidates, resolve_dest


class TestPrefixCandidates(unittest.TestCase):
    """A prefix is the only route by which a FOLDER name reaches a FILE name."""

    def test_dcim_folder_alone_comes_first(self):
        """Otoño 2025 arrived with a grid folder between station and DCIM folder.
        Joining the whole path would have put a space into 28 CT14 filenames to
        disambiguate nothing — the grid folder is constant within the deployment."""
        self.assertEqual(prefix_candidates(['M 11', '101EK113'])[0], '101EK113')

    def test_full_path_is_kept_as_a_fallback(self):
        got = prefix_candidates(['M 11', '101EK113'])
        self.assertEqual(len(got), 2)
        self.assertEqual(got[1], 'M_11_101EK113')

    def test_single_level_yields_one_candidate(self):
        """Otoño 2026's shape: no grid folder, so both candidates coincide."""
        self.assertEqual(prefix_candidates(['102EK113']), ['102EK113'])

    def test_unsafe_characters_never_reach_a_filename(self):
        """Grid folders are typed by hand in the field. `M17 (TC20)` and `M 6` were
        both real. A space or bracket in a filename survives every later join."""
        for parts, expected in (
            (['M17 (TC20)'], 'M17_TC20'),
            (['M 6'], 'M_6'),
            (['M18 (vacía, TC mala)'], 'M18_vac_a_TC_mala'),
        ):
            with self.subTest(parts=parts):
                got = prefix_candidates(parts)[0]
                self.assertEqual(got, expected)
                self.assertRegex(got, r'^[A-Za-z0-9_]+$')


class TestResolveDest(unittest.TestCase):

    def setUp(self):
        self.dep = Path(tempfile.mkdtemp()) / 'CT14'
        self.dep.mkdir()

    def test_free_name_is_moved_not_renamed(self):
        dest, action = resolve_dest(self.dep, ['M11', '100EK113'], '01160002.JPG')
        self.assertEqual(action, 'moved')
        self.assertEqual(dest.name, '01160002.JPG')

    def test_collision_takes_the_dcim_prefix(self):
        (self.dep / '01160002.JPG').touch()
        dest, action = resolve_dest(self.dep, ['M11', '101EK113'], '01160002.JPG')
        self.assertEqual(action, 'renamed')
        self.assertEqual(dest.name, '101EK113_01160002.JPG')

    def test_claimed_makes_dry_run_predict_renames(self):
        """Without `claimed` a --dry-run sees an untouched disk and reports zero
        renames for a deployment that would in fact rename dozens."""
        claimed = {self.dep / '01160002.JPG'}
        dest, action = resolve_dest(
            self.dep, ['M11', '101EK113'], '01160002.JPG', claimed)
        self.assertEqual(action, 'renamed')

    def test_counter_only_after_every_prefix_is_taken(self):
        (self.dep / '01160002.JPG').touch()
        (self.dep / '101EK113_01160002.JPG').touch()
        (self.dep / 'M11_101EK113_01160002.JPG').touch()
        dest, action = resolve_dest(self.dep, ['M11', '101EK113'], '01160002.JPG')
        self.assertEqual(action, 'renamed')
        self.assertEqual(dest.name, 'M11_101EK113_01160002_2.JPG')

    def test_nothing_is_ever_skipped(self):
        """An earlier version dropped same-name/same-size files as duplicates — which
        is exactly the signature of a reset camera re-emitting `0101xxxx` names. A
        duplicated image is a nuisance; a discarded one is unrecoverable."""
        (self.dep / '01010001.JPG').touch()
        dest, action = resolve_dest(self.dep, ['101EK113'], '01010001.JPG')
        self.assertIsNotNone(dest)
        self.assertNotEqual(dest, self.dep / '01010001.JPG')


if __name__ == '__main__':
    unittest.main()
