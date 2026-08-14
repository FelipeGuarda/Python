"""Fixtures for setup/flatten_for_camtrapdp.py — the naming decisions.

Run:  python3 -m unittest discover -s tests -v

stdlib unittest on purpose: these must run on the office Windows box and on the
Linux laptop without either growing a test dependency.

Only the destination-naming rule is covered here. The moving itself is I/O and is
exercised by the conservation check the script runs on every deployment; what needs
locking down is which NAME a colliding frame ends up with, because that name reaches
`file_name` in every export and every downstream join.
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'setup'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camtrap import stations
from flatten_for_camtrapdp import (
    find_nested_stations,
    prefix_candidates,
    resolve_dest,
)

_ALIAS_CSV = (
    Path(__file__).resolve().parents[1] / 'data' / 'campaigns' / 'station_aliases.csv'
)


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


class TestNamesAStation(unittest.TestCase):
    """The recogniser behind the nested-station refusal.

    Recognition is by SHAPE, so what it must not do is drift away from the spellings
    the campaigns actually used — hence the alias file itself is the fixture.
    """

    # The one alias that must NOT be recognised: an unrenamed SD-card folder that
    # became a deployment in primavera_2025. Every deployment contains DCIM folders,
    # so recognising it would refuse every flatten there has ever been.
    DCIM_ALIAS = '100EK113'

    def test_every_alias_spelling_is_recognised_except_the_dcim_one(self):
        """The 2026-08-13 hand-check — 34 TC-style rows, 0 disagreements — as a
        fixture that re-runs whenever a row is added."""
        with open(_ALIAS_CSV, encoding='utf-8', newline='') as f:
            spellings = {row['station_raw'].strip() for row in csv.DictReader(f)}
        self.assertIn(self.DCIM_ALIAS, spellings, 'fixture premise changed')

        for spelling in sorted(spellings):
            with self.subTest(spelling=spelling):
                self.assertEqual(
                    stations.names_a_station(spelling),
                    spelling != self.DCIM_ALIAS,
                )

    def test_canonical_is_recognised(self):
        self.assertTrue(stations.names_a_station('CT23'))

    def test_dcim_and_grid_folders_are_not_stations(self):
        """A false positive is now fatal, so the folders every deployment actually
        contains must be rejected. Grid names are typed by hand in the field."""
        for name in (
            '100EK113', '101EK113', '102EK113',
            'M5', 'M 11', 'M17 (TC20)', 'M18 (vacía, TC mala)', 'Backups',
        ):
            with self.subTest(name=name):
                self.assertFalse(stations.names_a_station(name))


class TestFindNestedStations(unittest.TestCase):
    """Attribution — the question conservation and ordering do not ask."""

    def test_the_tc23_case(self):
        """Primavera 2025: a whole camera inside CT22. Different filename schemes,
        so zero collisions — `moved=2460 renamed=0 lost=0` and every check passed."""
        files = [(Path(f'/x/IMAG{i:04d}.JPG'), ['TC23_M20.2']) for i in range(2460)]
        self.assertEqual(find_nested_stations(files), {'TC23_M20.2': 2460})

    def test_only_the_shallowest_component_is_reported(self):
        """The operator moves one folder; naming its DCIM children too would bury
        the instruction."""
        files = [(Path('/x/a.JPG'), ['M19', 'TC23_M20.2', '100EK113'])]
        self.assertEqual(find_nested_stations(files), {'M19/TC23_M20.2': 1})

    def test_a_clean_deployment_reports_nothing(self):
        files = [
            (Path('/x/01160002.JPG'), ['M 11', '101EK113']),
            (Path('/x/01160003.JPG'), ['M 11', '101EK113']),
        ]
        self.assertEqual(find_nested_stations(files), {})

    def test_files_are_counted_per_offending_folder(self):
        files = (
            [(Path('/x/a.JPG'), ['CT_23'])] * 3
            + [(Path('/x/b.JPG'), ['TC24_M21.2'])] * 2
            + [(Path('/x/c.JPG'), ['100EK113'])] * 9
        )
        self.assertEqual(find_nested_stations(files), {'CT_23': 3, 'TC24_M21.2': 2})

    def test_an_empty_station_folder_is_not_an_offence(self):
        """collect_subdir_files yields no rows for it, and a folder holding no media
        misattributes nothing."""
        self.assertEqual(find_nested_stations([]), {})


if __name__ == '__main__':
    unittest.main()
