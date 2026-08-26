"""Fixtures for setup/flatten_for_camtrapdp.py — the naming decisions.

Run:  python3 -m unittest discover -s tests -v

stdlib unittest on purpose: these must run on the office Windows box and on the
Linux laptop without either growing a test dependency.

The destination-naming rule is covered first: which NAME a colliding frame ends up with,
because that name reaches `file_name` in every export and every downstream join.

Since 2026-08-26 the MANIFEST is covered too, on real temp trees. Both recoveries of
2026-08-18 rested on it and neither had a fixture (V2-REVIEW 1.10 / B8):

  * otoño 2026's `dcim_manifest.csv` was rebuilt from a flatten log, recovering capture
    order for CT14 / CT20 / CT23 — 3,561 frames. What a rebuild has to reproduce is
    exactly what `process_deployment` emits, so that is what is pinned here: one row per
    file, the camera-folder rule, and agreement with `clocks.DCIM_MANIFEST_COLUMNS`.
  * 103 GB of restored Synology copies were deleted only after proving every file had a
    flattened counterpart matched by size — 10,808 files, 0 unaccounted. The evidence was
    the manifest's `size_bytes`, so what is pinned is that the manifest is a usable
    deletion ledger: every row resolves to a real file of the recorded size, and a
    mismatch is detectable rather than silent.
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'setup'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camtrap import stations
from camtrap.clocks import DCIM_MANIFEST_COLUMNS
from flatten_for_camtrapdp import (
    collect_subdir_files,
    find_nested_stations,
    prefix_candidates,
    process_deployment,
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

# ── the manifest: what a rebuild has to reproduce, and the deletion ledger ────────

def _tree(spec: dict, already_flat=()) -> Path:
    """A deployment directory. `spec` maps folder name -> {filename: contents}."""
    root = Path(tempfile.mkdtemp()) / 'CT14_M11.2'
    root.mkdir(parents=True)
    for name in already_flat:
        (root / name).write_text('at deployment level')
    for folder, files in spec.items():
        (root / folder).mkdir(parents=True)
        for name, contents in files.items():
            (root / folder / name).write_text(contents)
    return root


def _flatten(root: Path) -> list[dict]:
    rows: list[dict] = []
    summary = process_deployment(root, collect_subdir_files(root), False, rows)
    assert summary['lost'] == 0, summary
    return rows


class TestTheManifestDescribesEveryFile(unittest.TestCase):
    """`clocks.establish_order` refuses a partially described deployment, so a manifest
    that silently omits a file does not fail — it downgrades the evidence tier. That is
    why completeness is asserted here rather than left to the conservation count."""

    def test_one_row_per_file_including_the_already_flat_ones(self):
        root = _tree({'100EK113': {'01230001.JPG': 'a', '01230002.JPG': 'b'},
                      '101EK113': {'01240001.JPG': 'c'}},
                     already_flat=('01220099.JPG',))
        rows = _flatten(root)
        self.assertEqual(len(rows), 4)
        self.assertEqual({r['flat_name'] for r in rows},
                         {'01230001.JPG', '01230002.JPG', '01240001.JPG', '01220099.JPG'})
        flat = next(r for r in rows if r['flat_name'] == '01220099.JPG')
        self.assertEqual(flat['action'], 'already_flat')
        self.assertEqual(flat['dcim_folder'], '',
                         'a file with no camera folder must say so, not be omitted')

    def test_the_schema_is_the_one_clocks_reads(self):
        """A rebuild is written against `DCIM_MANIFEST_COLUMNS`. If the emitter drifts
        from it, the rebuilt manifest and the live one stop being the same document."""
        rows = _flatten(_tree({'100EK113': {'01230001.JPG': 'a'}}))
        self.assertEqual(list(rows[0]), DCIM_MANIFEST_COLUMNS)

    def test_only_a_camera_created_folder_is_recorded_as_one(self):
        """`M5`, `M 11`, `M17 (TC20)` were all real in otoño 2025. A folder a person made
        says nothing about capture order, and recording it would let clocks.py sort on it.
        The path is not lost — it is one column across, in `original_relpath`."""
        rows = _flatten(_tree({'M5': {'01230001.JPG': 'a'},
                               '100EK113': {'01240001.JPG': 'b'}}))
        by_name = {r['flat_name']: r for r in rows}
        self.assertEqual(by_name['01230001.JPG']['dcim_folder'], '')
        self.assertEqual(by_name['01230001.JPG']['original_relpath'], 'M5/01230001.JPG')
        self.assertEqual(by_name['01240001.JPG']['dcim_folder'], '100EK113')

    def test_a_wrapped_counter_keeps_both_frames_and_both_folders(self):
        """The 999-wrap: the same filename in two camera folders. This is precisely what
        the old duplicate-skip discarded, and what the manifest exists to survive."""
        rows = _flatten(_tree({'100EK113': {'01230001.JPG': 'first'},
                               '101EK113': {'01230001.JPG': 'second'}}))
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({r['flat_name'] for r in rows}), 2)
        self.assertEqual({r['dcim_folder'] for r in rows}, {'100EK113', '101EK113'})
        self.assertEqual(sorted(r['original_name'] for r in rows),
                         ['01230001.JPG', '01230001.JPG'])


class TestTheManifestIsADeletionLedger(unittest.TestCase):
    """The 2026-08-18 deletion was gated on 10,808 files with 0 unaccounted, matched by
    size. Nothing held that method afterwards."""

    @staticmethod
    def _unaccounted(root: Path, rows: list[dict]) -> list[str]:
        """Rows whose file is absent or whose size disagrees. Empty = safe to delete."""
        missing = []
        for r in rows:
            path = root / r['flat_name']
            if not path.exists() or path.stat().st_size != int(r['size_bytes']):
                missing.append(r['flat_name'])
        return missing

    def test_every_row_resolves_to_a_file_of_the_recorded_size(self):
        root = _tree({'100EK113': {'01230001.JPG': 'a' * 11,
                                   '01230002.JPG': 'b' * 2200},
                      '101EK113': {'01230001.JPG': 'c' * 33}},
                     already_flat=('01220099.JPG',))
        rows = _flatten(root)
        self.assertEqual(self._unaccounted(root, rows), [],
                         'nothing may be deleted while a file is unaccounted for')

    def test_a_size_mismatch_is_detected_rather_than_passing_quietly(self):
        """The failure this guards is a truncated or half-copied file that still exists
        under the right name — the one case a presence check reads as success."""
        root = _tree({'100EK113': {'01230001.JPG': 'a' * 500}})
        rows = _flatten(root)
        (root / '01230001.JPG').write_text('a' * 499)
        self.assertEqual(self._unaccounted(root, rows), ['01230001.JPG'])

    def test_a_missing_file_is_detected(self):
        root = _tree({'100EK113': {'01230001.JPG': 'a' * 500}})
        rows = _flatten(root)
        (root / '01230001.JPG').unlink()
        self.assertEqual(self._unaccounted(root, rows), ['01230001.JPG'])



if __name__ == '__main__':
    unittest.main()
