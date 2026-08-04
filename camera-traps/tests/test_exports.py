"""Fixtures for camtrap.exports — the full-category export gate.

Run:  python3 -m unittest discover -s tests -v

The gate's whole job is to refuse an export that cannot answer the clock question,
so every fixture here is a shape a real Timelapse2 export has taken or could take.
The one that matters most is `{animal, unclassified}`: that is otoño 2026's actual
file, it LOOKS category-labelled, and letting it through is how CT_18's four resets
became one.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camtrap import exports


def export(types: list[str]) -> pd.DataFrame:
    """A minimal export frame: one row per observationType given."""
    return pd.DataFrame({
        'Deployments': ['CT01'] * len(types),
        'File': [f'0101{i:04d}.JPG' for i in range(len(types))],
        exports.OBSERVATION_TYPE_COLUMN: types,
    })


OVERRIDE_TEXT = (
    'verified_by: Felipe Guarda\n'
    'date: 2026-08-03\n'
    'reason: swept every image in Timelapse2; this card holds no person frame\n'
)


class TestTheRule(unittest.TestCase):

    def test_person_present_passes(self):
        audit = exports.audit_categories(
            export(['empty', 'animal', 'person'])[exports.OBSERVATION_TYPE_COLUMN]
        )
        self.assertEqual(audit.verdict, exports.PASS)
        self.assertTrue(audit.passed)
        self.assertTrue(audit.usable)

    def test_vehicle_alone_also_proves_a_sweep(self):
        audit = exports.audit_categories(
            export(['empty', 'animal', 'vehicle'])[exports.OBSERVATION_TYPE_COLUMN]
        )
        self.assertEqual(audit.verdict, exports.PASS)

    def test_animal_and_unclassified_is_the_otono_2026_file(self):
        """The shape that must never pass: nothing was ever assigned."""
        audit = exports.audit_categories(
            export(['animal'] * 3 + ['unclassified'] * 10)[exports.OBSERVATION_TYPE_COLUMN]
        )
        self.assertEqual(audit.verdict, exports.NEVER_ASSIGNED)
        self.assertFalse(audit.usable)

    def test_animal_only_export_is_also_never_assigned(self):
        audit = exports.audit_categories(
            export(['animal'] * 5)[exports.OBSERVATION_TYPE_COLUMN]
        )
        self.assertEqual(audit.verdict, exports.NEVER_ASSIGNED)

    def test_swept_but_person_free_is_the_overridable_case(self):
        """`empty` present means the sweep happened; no person is the exception."""
        audit = exports.audit_categories(
            export(['empty'] * 8 + ['animal'])[exports.OBSERVATION_TYPE_COLUMN]
        )
        self.assertEqual(audit.verdict, exports.NO_PROOF_OF_SWEEP)
        self.assertIn(audit.verdict, exports.OVERRIDABLE_VERDICTS)

    def test_no_rows(self):
        audit = exports.audit_categories(pd.Series([], dtype=str))
        self.assertEqual(audit.verdict, exports.NO_ROWS)

    def test_blank_counts_as_unassigned_not_as_empty(self):
        audit = exports.audit_categories(
            export(['', '', 'animal'])[exports.OBSERVATION_TYPE_COLUMN]
        )
        self.assertEqual(audit.verdict, exports.NEVER_ASSIGNED)

    def test_case_and_whitespace_are_normalised(self):
        audit = exports.audit_categories(
            export([' Person ', 'ANIMAL', 'empty'])[exports.OBSERVATION_TYPE_COLUMN]
        )
        self.assertEqual(audit.verdict, exports.PASS)

    def test_unrecognised_category_is_noted_not_counted_as_proof(self):
        audit = exports.audit_categories(
            export(['empty', 'animal', 'dog'])[exports.OBSERVATION_TYPE_COLUMN]
        )
        self.assertEqual(audit.verdict, exports.NO_PROOF_OF_SWEEP)
        self.assertTrue(any('dog' in n for n in audit.notes))


class TestOverride(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, text: str) -> Path:
        path = self.dir / exports.OVERRIDE_FILENAME
        path.write_text(text, encoding='utf-8')
        return path

    def test_override_admits_a_person_free_sweep(self):
        self._write(OVERRIDE_TEXT)
        audit = exports.require_full_category(
            export(['empty'] * 4 + ['animal']),
            source='test', override_dir=self.dir,
        )
        self.assertFalse(audit.passed)          # the rule still says no
        self.assertTrue(audit.usable)           # a human signed for it
        self.assertEqual(audit.override.verified_by, 'Felipe Guarda')

    def test_override_cannot_rescue_an_unswept_export(self):
        """No signature turns unassigned rows into a sweep."""
        self._write(OVERRIDE_TEXT)
        with self.assertRaises(exports.ExportGateError) as cm:
            exports.require_full_category(
                export(['animal'] * 3 + ['unclassified'] * 9),
                source='test', override_dir=self.dir,
            )
        self.assertIn('not an overridable verdict', str(cm.exception))

    def test_missing_reason_is_refused(self):
        self._write('verified_by: Felipe Guarda\ndate: 2026-08-03\n')
        with self.assertRaises(exports.ExportGateError) as cm:
            exports.require_full_category(
                export(['empty', 'animal']), source='test', override_dir=self.dir,
            )
        self.assertIn('reason', str(cm.exception))

    def test_indented_continuation_joins_the_previous_value(self):
        self._write(
            'verified_by: Felipe Guarda\n'
            'date: 2026-08-03\n'
            'reason: swept all 12068 images;\n'
            '    the technician never triggered this camera\n'
        )
        audit = exports.require_full_category(
            export(['empty', 'animal']), source='test', override_dir=self.dir,
        )
        self.assertIn('never triggered', audit.override.reason)

    def test_no_override_file_means_rejection(self):
        with self.assertRaises(exports.ExportGateError):
            exports.require_full_category(
                export(['empty', 'animal']), source='test', override_dir=self.dir,
            )

    def test_missing_observation_type_column_is_not_an_export(self):
        with self.assertRaises(exports.ExportGateError) as cm:
            exports.require_full_category(
                pd.DataFrame({'File': ['a.JPG']}),
                source='test', override_dir=self.dir,
            )
        self.assertIn('not a Timelapse2', str(cm.exception))


class TestReadTotalExport(unittest.TestCase):

    def test_missing_file_names_the_animal_export_as_no_substitute(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(exports.ExportGateError) as cm:
                exports.read_total_export(Path(tmp))
        self.assertIn(exports.ANIMAL_EXPORT_FILENAME, str(cm.exception))


if __name__ == '__main__':
    unittest.main()
