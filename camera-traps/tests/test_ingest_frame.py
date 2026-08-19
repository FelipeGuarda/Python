"""Fixtures for compose_ingest_frame / resolve_observation — which rows the table has.

Run:  python3 -m unittest discover -s tests -v

Before 2026-08-19 the canonical table's row set was the reviewed CSV, so a station that
recorded no animal was absent from it entirely — indistinguishable from a station that
was never deployed. Seven station-campaigns were missing that way. These fixtures pin the
row set to the export and pin where each row's verdict comes from.
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camtrap import exports
from camtrap.observations import (
    RESOLUTION_SPECIES_NAMED,
    RESOLUTION_SWEEP_ONLY,
    RESOLUTION_TYPE_FROM_COMMENT,
    REVIEWED_FLAG,
    ReviewedRowNotInExport,
    UnmappedReviewComment,
    compose_ingest_frame,
    resolve_observation,
)

CAMPAIGN = 'otono_2026'


def total(rows):
    """All-images export rows: (station, file, observationType)."""
    return pd.DataFrame({
        'Deployments': [r[0] for r in rows],
        'RelativePath': [r[0] for r in rows],
        'File': [r[1] for r in rows],
        'filePath': [''] * len(rows),
        exports.OBSERVATION_TYPE_COLUMN: [r[2] for r in rows],
        # The export carries these NAMES too, always empty. Their presence is the trap
        # compose_ingest_frame has to avoid.
        'scientificName': [''] * len(rows),
        'observationComments': [''] * len(rows),
    })


def review(rows):
    """Reviewed rows: (station, file, scientificName, observationComments)."""
    return pd.DataFrame({
        'Deployments': [r[0] for r in rows],
        'File': [r[1] for r in rows],
        'scientificName': [r[2] for r in rows],
        'observationComments': [r[3] for r in rows],
        'reviewOutcome': ['corrected'] * len(rows),
    })


class RowSetIsTheExport(unittest.TestCase):

    def test_every_export_row_survives_even_with_no_review(self):
        t = total([('CT01', 'a.JPG', 'blank'), ('CT01', 'b.JPG', 'blank')])
        out = compose_ingest_frame(t, review([]), CAMPAIGN)
        self.assertEqual(len(out), 2)
        self.assertFalse(out[REVIEWED_FLAG].any())

    def test_a_station_with_no_animal_still_appears(self):
        """CT01/CT06/CT17/CT22 in primavera — 6, 21, 7 and 18 frames, no animal."""
        t = total([('CT06', 'a.JPG', 'blank'), ('CT07', 'b.JPG', 'animal')])
        r = review([('CT07', 'b.JPG', 'Puma concolor', 'Puma')])
        out = compose_ingest_frame(t, r, CAMPAIGN)
        self.assertEqual(sorted(out['Deployments']), ['CT06', 'CT07'])

    def test_review_columns_are_not_shadowed_by_the_export(self):
        """The export's own empty scientificName must not win the merge."""
        t = total([('CT01', 'a.JPG', 'animal')])
        r = review([('CT01', 'a.JPG', 'Puma concolor', 'Puma')])
        out = compose_ingest_frame(t, r, CAMPAIGN)
        self.assertEqual(out['scientificName'][0], 'Puma concolor')
        self.assertNotIn('scientificName_rev', out.columns)
        self.assertTrue(out[REVIEWED_FLAG][0])

    def test_a_reviewed_row_absent_from_the_export_is_refused(self):
        """The guarantee the export gate exists for; it moved here from repair_campaign."""
        t = total([('CT01', 'a.JPG', 'blank')])
        r = review([('CT01', 'ghost.JPG', 'Puma concolor', 'Puma')])
        with self.assertRaises(ReviewedRowNotInExport) as ctx:
            compose_ingest_frame(t, r, CAMPAIGN)
        self.assertIn('ghost.JPG', str(ctx.exception))

    def test_station_spelling_does_not_break_the_join(self):
        """The export writes CT_01 and the review CT01 in otoño 2026."""
        t = total([('CT_01', 'a.JPG', 'animal')])
        r = review([('CT01', 'a.JPG', 'Puma concolor', 'Puma')])
        out = compose_ingest_frame(t, r, CAMPAIGN)
        self.assertTrue(out[REVIEWED_FLAG].all())
        self.assertEqual(out['scientificName'][0], 'Puma concolor')


class WhereTheVerdictComesFrom(unittest.TestCase):

    def test_unreviewed_row_takes_the_sweeps_type_and_no_species(self):
        t = total([('CT01', 'a.JPG', 'blank'), ('CT01', 'b.JPG', 'human')])
        out = resolve_observation(compose_ingest_frame(t, review([]), CAMPAIGN))
        self.assertEqual(list(out['observation_type']),
                         [exports.TYPE_BLANK, exports.TYPE_HUMAN])
        self.assertTrue((out['species_latin'] == '').all())
        self.assertTrue((out['review_resolution'] == RESOLUTION_SWEEP_ONLY).all())

    def test_reviewed_row_uses_the_review_not_the_sweep(self):
        """The sweep said human; the review named a dog. animal > vehicle > human."""
        t = total([('CT01', 'a.JPG', 'human')])
        r = review([('CT01', 'a.JPG', 'Canis lupus familiaris', 'Perro')])
        out = resolve_observation(compose_ingest_frame(t, r, CAMPAIGN))
        self.assertEqual(out['observation_type'][0], exports.TYPE_ANIMAL)
        self.assertEqual(out['review_resolution'][0], RESOLUTION_SPECIES_NAMED)

    def test_negating_review_beats_the_sweeps_animal(self):
        t = total([('CT01', 'a.JPG', 'animal')])
        r = review([('CT01', 'a.JPG', '', 'No es un animal')])
        out = resolve_observation(compose_ingest_frame(t, r, CAMPAIGN))
        self.assertEqual(out['observation_type'][0], exports.TYPE_BLANK)
        self.assertEqual(out['review_resolution'][0], RESOLUTION_TYPE_FROM_COMMENT)

    def test_mixed_frame_resolves_each_row_by_its_own_source(self):
        t = total([('CT01', 'a.JPG', 'animal'), ('CT01', 'b.JPG', 'blank'),
                   ('CT01', 'c.JPG', 'vehicle')])
        r = review([('CT01', 'a.JPG', 'Pudu puda', 'Pudú')])
        out = resolve_observation(compose_ingest_frame(t, r, CAMPAIGN))
        self.assertEqual(list(out['review_resolution']),
                         [RESOLUTION_SPECIES_NAMED, RESOLUTION_SWEEP_ONLY,
                          RESOLUTION_SWEEP_ONLY])
        self.assertEqual(list(out['observation_type']),
                         [exports.TYPE_ANIMAL, exports.TYPE_BLANK, exports.TYPE_VEHICLE])

    def test_unclassified_on_an_unreviewed_row_is_refused(self):
        """The export gate refuses that campaign, so reaching here means it was bypassed."""
        t = total([('CT01', 'a.JPG', 'unclassified')])
        with self.assertRaises(UnmappedReviewComment):
            resolve_observation(compose_ingest_frame(t, review([]), CAMPAIGN))

    def test_no_sweep_only_row_is_ever_typed_animal(self):
        """Load-bearing: it is why the annual report's animal filter picks up nothing new."""
        t = total([('CT01', 'a.JPG', 'animal')])
        r = review([('CT01', 'a.JPG', 'Puma concolor', 'Puma')])
        out = resolve_observation(compose_ingest_frame(t, r, CAMPAIGN))
        sweep_animals = out[(out['review_resolution'] == RESOLUTION_SWEEP_ONLY)
                            & (out['observation_type'] == exports.TYPE_ANIMAL)]
        self.assertEqual(len(sweep_animals), 0)


if __name__ == '__main__':
    unittest.main()
