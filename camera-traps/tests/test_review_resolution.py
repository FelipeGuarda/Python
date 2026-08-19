"""Fixtures for camtrap.observations.resolve_review — the reviewer-verdict rules.

Run:  python3 -m unittest discover -s tests -v

Every fixture here is a row shape that appeared in a real reviewed CSV on 2026-08-19.
The two that matter most are the pair that pull in opposite directions:

    a named species against a sweep that said human/vehicle  -> animal wins
    a negating comment against a sweep that said animal      -> the negation wins

Getting the second one wrong is not hypothetical: it is the state the three campaigns
were actually in, with 815 rows typed `animal` while the reviewer had written that the
frame holds no animal.
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camtrap import exports
from camtrap.observations import (
    RESOLUTION_PENDING_REVIEW,
    RESOLUTION_PENDING_TAXON,
    RESOLUTION_SPECIES_FROM_COMMENT,
    RESOLUTION_SPECIES_NAMED,
    RESOLUTION_TYPE_FROM_COMMENT,
    UnmappedReviewComment,
    audit_review_comments,
    resolve_review,
)


def reviewed(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """A minimal reviewed frame: (scientificName, observationComments) per row."""
    return pd.DataFrame({
        'scientificName': [r[0] for r in rows],
        'observationComments': [r[1] for r in rows],
    })


class NamedSpeciesWins(unittest.TestCase):
    """R1 — a typed scientificName makes the row `animal`, whatever else is around."""

    def test_named_species_is_animal(self):
        out = resolve_review(reviewed([('Puma concolor', 'Puma')]))
        self.assertEqual(out['observation_type'][0], exports.TYPE_ANIMAL)
        self.assertEqual(out['species_latin'][0], 'Puma concolor')
        self.assertEqual(out['review_resolution'][0], RESOLUTION_SPECIES_NAMED)

    def test_dog_outranks_the_sweeps_vehicle_and_human(self):
        """The 13 Perro rows. animal > vehicle > human, agreed 2026-08-19."""
        out = resolve_review(reviewed([('Canis lupus familiaris', 'Perro')]))
        self.assertEqual(out['observation_type'][0], exports.TYPE_ANIMAL)
        self.assertEqual(out['species_latin'][0], 'Canis lupus familiaris')

    def test_horse_and_cow_rank_the_same_as_the_dog(self):
        """23 Caballo + 1 Vaca rows. The rule is any animal, not specifically a dog."""
        out = resolve_review(reviewed([
            ('Equus caballus', 'Caballo'),
            ('Bos taurus', 'Vaca'),
        ]))
        self.assertTrue((out['observation_type'] == exports.TYPE_ANIMAL).all())
        self.assertEqual(list(out['species_latin']), ['Equus caballus', 'Bos taurus'])

    def test_the_sweep_is_not_an_input_at_all(self):
        """An observationType column present in the frame must not change the result."""
        frame = reviewed([('Pudu puda', 'Pudú')])
        frame['observationType'] = ['human']
        out = resolve_review(frame)
        self.assertEqual(out['observation_type'][0], exports.TYPE_ANIMAL)


class NegationWins(unittest.TestCase):
    """R2 — an empty species plus a negating comment overrides the sweep's `animal`."""

    def test_not_an_animal_becomes_blank(self):
        out = resolve_review(reviewed([('', 'No es un animal')]))
        self.assertEqual(out['observation_type'][0], exports.TYPE_BLANK)
        self.assertEqual(out['species_latin'][0], '')
        self.assertEqual(out['review_resolution'][0], RESOLUTION_TYPE_FROM_COMMENT)

    def test_unrecognisable_becomes_unknown_not_blank(self):
        """`unknown` is Camtrap DP for looked-at-but-cannot-tell; `blank` claims empty."""
        out = resolve_review(reviewed([('', 'No reconocible')]))
        self.assertEqual(out['observation_type'][0], exports.TYPE_UNKNOWN)

    def test_humano_and_vehiculo_become_their_own_types(self):
        out = resolve_review(reviewed([('', 'humano'), ('', 'vehiculo')]))
        self.assertEqual(
            list(out['observation_type']), [exports.TYPE_HUMAN, exports.TYPE_VEHICLE]
        )

    def test_case_and_padding_do_not_matter(self):
        out = resolve_review(reviewed([('', '  NO ES UN ANIMAL  ')]))
        self.assertEqual(out['observation_type'][0], exports.TYPE_BLANK)


class SpeciesFromComment(unittest.TestCase):
    """R3 — the "Otro (especificar)" path, via species.yaml and nowhere else."""

    def test_spanish_common_name_resolves_to_latin(self):
        out = resolve_review(reviewed([('', 'chingue')]))
        self.assertEqual(out['observation_type'][0], exports.TYPE_ANIMAL)
        self.assertEqual(out['species_latin'][0], 'Conepatus chinga')
        self.assertEqual(out['review_resolution'][0], RESOLUTION_SPECIES_FROM_COMMENT)


class PendingDecisions(unittest.TestCase):
    """R4 — resolves to `unknown`, but tagged so the open question stays findable."""

    def test_coarse_taxon_is_tagged_apart_from_genuine_unknowns(self):
        out = resolve_review(reviewed([('', 'ave'), ('', 'roedor'), ('', 'No reconocible')]))
        self.assertEqual(
            list(out['review_resolution']),
            [RESOLUTION_PENDING_TAXON, RESOLUTION_PENDING_TAXON,
             RESOLUTION_TYPE_FROM_COMMENT],
        )
        self.assertTrue((out['observation_type'] == exports.TYPE_UNKNOWN).all())

    def test_workflow_note_is_tagged_apart_too(self):
        """Merging these into the genuine unknowns silently closes an open task."""
        out = resolve_review(reviewed([('', 'identificar'), ('', 'error de imagen')]))
        self.assertTrue(
            (out['review_resolution'] == RESOLUTION_PENDING_REVIEW).all()
        )

    def test_accented_and_unaccented_pitio_both_resolve(self):
        out = resolve_review(reviewed([('', 'Pitío'), ('', 'pitio')]))
        self.assertTrue((out['review_resolution'] == RESOLUTION_PENDING_TAXON).all())


class FailClosed(unittest.TestCase):
    """R5 — an unrecognised comment refuses the ingest rather than guessing."""

    def test_unknown_comment_refuses(self):
        with self.assertRaises(UnmappedReviewComment):
            resolve_review(reviewed([('', 'algo que nadie ha visto')]))

    def test_every_unknown_comment_is_named_at_once(self):
        """One run must surface the whole backlog, not the first item ten times."""
        frame = reviewed([('', 'comentario uno'), ('', 'comentario dos'),
                          ('', 'comentario uno'), ('Puma concolor', 'Puma')])
        with self.assertRaises(UnmappedReviewComment) as ctx:
            resolve_review(frame)
        message = str(ctx.exception)
        self.assertIn('comentario uno', message)
        self.assertIn('comentario dos', message)
        self.assertIn('3 reviewed row(s)', message)

    def test_empty_comment_on_empty_species_refuses(self):
        """That row is `unclassified` — never looked at. No rule may invent a verdict."""
        with self.assertRaises(UnmappedReviewComment):
            resolve_review(reviewed([('', '')]))

    def test_audit_reports_without_raising(self):
        counts = audit_review_comments(reviewed([('', 'no visto'), ('', 'no visto'),
                                                 ('', 'No es un animal')]))
        self.assertEqual(counts, {'no visto': 2})

    def test_audit_is_empty_when_every_row_resolves(self):
        self.assertEqual(audit_review_comments(reviewed([('Puma concolor', 'Puma')])), {})


class FrameShape(unittest.TestCase):
    """Alignment invariants the caller depends on."""

    def test_index_is_preserved(self):
        frame = reviewed([('Puma concolor', 'Puma'), ('', 'No es un animal')])
        frame.index = [7, 42]
        out = resolve_review(frame)
        self.assertEqual(list(out.index), [7, 42])

    def test_missing_comments_column_does_not_misalign(self):
        """A short Series would silently misalign every .loc[] against the frame."""
        frame = pd.DataFrame({'scientificName': ['Puma concolor', 'Pudu puda']})
        out = resolve_review(frame)
        self.assertEqual(len(out), 2)
        self.assertTrue((out['observation_type'] == exports.TYPE_ANIMAL).all())

    def test_empty_frame_resolves_to_empty(self):
        out = resolve_review(reviewed([]))
        self.assertEqual(len(out), 0)
        self.assertEqual(
            list(out.columns),
            ['observation_type', 'species_latin', 'review_resolution'],
        )


if __name__ == '__main__':
    unittest.main()
