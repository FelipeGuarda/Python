"""Co-presence tags — `camtrap/copresence.py`.

Run:  python -m unittest tests.test_copresence -v

The test that matters most is the scope one: an empty tag in a campaign nobody
checked must come out UNKNOWN, never FALSE. Three campaigns were reviewed before the
convention existed; reading their empty tags as "the dog was alone" would turn the
absence of a convention into a finding about dogs.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from camtrap import copresence
from camtrap.copresence import InvalidObservationTag, compose, parse, resolve

DOG = 'Canis lupus familiaris'


def rows(*spec):
    """spec: (tags, observation_type, species, reviewed)."""
    df = pd.DataFrame(spec, columns=['tags', 'kind', 'species', 'reviewed'])
    return df


def run(df, campaign='otono_2026'):
    return resolve(df['tags'], df['kind'], df['species'], df['reviewed'], campaign)


def value(series, i):
    v = series.iloc[i]
    return None if pd.isna(v) else bool(v)


class TestComposeAndParse(unittest.TestCase):

    def test_round_trip(self):
        for human in (False, True):
            for vehicle in (False, True):
                got = parse(compose(human, vehicle))
                self.assertEqual(copresence.TAG_HUMAN in got, human)
                self.assertEqual(copresence.TAG_VEHICLE in got, vehicle)

    def test_nothing_ticked_is_an_empty_cell(self):
        self.assertEqual(compose(False, False), '')

    def test_parse_tolerates_spacing_and_case(self):
        self.assertEqual(parse(' With:Human | with:vehicle |'),
                         {'with:human', 'with:vehicle'})

    def test_parse_of_an_empty_or_missing_cell(self):
        self.assertEqual(parse(''), frozenset())
        self.assertEqual(parse(None), frozenset())


class TestScope(unittest.TestCase):
    """FALSE is only ever written where someone looked."""

    def test_unchecked_campaign_empty_tag_is_unknown(self):
        with mock.patch.dict(copresence.CHECKED, {}, clear=True):
            out = run(rows(('', 'animal', DOG, True)))
        self.assertIsNone(value(out.with_human, 0))
        self.assertIsNone(value(out.with_vehicle, 0))

    def test_unchecked_campaign_present_tag_is_still_true(self):
        """An observed person is evidence whether or not the campaign was declared."""
        with mock.patch.dict(copresence.CHECKED, {}, clear=True):
            out = run(rows(('with:human', 'animal', DOG, True)))
        self.assertTrue(value(out.with_human, 0))
        self.assertIsNone(value(out.with_vehicle, 0))

    def test_species_scope_marks_only_that_species(self):
        checked = {'otono_2026': frozenset({DOG})}
        with mock.patch.dict(copresence.CHECKED, checked, clear=True):
            out = run(rows(('', 'animal', DOG, True),
                           ('', 'animal', 'Lycalopex culpaeus', True)))
        self.assertFalse(value(out.with_human, 0))
        self.assertIsNone(value(out.with_human, 1))

    def test_all_reviewed_scope_marks_every_reviewed_row(self):
        checked = {'otono_2026': copresence.ALL_REVIEWED}
        with mock.patch.dict(copresence.CHECKED, checked, clear=True):
            out = run(rows(('', 'animal', 'Lycalopex culpaeus', True),
                           ('', 'animal', DOG, False)))
        self.assertFalse(value(out.with_human, 0))
        self.assertIsNone(value(out.with_human, 1), 'a sweep-only row was never looked at')

    def test_scope_belongs_to_one_campaign(self):
        checked = {'otono_2025': copresence.ALL_REVIEWED}
        with mock.patch.dict(copresence.CHECKED, checked, clear=True):
            out = run(rows(('', 'animal', DOG, True)), campaign='otono_2026')
        self.assertIsNone(value(out.with_human, 0))


class TestThePriority(unittest.TestCase):
    """perro > person > vehicle: a tag may only name what ranks below the label."""

    def test_an_animal_may_carry_both(self):
        out = run(rows(('with:human|with:vehicle', 'animal', DOG, True)))
        self.assertTrue(value(out.with_human, 0))
        self.assertTrue(value(out.with_vehicle, 0))

    def test_a_person_may_carry_a_vehicle(self):
        out = run(rows(('with:vehicle', 'human', '', True)))
        self.assertTrue(value(out.with_vehicle, 0))
        self.assertIsNone(value(out.with_human, 0), 'with_human means nothing on a human row')

    def test_a_vehicle_tagged_with_a_person_is_refused(self):
        """The person outranks the vehicle, so the frame should have been `human`."""
        with self.assertRaises(InvalidObservationTag) as ctx:
            run(rows(('with:human', 'vehicle', '', True)))
        self.assertIn("'with:human' on a 'vehicle' row", str(ctx.exception))

    def test_a_row_tagged_with_its_own_type_is_refused(self):
        with self.assertRaises(InvalidObservationTag):
            run(rows(('with:human', 'human', '', True)))

    def test_a_blank_carrying_a_tag_is_refused(self):
        with self.assertRaises(InvalidObservationTag):
            run(rows(('with:human', 'blank', '', True)))

    def test_not_applicable_rows_stay_unknown_even_in_scope(self):
        with mock.patch.dict(copresence.CHECKED, {'otono_2026': copresence.ALL_REVIEWED},
                             clear=True):
            out = run(rows(('', 'blank', '', True), ('', 'vehicle', '', True)))
        self.assertIsNone(value(out.with_human, 0))
        self.assertIsNone(value(out.with_vehicle, 1))


class TestTheVocabulary(unittest.TestCase):

    def test_a_typo_stops_the_ingest_and_is_named(self):
        """A misspelt tag must not read as 'alone'."""
        with self.assertRaises(InvalidObservationTag) as ctx:
            run(rows(('with:humna', 'animal', DOG, True)))
        self.assertIn("'with:humna'", str(ctx.exception))

    def test_every_problem_is_listed_at_once(self):
        with self.assertRaises(InvalidObservationTag) as ctx:
            run(rows(('with:humna', 'animal', DOG, True),
                     ('with:human', 'vehicle', '', True)))
        self.assertIn('with:humna', str(ctx.exception))
        self.assertIn("on a 'vehicle' row", str(ctx.exception))

    def test_dtype_and_index(self):
        df = rows(('', 'animal', DOG, True), ('with:human', 'animal', DOG, True))
        df.index = [10, 20]
        out = run(df)
        self.assertEqual(list(out.index), [10, 20])
        self.assertEqual(str(out.with_human.dtype), 'boolean')


if __name__ == '__main__':
    unittest.main()
