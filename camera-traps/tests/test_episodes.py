"""The independence rule — `camtrap/episodes.py`.

The first test is the one that matters most: detections at 0, 20 and 40 minutes are
TWO events, not one. `apply_verdicts.py` got this wrong until 2026-08-26 by comparing
each row against its predecessor, and it undercounted by 33% (523 against 696) in the
script that writes `events_clean.parquet`. That is the failure mode a fixture has to
hold, because both answers look plausible in a report.
"""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from camtrap import episodes
from camtrap.observations import CAMPAIGN_ORDER, read_campaigns

BASE = datetime(2026, 5, 14, 9, 0, 0)


def frame(rows) -> pd.DataFrame:
    """`rows` are (minutes_from_BASE, station, species, segment); None minutes = no clock."""
    return pd.DataFrame([
        {'campaign': 'otono_2026', 'station_canonical': station,
         'species_latin': species,
         'datetime': None if minutes is None else BASE + timedelta(minutes=minutes),
         'segment': segment}
        for minutes, station, species, segment in rows
    ])


def label(rows) -> list:
    df = frame(rows)
    return list(episodes.label(df, segments=df['segment']))


def n_episodes(rows) -> int:
    df = frame(rows)
    return int(episodes.label(df, segments=df['segment']).nunique(dropna=True))


class TestTheGapIsMeasuredFromTheLastRetainedDetection(unittest.TestCase):

    def test_zero_twenty_forty_is_two_events(self):
        """The camtrapR definition, and the one pehuen's R/00_admissibility.R uses.
        Comparing each row against its predecessor makes this ONE event, because
        neither step exceeds 30 minutes."""
        self.assertEqual(n_episodes([(0, 'CT14', 'Puma concolor', '1'),
                                     (20, 'CT14', 'Puma concolor', '1'),
                                     (40, 'CT14', 'Puma concolor', '1')]), 2)

    def test_a_long_burst_is_still_counted_from_the_retained_frame(self):
        """Ten minutes apart for two hours: an event every 30 minutes, not one event
        and not thirteen."""
        rows = [(m, 'CT14', 'Puma concolor', '1') for m in range(0, 121, 10)]
        self.assertEqual(n_episodes(rows), 5)     # 0, 30, 60, 90, 120

    def test_exactly_the_threshold_starts_a_new_event(self):
        self.assertEqual(n_episodes([(0, 'CT14', 'Puma concolor', '1'),
                                     (30, 'CT14', 'Puma concolor', '1')]), 2)

    def test_a_second_under_the_threshold_does_not(self):
        df = frame([(0, 'CT14', 'Puma concolor', '1')])
        df.loc[1] = {'campaign': 'otono_2026', 'station_canonical': 'CT14',
                     'species_latin': 'Puma concolor',
                     'datetime': BASE + timedelta(minutes=29, seconds=59),
                     'segment': '1'}
        self.assertEqual(episodes.label(df, segments=df['segment']).nunique(), 1)


class TestTheKey(unittest.TestCase):

    def test_two_species_at_one_station_are_two_events(self):
        self.assertEqual(n_episodes([(0, 'CT14', 'Puma concolor', '1'),
                                     (1, 'CT14', 'Lycalopex culpaeus', '1')]), 2)

    def test_one_species_at_two_stations_is_two_events(self):
        self.assertEqual(n_episodes([(0, 'CT14', 'Puma concolor', '1'),
                                     (1, 'CT20', 'Puma concolor', '1')]), 2)

    def test_the_campaign_is_in_the_id_so_two_campaigns_cannot_collide(self):
        """`read_campaigns` concatenates campaigns. Without the campaign in the id,
        `CT14|Puma concolor|1` names a different event in each one and a cross-campaign
        nunique() undercounts."""
        df = frame([(0, 'CT14', 'Puma concolor', '1')])
        other = df.copy()
        other['campaign'] = 'primavera_2025'
        both = pd.concat([df, other], ignore_index=True)
        ids = episodes.label(both, segments=both['segment'])
        self.assertEqual(ids.nunique(), 2)
        self.assertTrue(ids.iloc[0].startswith('otono_2026|'))

    def test_the_id_reads_as_what_it_is(self):
        self.assertEqual(label([(0, 'CT14', 'Puma concolor', '1')])[0],
                         'otono_2026|CT14|Puma concolor|1')


class TestAnEpisodeCannotCrossAClockSegment(unittest.TestCase):

    def test_a_segment_change_breaks_an_episode_however_small_the_gap(self):
        """A segment boundary is a clock reset, so the interval across it is arithmetic
        on two different clocks. Twenty minutes across a reset is not twenty minutes."""
        self.assertEqual(n_episodes([(0, 'CT14', 'Puma concolor', '1'),
                                     (5, 'CT14', 'Puma concolor', '2')]), 2)

    def test_ids_stay_unique_across_segments(self):
        """The counter runs per station-species, not per segment — otherwise both
        segments produce episode 1 and the two collide into one id."""
        ids = label([(0, 'CT14', 'Puma concolor', '1'),
                     (5, 'CT14', 'Puma concolor', '2')])
        self.assertEqual(len(set(ids)), 2)

    def test_without_segments_the_frame_is_treated_as_one(self):
        df = frame([(0, 'CT14', 'Puma concolor', ''), (5, 'CT14', 'Puma concolor', '')])
        self.assertEqual(episodes.label(df).nunique(), 1)


class TestWhatGetsNoEpisode(unittest.TestCase):

    def test_a_row_with_no_clock_gets_none(self):
        """419 animal rows carry a station and no datetime. They are valid PRESENCE
        records and must not read as an event of their own."""
        self.assertTrue(pd.isna(label([(None, 'CT14', 'Puma concolor', '1')])[0]))

    def test_a_row_with_no_species_gets_none(self):
        self.assertTrue(pd.isna(label([(0, 'CT14', '', '1')])[0]))

    def test_count_ignores_them(self):
        df = frame([(0, 'CT14', 'Puma concolor', '1'),
                    (None, 'CT14', 'Puma concolor', '1'),
                    (5, 'CT14', '', '1')])
        df[episodes.COLUMN] = episodes.label(df, segments=df['segment'])
        self.assertEqual(episodes.count(df), 1)

    def test_an_empty_frame_is_not_an_error(self):
        df = frame([(None, 'CT14', '', '')])
        self.assertEqual(int(episodes.label(df).notna().sum()), 0)


class TestOrderIndependence(unittest.TestCase):

    def test_row_order_does_not_change_the_answer(self):
        """A transcription or export can arrive in any order; the grouping is by time,
        not by position."""
        rows = [(0, 'CT14', 'Puma concolor', '1'), (20, 'CT14', 'Puma concolor', '1'),
                (40, 'CT14', 'Puma concolor', '1'), (5, 'CT20', 'Puma concolor', '1')]
        forward = frame(rows)
        backward = frame(list(reversed(rows)))
        self.assertEqual(
            episodes.label(forward, segments=forward['segment']).nunique(),
            episodes.label(backward, segments=backward['segment']).nunique())


class TestThePublishedTables(unittest.TestCase):

    def test_the_committed_tables_carry_the_column(self):
        df = read_campaigns(*CAMPAIGN_ORDER)
        self.assertIn(episodes.COLUMN, df.columns)

    def test_only_identified_animals_have_episodes(self):
        """The column answers "how many independent detections of this species", so a
        blank or a human frame has no episode by construction."""
        df = read_campaigns(*CAMPAIGN_ORDER)
        with_id = df[df[episodes.COLUMN].notna()]
        self.assertEqual(set(with_id.observation_type), {'animal'})
        self.assertFalse((with_id.species_latin == '').any())
        self.assertFalse(with_id.datetime.isna().any())

    def test_the_count_is_the_one_the_corrected_rule_produces(self):
        """696 across the three campaigns, measured 2026-08-26 before the column
        existed. The stale predecessor rule gives 523; if this figure ever reads 523
        again, the rule regressed rather than the data changing."""
        self.assertEqual(episodes.count(read_campaigns(*CAMPAIGN_ORDER)), 696)


if __name__ == '__main__':
    unittest.main()
