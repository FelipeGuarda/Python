"""The published contract must actually gate. A contract nobody verifies is a comment.

These fixtures exist because of a specific near-miss: on 2026-08-19 the canonical tables
went from 3,359 rows to 35,807 and every consumer kept running silently. The report
happened to filter on `observation_type` and stayed correct; nothing checked that.
"""

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camtrap import canonical_state, observations


def _frame(**over) -> pd.DataFrame:
    """A one-row frame in canonical shape."""
    row = {
        "campaign": "otono_2025", "camera_num": 5, "station_canonical": "CT05",
        "datetime": pd.Timestamp("2025-01-01 12:00"), "valid_date": True,
        "valid_time_of_day": True, "valid_effort": True, "repair_method": "none",
        "observation_type": "animal", "species_latin": "Puma concolor",
        "review_outcome": "confirmed", "review_resolution": "species_named",
        "file_name": "01010001.JPG", "rel_path": "CT05/01010001.JPG",
        "observation_comments": "Puma", "classification_probability": 0.91,
    }
    row.update(over)
    return pd.DataFrame([row])


class TestSchemaIsTheContract(unittest.TestCase):
    def test_published_columns_are_exactly_canonical_columns(self):
        """The state file must not drift from the module that defines the schema."""
        state = canonical_state.build()
        self.assertEqual(state["columns"], list(observations.CANONICAL_COLUMNS))

    def test_the_two_audit_columns_are_in_the_contract(self):
        """Added 2026-08-20; they are the only Timelapse fields nothing else can supply."""
        cols = list(observations.CANONICAL_COLUMNS)
        self.assertIn("observation_comments", cols)
        self.assertIn("classification_probability", cols)

    def test_retired_campaign_is_not_in_the_published_state(self):
        """pv_2025_2026 is a review pass, not a campaign.

        It reached CAMPAIGN_ORDER once and silently outranked primavera, reverting 606
        freshly reviewed rows. A directory kept for provenance must never re-enter the
        contract just by existing on disk.
        """
        self.assertNotIn("pv_2025_2026", canonical_state.PUBLISHED_CAMPAIGNS)


class TestDiffDetectsRealChanges(unittest.TestCase):
    def setUp(self):
        self.current = canonical_state.build()

    def test_agrees_with_itself(self):
        self.assertEqual(canonical_state.diff(self.current, self.current), [])

    def test_row_count_change_is_caught(self):
        """The 3,359 -> 35,807 rebuild, which passed unnoticed before this gate existed."""
        published = json.loads(json.dumps(self.current))
        published["campaigns"]["primavera_2025"]["n_rows"] = 744
        problems = canonical_state.diff(published, self.current)
        self.assertTrue(any("primavera_2025.n_rows" in p for p in problems), problems)

    def test_added_column_is_caught(self):
        published = json.loads(json.dumps(self.current))
        published["columns"] = [c for c in published["columns"]
                                if c != "classification_probability"]
        problems = canonical_state.diff(published, self.current)
        self.assertTrue(any("added since publish" in p for p in problems), problems)

    def test_column_reorder_is_caught(self):
        """CANONICAL_COLUMNS declares an ORDER; to_canonical reindexes to match it."""
        published = json.loads(json.dumps(self.current))
        published["columns"] = list(reversed(published["columns"]))
        problems = canonical_state.diff(published, self.current)
        self.assertTrue(any("ORDER changed" in p for p in problems), problems)

    def test_schema_version_bump_is_caught(self):
        published = json.loads(json.dumps(self.current))
        published["schema_version"] = canonical_state.SCHEMA_VERSION - 1
        problems = canonical_state.diff(published, self.current)
        self.assertTrue(any("schema_version" in p for p in problems), problems)

    def test_station_count_change_is_caught(self):
        """A station appearing or vanishing is the defect this whole pass was about."""
        published = json.loads(json.dumps(self.current))
        published["campaigns"]["otono_2025"]["n_stations"] = 20
        problems = canonical_state.diff(published, self.current)
        self.assertTrue(any("otono_2025.n_stations" in p for p in problems), problems)


class TestConsumerGuard(unittest.TestCase):
    def test_canonical_frame_passes(self):
        canonical_state.assert_columns(_frame(), canonical_state.build())

    def test_frame_missing_a_column_is_refused(self):
        df = _frame().drop(columns=["valid_effort"])
        with self.assertRaises(canonical_state.CanonicalStateError) as cm:
            canonical_state.assert_columns(df, canonical_state.build())
        self.assertIn("valid_effort", str(cm.exception))

    def test_pre_contract_frame_is_refused_with_a_useful_message(self):
        """A parquet written before 2026-08-20 lacks the two audit columns."""
        df = _frame().drop(columns=["observation_comments", "classification_probability"])
        with self.assertRaises(canonical_state.CanonicalStateError) as cm:
            canonical_state.assert_columns(df, canonical_state.build())
        msg = str(cm.exception)
        self.assertIn("observation_comments", msg)
        self.assertIn("timestamps.py", msg)  # tells the reader how to fix it

    def test_extra_columns_are_allowed(self):
        """A consumer that adds its own derived columns must not be punished for it."""
        df = _frame()
        df["my_own_derived_column"] = 1
        canonical_state.assert_columns(df, canonical_state.build())


class TestPublishedFileIsCurrent(unittest.TestCase):
    def test_the_committed_state_file_matches_the_committed_parquets(self):
        """The real gate. Fails if someone rebuilt the parquets and forgot to re-publish.

        This is the assertion the whole module exists for: it turns "the contract is
        documented" into "the contract is true right now".
        """
        try:
            canonical_state.verify()
        except canonical_state.CanonicalStateError as e:
            self.fail(str(e))


if __name__ == "__main__":
    unittest.main()
