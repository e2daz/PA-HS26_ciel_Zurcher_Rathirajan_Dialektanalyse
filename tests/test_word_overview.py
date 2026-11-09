"""Tests for the statistical word overview (DIAL-27)."""

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "domain"))

from word_overview import (  # noqa: E402
    DELETION, DELETION_LABEL, EXACT_MATCH, VARIANT, summarize_model, variant_frequencies,
)


def _rows(hypotheses, similarities=None):
    similarities = similarities or [0.5] * len(hypotheses)
    return pd.DataFrame({"hypothesis_word": hypotheses, "similarity": similarities})


class TestSummarizeModel(unittest.TestCase):
    def test_counts_exact_matches_deletions_and_distinct_variants(self):
        rows = _rows(["nicht", "nicht", "nid", "nid", "net", None])

        summary = summarize_model(rows, "nicht")

        self.assertEqual(summary.occurrences, 6)
        self.assertEqual(summary.exact_matches, 2)
        self.assertEqual(summary.deletions, 1)
        self.assertEqual(summary.variant_count, 2)
        self.assertAlmostEqual(summary.exact_match_share, 2 / 6)

    def test_empty_rows_have_zero_share(self):
        summary = summarize_model(_rows([]), "nicht")

        self.assertEqual(summary.occurrences, 0)
        self.assertEqual(summary.exact_match_share, 0.0)


class TestVariantFrequencies(unittest.TestCase):
    def test_sorted_by_count_with_kind_share_and_mean_similarity(self):
        rows = _rows(["nid", "nicht", "nid", None, "nid", "nicht"], [0.4, 1.0, 0.6, 0.0, 0.5, 1.0])

        table = variant_frequencies(rows, "nicht")

        self.assertEqual(table["transcription"].tolist(), ["nid", "nicht", DELETION_LABEL])
        self.assertEqual(table["kind"].tolist(), [VARIANT, EXACT_MATCH, DELETION])
        self.assertEqual(table["count"].tolist(), [3, 2, 1])
        self.assertAlmostEqual(table["share"].sum(), 100.0)
        self.assertAlmostEqual(table.loc[0, "share"], 50.0)
        self.assertAlmostEqual(table.loc[0, "mean_similarity"], 0.5)

    def test_ties_are_ordered_alphabetically(self):
        table = variant_frequencies(_rows(["net", "nid"]), "nicht")

        self.assertEqual(table["transcription"].tolist(), ["net", "nid"])

    def test_empty_rows_give_empty_table_with_columns(self):
        table = variant_frequencies(_rows([]), "nicht")

        self.assertTrue(table.empty)
        self.assertIn("share", table.columns)


if __name__ == "__main__":
    unittest.main()
