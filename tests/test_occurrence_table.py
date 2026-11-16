"""Tests for the per-occurrence alignment table (DIAL-29)."""

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "domain"))

from occurrence_table import (  # noqa: E402
    DELETION, DELETION_LABEL, EXACT_MATCH, SHOW_DIT_DIFFERS, SHOW_MODELS_DIFFER, VARIANT,
    build_occurrence_table, filter_occurrences, words_by_frequency,
)


def _alignment(path, index, model, hypothesis, similarity, region="ZH", reference="das ist nicht gut"):
    return {
        "dataset": "STT4SG-350", "path": path, "reference_index": index, "model": model,
        "reference_word": "nicht", "hypothesis_word": hypothesis, "similarity": similarity,
        "dialect_region": region, "gender": "f", "age": "30-39", "reference": reference,
        "dit_hypothesis": f"dit {path}", "dat_hypothesis": f"dat {path}",
    }


def _word_rows():
    return pd.DataFrame([
        _alignment("a", 2, "dialect-ignorant", "nid", 0.6),
        _alignment("a", 2, "dialect-aware", "nicht", 1.0),
        # The same sentence contains the word twice: two occurrences.
        _alignment("a", 5, "dialect-ignorant", "nicht", 1.0),
        _alignment("a", 5, "dialect-aware", "nicht", 1.0),
        _alignment("b", 1, "dialect-ignorant", None, 0.0, region="BE", reference="nicht schon wieder"),
        _alignment("b", 1, "dialect-aware", "nid", 0.6, region="BE", reference="nicht schon wieder"),
    ])


class TestBuildOccurrenceTable(unittest.TestCase):
    def test_one_row_per_occurrence_with_both_models(self):
        table = build_occurrence_table(_word_rows(), "nicht")

        self.assertEqual(len(table), 3)
        self.assertEqual(table[["path", "reference_index"]].values.tolist(), [["a", 2], ["a", 5], ["b", 1]])
        self.assertEqual(table["dit_word"].tolist(), ["nid", "nicht", DELETION_LABEL])
        self.assertEqual(table["dat_word"].tolist(), ["nicht", "nicht", "nid"])
        self.assertEqual(table["dit_result"].tolist(), [VARIANT, EXACT_MATCH, DELETION])
        self.assertEqual(table["dat_similarity"].tolist(), [1.0, 1.0, 0.6])

    def test_sentence_metadata_is_attached(self):
        table = build_occurrence_table(_word_rows(), "nicht")

        self.assertEqual(table["dialect_region"].tolist(), ["ZH", "ZH", "BE"])
        self.assertEqual(table.loc[2, "reference"], "nicht schon wieder")
        self.assertEqual(table.loc[0, "dit_hypothesis"], "dit a")

    def test_empty_input_gives_empty_table(self):
        table = build_occurrence_table(_word_rows().iloc[0:0], "nicht")

        self.assertTrue(table.empty)


class TestFilterOccurrences(unittest.TestCase):
    def setUp(self):
        self.table = build_occurrence_table(_word_rows(), "nicht")

    def test_no_filters_keep_everything(self):
        self.assertEqual(len(filter_occurrences(self.table)), 3)

    def test_search_is_case_insensitive_across_sentences_and_words(self):
        self.assertEqual(filter_occurrences(self.table, search="SCHON")["path"].tolist(), ["b"])
        self.assertEqual(filter_occurrences(self.table, search="nid")["path"].tolist(), ["a", "b"])

    def test_word_and_region_filters(self):
        self.assertEqual(len(filter_occurrences(self.table, dit_words=["nid", DELETION_LABEL])), 2)
        self.assertEqual(len(filter_occurrences(self.table, dat_words=["nicht"])), 2)
        self.assertEqual(filter_occurrences(self.table, regions=["BE"])["path"].tolist(), ["b"])

    def test_show_filters(self):
        self.assertEqual(filter_occurrences(self.table, show=SHOW_DIT_DIFFERS)["reference_index"].tolist(), [2, 1])
        self.assertEqual(filter_occurrences(self.table, show=SHOW_MODELS_DIFFER)["reference_index"].tolist(), [2, 1])

    def test_unknown_show_filter_raises(self):
        with self.assertRaises(ValueError):
            filter_occurrences(self.table, show="nonsense")

    def test_words_by_frequency(self):
        self.assertEqual(words_by_frequency(self.table, "dat_word"), ["nicht", "nid"])
        self.assertEqual(words_by_frequency(self.table, "dit_word"), [DELETION_LABEL, "nicht", "nid"])


if __name__ == "__main__":
    unittest.main()
