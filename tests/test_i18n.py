"""Tests for the translation files and the t() helper (DIAL-30)."""

import re
import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIS_DIR = PROJECT_ROOT / "visualization"
sys.path.insert(0, str(VIS_DIR))

import i18n  # noqa: E402

# Keys the code builds at run time, e.g. t(f"tabs.{key}"); each entry lists every value the f-string can take.
_DYNAMIC_KEYS = {
    "tabs.{key}": [f"tabs.{key}" for key in ("overview", "levenshtein", "regions", "occurrences")],
    "tabs.{key}_{side}": [f"tabs.{key}_{side}" for key in ("overview", "levenshtein", "regions", "occurrences")
                          for side in ("reference", "dit")],
    "search.side_{side}": ["search.side_reference", "search.side_dit"],
    "search.box_{search_side}": ["search.box_reference", "search.box_dit"],
    "search.box_{search_side}_help": ["search.box_reference_help", "search.box_dit_help"],
    "examples.type_{kind}": ["examples.type_lexical", "examples.type_phonetic"],
    "cloud.compare_{option}": ["cloud.compare_ref_dit", "cloud.compare_dat_dit"],
    "levenshtein.rank_{rank_by}": ["levenshtein.rank_tfidf", "levenshtein.rank_count", "levenshtein.rank_similarity"],
    "regions.values_{option}": ["regions.values_count", "regions.values_share"],
    "occurrences.show_{option}": ["occurrences.show_all", "occurrences.show_dit_differs",
                                  "occurrences.show_models_differ"],
    "graph.{key}": [f"graph.{key}" for key in ("no_variants", "summary", "edge_tooltip", "hidden", "show_again",
                                               "png_failed", "ring_distance", "node_distance", "node_alignments",
                                               "node_similarity", "node_regions")],
}
# Keys passed around as values instead of directly to t().
_INDIRECT_KEYS = ["regional.model_dat", "regional.model_dit"]

# t("key"), plus the graph's _attr("key") and _help_icon("key"), which call t() themselves.
_T_CALL = re.compile(r"""\b(?:t|_attr|_help_icon)\(\s*(f?)(["'])(.+?)\2""")


def _keys_used_in_code() -> set[str]:
    keys = set(_INDIRECT_KEYS)
    for path in VIS_DIR.rglob("*.py"):
        for is_fstring, _, key in _T_CALL.findall(path.read_text(encoding="utf-8")):
            if is_fstring:
                if key not in _DYNAMIC_KEYS:
                    raise AssertionError(f"{path.name}: add t(f\"{key}\") to _DYNAMIC_KEYS")
                keys.update(_DYNAMIC_KEYS[key])
            else:
                keys.add(key)
    return keys


class TestLocaleFiles(unittest.TestCase):
    def setUp(self):
        self.en = i18n.translations("en")
        self.de = i18n.translations("de")

    def test_both_files_have_the_same_keys(self):
        self.assertEqual(sorted(set(self.en) - set(self.de)), [], "missing in de.json")
        self.assertEqual(sorted(set(self.de) - set(self.en)), [], "missing in en.json")

    def test_placeholders_match(self):
        for key, text in self.en.items():
            with self.subTest(key=key):
                self.assertEqual(i18n.placeholders(self.de[key]), i18n.placeholders(text))

    def test_no_empty_texts(self):
        for language, texts in (("en", self.en), ("de", self.de)):
            for key, text in texts.items():
                with self.subTest(language=language, key=key):
                    self.assertTrue(text.strip())

    def test_every_key_used_in_the_code_exists(self):
        self.assertEqual(sorted(_keys_used_in_code() - set(self.en)), [])

    def test_no_unused_keys(self):
        self.assertEqual(sorted(set(self.en) - _keys_used_in_code()), [])


class TestTranslate(unittest.TestCase):
    def test_translates_and_formats(self):
        self.assertEqual(i18n.translate("counts.alignments", "de", n="12"), "12 Alignments")
        self.assertEqual(i18n.translate("search.dit_title", "en", word="jetzt"), "DIT word: jetzt")

    def test_missing_german_key_falls_back_to_english(self):
        english = {"only.english": "Only English"}
        with mock.patch.object(i18n, "translations", lambda language: english if language == "en" else {}):
            self.assertEqual(i18n.translate("only.english", "de"), "Only English")

    def test_broken_german_placeholder_falls_back_to_english(self):
        texts = {"en": {"k": "{n} items"}, "de": {"k": "{anzahl} Stück"}}
        with mock.patch.object(i18n, "translations", lambda language: texts[language]):
            self.assertEqual(i18n.translate("k", "de", n=3), "3 items")

    def test_unknown_key_comes_back_unchanged(self):
        self.assertEqual(i18n.translate("no.such.key", "de"), "no.such.key")

    def test_unknown_language_uses_english(self):
        self.assertEqual(i18n.translate("nav.about", "fr"), "About")

    def test_all_translations(self):
        self.assertEqual(i18n.all_translations("tabs.overview"), {"Overview", "Übersicht"})


if __name__ == "__main__":
    unittest.main()
