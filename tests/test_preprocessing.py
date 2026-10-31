"""Tests for preprocessing.clean_word."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "domain"))

from preprocessing import clean_word


class TestCleanWordBasics(unittest.TestCase):

    def test_lowercase(self):
        self.assertEqual(clean_word("Haus"), "haus")

    def test_eszett_normalised_to_ss(self):
        self.assertEqual(clean_word("Straße"), "strasse")

    def test_umlauts_preserved(self):
        self.assertEqual(clean_word("über"), "über")

    def test_punctuation_stripped(self):
        self.assertEqual(clean_word("Hallo,"), "hallo")
        self.assertEqual(clean_word("Welt!"), "welt")

    def test_nan_returns_empty_string(self):
        import pandas as pd
        self.assertEqual(clean_word(pd.NA), "")
        self.assertEqual(clean_word(float("nan")), "")

    def test_common_words_uncorrupted(self):
        self.assertEqual(clean_word("Frau"), "frau")
        self.assertEqual(clean_word("laufen"), "laufen")


class TestCleanWordAbbreviations(unittest.TestCase):
    """Spelled-out unit/currency/magnitude words contract to their abbreviation,
    so REF "Zentimeter" and DIT "cm" both clean to the same token."""

    def test_zentimeter_normalises_to_cm(self):
        self.assertEqual(clean_word("Zentimeter"), "cm")
        self.assertEqual(clean_word("cm"), "cm")

    def test_kilogramm_normalises_to_kg(self):
        self.assertEqual(clean_word("Kilogramm"), "kg")
        self.assertEqual(clean_word("kg"), "kg")

    def test_kilometer_normalises_to_km(self):
        self.assertEqual(clean_word("Kilometer"), "km")
        self.assertEqual(clean_word("km"), "km")

    def test_franken_normalises_to_fr(self):
        self.assertEqual(clean_word("Franken"), "fr")
        self.assertEqual(clean_word("Fr."), "fr")

    def test_millionen_normalises_to_mio(self):
        self.assertEqual(clean_word("Millionen"), "mio")
        self.assertEqual(clean_word("Mio."), "mio")

    def test_milliarden_normalises_to_mrd(self):
        self.assertEqual(clean_word("Milliarden"), "mrd")
        self.assertEqual(clean_word("Mrd."), "mrd")

    def test_quadratmeter_normalises_to_m2(self):
        self.assertEqual(clean_word("Quadratmeter"), "m2")
        self.assertEqual(clean_word("m2"), "m2")

    def test_kubikmeter_normalises_to_m3(self):
        self.assertEqual(clean_word("Kubikmeter"), "m3")
        self.assertEqual(clean_word("m3"), "m3")

    def test_beziehungsweise_normalises_to_bzw(self):
        self.assertEqual(clean_word("beziehungsweise"), "bzw")
        self.assertEqual(clean_word("bzw."), "bzw")

    def test_percent_inside_word_expands_to_prozent(self):
        # "%" is replaced inside the token (so "60%" stays a single word "60prozent").
        self.assertEqual(clean_word("60%"), "60prozent")
        self.assertEqual(clean_word("Prozent"), "prozent")


class TestCleanWordDigits(unittest.TestCase):

    def test_digits_preserved(self):
        self.assertEqual(clean_word("175"), "175")

    def test_decimal_separator_stripped(self):
        # "2,6" — punctuation strip removes the comma, digits stay.
        self.assertEqual(clean_word("2,6"), "26")
        self.assertEqual(clean_word("2.6"), "26")


class TestCleanWordSpelledNumerals(unittest.TestCase):
    """text2num normalises single-word spelled numerals to digits so REF
    "hundertsiebzig" and DIT "170" both end up as the digit form."""

    def test_compound_numeral_becomes_digits(self):
        self.assertEqual(clean_word("hundertsiebzig"), "170")

    def test_year_form_becomes_digits(self):
        self.assertEqual(clean_word("zweitausendfünf"), "2005")


class TestCleanWordOrdinals(unittest.TestCase):
    """Spelled-out ordinals of any inflection match REF digit ordinals like "1."."""

    def test_low_ordinals_match_digit_ordinal(self):
        self.assertEqual(clean_word("1."), "1")
        for word in ("erste", "ersten", "erster", "erstes"):
            self.assertEqual(clean_word(word), "1", word)
        self.assertEqual(clean_word("zweiten"), "2")
        self.assertEqual(clean_word("dritte"), "3")

    def test_higher_ordinals_match_digit_ordinal(self):
        self.assertEqual(clean_word("vierte"), "4")
        self.assertEqual(clean_word("zwanzigste"), "20")

    def test_non_numeral_lookalikes_untouched(self):
        self.assertEqual(clean_word("ein"), "ein")
        self.assertEqual(clean_word("eine"), "eine")
        self.assertEqual(clean_word("viertel"), "viertel")


if __name__ == "__main__":
    unittest.main()