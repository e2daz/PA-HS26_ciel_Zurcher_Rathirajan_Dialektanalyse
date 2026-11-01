import re

import pandas as pd
from text_to_num import alpha2digit


def clean_word(word: str) -> str:
    """Clean and normalize a single word for comparison.

    Lowercases, normalizes ß, replaces "%" with "prozent", contracts spelled-out
    unit/currency/magnitude words to their abbreviations (so "Franken" and
    "Fr." both end up as "fr"), converts spelled-out numerals to digits via
    text2num (ordinals included: "ersten" and "1." both become "1"), and
    strips punctuation.
    """
    if pd.isna(word):
        return ""

    word = str(word).lower()
    word = word.replace("ß", "ss")
    word = word.replace("%", "prozent")
    word = word.replace("zentimeter", "cm")
    word = word.replace("millimeter", "mm")
    word = word.replace("kilometer", "km")
    word = word.replace("kilogramm", "kg")
    word = word.replace("quadratmeter", "m2")
    word = word.replace("kubikmeter", "m3")
    word = word.replace("milligramm", "mg")
    word = word.replace("milliliter", "ml")
    word = word.replace("millionen", "mio")
    word = word.replace("milliarden", "mrd")
    word = word.replace("beziehungsweise", "bzw")
    word = word.replace("franken", "fr")

    # ordinal_threshold=0: text2num keeps ordinals <= 3 ("erste", "zweiten", "dritte")
    # as words by default, so REF "1." would never meet a transcript's "ersten".
    word = alpha2digit(word, "de", ordinal_threshold=0)
    return re.sub(r"[^a-z0-9äöü]", "", word)


def tokenize(text: str) -> list[str]:
    """Split a sentence on whitespace and clean each token, dropping empty results.

    A percent sign glued to a number ("6%", as Whisper writes it) is split off
    first so it aligns with the reference's two tokens "6 Prozent".
    """
    text = re.sub(r"(\d)\s*%", r"\1 %", str(text))
    return [w for w in (clean_word(t) for t in text.split()) if w]
