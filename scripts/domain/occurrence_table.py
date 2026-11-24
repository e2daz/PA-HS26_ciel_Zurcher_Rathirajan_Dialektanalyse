"""One row per occurrence of a searched word, with what the other transcripts have there (DIAL-29, DIAL-35).

The searched word is either a reference word (one occurrence = one reference token, paired with what DIT and DAT
made of it) or a DIT word (one occurrence = one DIT token, paired with the reference word and the DAT word it was
aligned to).
"""

from collections.abc import Collection
from typing import Literal

import pandas as pd

DIT_MODEL = "dialect-ignorant"
DAT_MODEL = "dialect-aware"
DAT_DIT_MODEL = "dat-dit"
DELETION_LABEL = "(deleted)"
INSERTION_LABEL = "(inserted)"

EXACT_MATCH = "exact match"
VARIANT = "variant"
DELETION = "deletion"
INSERTION = "insertion"

Anchor = Literal["reference", "dit"]

# An occurrence is one token of the searched word: a sentence (path) and the word's position in its transcript.
_OCCURRENCE_KEYS: dict[str, list[str]] = {
    "reference": ["dataset", "path", "reference_index"],
    "dit": ["dataset", "path", "hypothesis_index"],
}
_SENTENCE_COLUMNS = ["dialect_region", "gender", "age", "reference", "dit_hypothesis", "dat_hypothesis"]

SHOW_ALL = "all"
SHOW_DIT_DIFFERS = "dit_differs"  # DIT and the reference have different words at the occurrence
SHOW_MODELS_DIFFER = "models_differ"  # DIT and DAT transcribed the occurrence differently


def _result(words: pd.Series, searched_word: str, missing: str) -> pd.Series:
    result = pd.Series(VARIANT, index=words.index)
    result[words == searched_word] = EXACT_MATCH
    result[words.isna()] = missing
    return result


def occurrence_key(anchor: Anchor) -> list[str]:
    """Columns that identify one occurrence of a word searched from `anchor`."""
    if anchor not in _OCCURRENCE_KEYS:
        raise ValueError(f"anchor must be one of {sorted(_OCCURRENCE_KEYS)}; got {anchor!r}")
    return _OCCURRENCE_KEYS[anchor]


def build_occurrence_table(word_rows: pd.DataFrame, word: str, anchor: Anchor = "reference") -> pd.DataFrame:
    """Pair each occurrence's alignments and attach its sentence metadata.

    anchor="reference": `word_rows` holds the REF -> DIT and REF -> DAT alignments of reference word `word`.
    Returns one row per (dataset, path, reference_index) with `dit_word`/`dat_word` (deletions as
    `DELETION_LABEL`), `dit_result`/`dat_result` (exact match / variant / deletion) and both similarities.

    anchor="dit": `word_rows` holds the REF -> DIT and DAT -> DIT alignments whose DIT word is `word`.
    Returns one row per (dataset, path, hypothesis_index) with `ref_word`/`dat_word` (no word on that side,
    i.e. DIT inserted the word, as `INSERTION_LABEL`), `ref_result`/`dat_result` (exact match / variant /
    insertion) and both similarities.

    Either way the table has `ref_word`, `dit_word` and `dat_word` (the searched word fills its own column) and
    the sentence columns that are present.
    """
    key = occurrence_key(anchor)
    if anchor == "reference":
        # The searched word is the reference word; the other transcripts are in `hypothesis_word`.
        sides = {"dit": DIT_MODEL, "dat": DAT_MODEL}
        other_column, own_column, missing, missing_label = "hypothesis_word", "ref_word", DELETION, DELETION_LABEL
    else:
        # The searched word is the DIT word; the reference (REF -> DIT) and DAT (DAT -> DIT) are in
        # `reference_word`, the source side of those alignments.
        sides = {"ref": DIT_MODEL, "dat": DAT_DIT_MODEL}
        other_column, own_column, missing, missing_label = "reference_word", "dit_word", INSERTION, INSERTION_LABEL

    def model_side(model: str, prefix: str) -> pd.DataFrame:
        side = word_rows[word_rows["model"] == model][key + [other_column, "similarity"]]
        side = side.assign(**{f"{prefix}_result": _result(side[other_column], word, missing)})
        return side.rename(columns={other_column: f"{prefix}_word", "similarity": f"{prefix}_similarity"})

    rows = word_rows.assign(dataset=word_rows["dataset"].astype(str))
    (first, first_model), (second, second_model) = sides.items()
    occurrences = model_side(first_model, first).merge(
        model_side(second_model, second), on=key, how="outer",
    ) if not word_rows.empty else pd.DataFrame(columns=key)
    occurrences = occurrences.assign(dataset=occurrences["dataset"].astype(str))

    sentence_columns = [c for c in _SENTENCE_COLUMNS if c in rows.columns]
    sentences = rows.drop_duplicates(["dataset", "path"])[["dataset", "path"] + sentence_columns]
    table = occurrences.merge(sentences, on=["dataset", "path"], how="left")
    if "dialect_region" in table:
        table["dialect_region"] = table["dialect_region"].astype(str)

    for prefix in sides:
        word_column = f"{prefix}_word"
        if word_column in table:
            table[word_column] = table[word_column].fillna(missing_label)
    table[own_column] = word
    return table.sort_values(key).reset_index(drop=True)


def filter_occurrences(table: pd.DataFrame, search: str = "", dit_words: Collection[str] = (),
                       dat_words: Collection[str] = (), regions: Collection[str] = (),
                       show: str = SHOW_ALL, ref_words: Collection[str] = ()) -> pd.DataFrame:
    """Rows of `table` matching every given filter; empty filters keep everything.

    `search` is a case-insensitive substring of the reference sentence, either transcript or any of the three
    words of the occurrence.
    """
    keep = pd.Series(True, index=table.index)
    if search.strip():
        needle = search.strip().casefold()
        searched = [c for c in ("reference", "dit_hypothesis", "dat_hypothesis", "ref_word", "dit_word", "dat_word")
                    if c in table]
        keep &= table[searched].fillna("").astype(str).apply(
            lambda column: column.str.casefold().str.contains(needle, regex=False)).any(axis=1)
    if ref_words:
        keep &= table["ref_word"].isin(ref_words)
    if dit_words:
        keep &= table["dit_word"].isin(dit_words)
    if dat_words:
        keep &= table["dat_word"].isin(dat_words)
    if regions:
        keep &= table["dialect_region"].isin(regions)
    if show == SHOW_DIT_DIFFERS:
        keep &= table["dit_word"] != table["ref_word"]
    elif show == SHOW_MODELS_DIFFER:
        keep &= table["dit_word"] != table["dat_word"]
    elif show != SHOW_ALL:
        raise ValueError(f"Unknown show filter: {show}")
    return table[keep]


def words_by_frequency(table: pd.DataFrame, column: str) -> list[str]:
    """Distinct values of `column`, most frequent first (ties alphabetically), for filter options."""
    counts = table[column].value_counts()
    return sorted(counts.index, key=lambda word: (-counts[word], word))
