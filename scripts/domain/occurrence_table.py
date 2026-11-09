"""One row per occurrence of a reference word, with what DIT and DAT made of it (DIAL-29)."""

from collections.abc import Collection

import pandas as pd

DIT_MODEL = "dialect-ignorant"
DAT_MODEL = "dialect-aware"
DELETION_LABEL = "(deleted)"

EXACT_MATCH = "exact match"
VARIANT = "variant"
DELETION = "deletion"

# An occurrence is one reference token: a sentence (path) and the word's position in it.
_OCCURRENCE_KEY = ["dataset", "path", "reference_index"]
_SENTENCE_COLUMNS = ["dialect_region", "gender", "age", "reference", "dit_hypothesis", "dat_hypothesis"]

SHOW_ALL = "all"
SHOW_DIT_DIFFERS = "dit_differs"  # DIT did not transcribe the reference word itself
SHOW_MODELS_DIFFER = "models_differ"  # DIT and DAT transcribed the occurrence differently


def _result(hypotheses: pd.Series, reference_word: str) -> pd.Series:
    result = pd.Series(VARIANT, index=hypotheses.index)
    result[hypotheses == reference_word] = EXACT_MATCH
    result[hypotheses.isna()] = DELETION
    return result


def build_occurrence_table(word_rows: pd.DataFrame, reference_word: str) -> pd.DataFrame:
    """Pair each occurrence's DIT and DAT alignment and attach its sentence metadata.

    `word_rows` holds the REF -> DIT and REF -> DAT alignments of `reference_word` (column `model`).
    Returns one row per (dataset, path, reference_index) with `dit_word`/`dat_word` (deletions as
    `DELETION_LABEL`), `dit_result`/`dat_result` (exact match / variant / deletion), both similarities
    and the sentence columns that are present.
    """
    def model_side(model: str, prefix: str) -> pd.DataFrame:
        side = word_rows[word_rows["model"] == model][_OCCURRENCE_KEY + ["hypothesis_word", "similarity"]]
        side = side.assign(**{f"{prefix}_result": _result(side["hypothesis_word"], reference_word)})
        return side.rename(columns={"hypothesis_word": f"{prefix}_word", "similarity": f"{prefix}_similarity"})

    rows = word_rows.assign(dataset=word_rows["dataset"].astype(str))
    occurrences = model_side(DIT_MODEL, "dit").merge(
        model_side(DAT_MODEL, "dat"), on=_OCCURRENCE_KEY, how="outer",
    ) if not word_rows.empty else pd.DataFrame(columns=_OCCURRENCE_KEY)
    occurrences = occurrences.assign(dataset=occurrences["dataset"].astype(str))

    sentence_columns = [c for c in _SENTENCE_COLUMNS if c in rows.columns]
    sentences = rows.drop_duplicates(["dataset", "path"])[["dataset", "path"] + sentence_columns]
    table = occurrences.merge(sentences, on=["dataset", "path"], how="left")
    if "dialect_region" in table:
        table["dialect_region"] = table["dialect_region"].astype(str)

    for prefix in ("dit", "dat"):
        word_column = f"{prefix}_word"
        if word_column in table:
            table[word_column] = table[word_column].fillna(DELETION_LABEL)
    return table.sort_values(_OCCURRENCE_KEY).reset_index(drop=True)


def filter_occurrences(table: pd.DataFrame, search: str = "", dit_words: Collection[str] = (),
                       dat_words: Collection[str] = (), regions: Collection[str] = (),
                       show: str = SHOW_ALL) -> pd.DataFrame:
    """Rows of `table` matching every given filter; empty filters keep everything.

    `search` is a case-insensitive substring of the reference sentence, either transcript or either
    transcribed word.
    """
    keep = pd.Series(True, index=table.index)
    if search.strip():
        needle = search.strip().casefold()
        searched = [c for c in ("reference", "dit_hypothesis", "dat_hypothesis", "dit_word", "dat_word")
                    if c in table]
        keep &= table[searched].fillna("").astype(str).apply(
            lambda column: column.str.casefold().str.contains(needle, regex=False)).any(axis=1)
    if dit_words:
        keep &= table["dit_word"].isin(dit_words)
    if dat_words:
        keep &= table["dat_word"].isin(dat_words)
    if regions:
        keep &= table["dialect_region"].isin(regions)
    if show == SHOW_DIT_DIFFERS:
        keep &= table["dit_result"] != EXACT_MATCH
    elif show == SHOW_MODELS_DIFFER:
        keep &= table["dit_word"] != table["dat_word"]
    elif show != SHOW_ALL:
        raise ValueError(f"Unknown show filter: {show}")
    return table[keep]


def words_by_frequency(table: pd.DataFrame, column: str) -> list[str]:
    """Distinct values of `column`, most frequent first (ties alphabetically), for filter options."""
    counts = table[column].value_counts()
    return sorted(counts.index, key=lambda word: (-counts[word], word))
