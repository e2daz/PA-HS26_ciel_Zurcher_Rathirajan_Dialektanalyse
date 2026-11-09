"""Summary figures for how one model transcribed a reference word (DIAL-27)."""

from dataclasses import dataclass

import pandas as pd

EXACT_MATCH = "exact match"
VARIANT = "variant"
DELETION = "deletion"
DELETION_LABEL = "(deleted)"


@dataclass(frozen=True)
class ModelSummary:
    """Counts over one model's alignments of a reference word; each alignment row is one occurrence."""
    occurrences: int
    exact_matches: int
    deletions: int
    variant_count: int  # distinct transcriptions other than the reference word, deletions excluded

    @property
    def exact_match_share(self) -> float:
        return self.exact_matches / self.occurrences if self.occurrences else 0.0


def summarize_model(rows: pd.DataFrame, reference_word: str) -> ModelSummary:
    """Summarize `rows`: one model's alignments of `reference_word` (a missing hypothesis is a deletion)."""
    hypotheses = rows["hypothesis_word"]
    is_deletion = hypotheses.isna()
    is_exact = hypotheses == reference_word
    return ModelSummary(
        occurrences=len(rows),
        exact_matches=int(is_exact.sum()),
        deletions=int(is_deletion.sum()),
        variant_count=int(hypotheses[~is_deletion & ~is_exact].nunique()),
    )


def variant_frequencies(rows: pd.DataFrame, reference_word: str) -> pd.DataFrame:
    """One row per transcription of `reference_word`, most frequent first.

    Columns: `transcription` (deletions as `DELETION_LABEL`), `kind` (exact match / variant / deletion),
    `count`, `share` (percent of all occurrences) and `mean_similarity`. Ties are ordered alphabetically
    so the table is stable.
    """
    columns = ["transcription", "kind", "count", "share", "mean_similarity"]
    if rows.empty:
        return pd.DataFrame(columns=columns)

    hypotheses = rows["hypothesis_word"]
    kind = pd.Series(VARIANT, index=rows.index)
    kind[hypotheses == reference_word] = EXACT_MATCH
    kind[hypotheses.isna()] = DELETION

    out = (
        rows.assign(transcription=hypotheses.fillna(DELETION_LABEL), kind=kind)
        .groupby(["transcription", "kind"], as_index=False)
        .agg(count=("hypothesis_word", "size"), mean_similarity=("similarity", "mean"))
    )
    out["share"] = out["count"] / len(rows) * 100
    return (
        out.sort_values(["count", "transcription"], ascending=[False, True])
        .reset_index(drop=True)[columns]
    )
