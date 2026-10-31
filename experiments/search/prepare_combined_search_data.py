"""Prepare normalized sentence and word-occurrence data for combined search.

The existing alignment tables are kept as the source of truth for word
positions.  This script normalizes the two corpora into one sentence schema
and writes an occurrence-level index containing reference, DAT, and DIT words.

Run from the repository root:

    python experiments/search/prepare_combined_search_data.py

The generated Parquets are intended to be consumed by the search index/UI,
not committed as replacements for the alignment artifacts.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "domain"))

from preprocessing import clean_word  # noqa: E402

ALIGNMENT_DIR = PROJECT_ROOT / "experiments" / "analysis"
TRANSCRIPT_DIR = PROJECT_ROOT / "transcripts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "search"

DATASET_ORDER = ("STT4SG-350", "SDS-200")


@dataclass(frozen=True)
class DatasetPaths:
    """Input artifacts for one dataset."""

    name: str
    dat_alignment: Path
    dit_alignment: Path
    dat_tsv: Path
    dit_tsv: Path
    metadata_join_key: str


DATASETS = {
    "STT4SG-350": DatasetPaths(
        name="STT4SG-350",
        dat_alignment=ALIGNMENT_DIR / "stt4sg" / "train_all_alignments_dialect-aware.parquet",
        dit_alignment=ALIGNMENT_DIR / "stt4sg" / "train_all_alignments_dialect-ignorant.parquet",
        dat_tsv=TRANSCRIPT_DIR / "dialect-aware" / "fhnw" / "stt4sg" / "train_all_transcribed.tsv",
        dit_tsv=TRANSCRIPT_DIR / "dialect-ignorant" / "whisper-large-v2" / "stt4sg" / "train_all_enriched_transcribed_praet.tsv",
        metadata_join_key="path",
    ),
    "SDS-200": DatasetPaths(
        name="SDS-200",
        dat_alignment=ALIGNMENT_DIR / "sds-200" / "train_all_alignments_dialect-aware.parquet",
        dit_alignment=ALIGNMENT_DIR / "sds-200" / "train_all_alignments_dialect-ignorant.parquet",
        dat_tsv=TRANSCRIPT_DIR / "dialect-aware" / "fhnw" / "sds-200" / "train_all_transcribed.tsv",
        dit_tsv=TRANSCRIPT_DIR / "dialect-ignorant" / "whisper-large-v2" / "sds-200" / "export_20211220_enriched_transcribed_praet.tsv",
        metadata_join_key="clip_id",
    ),
}

SENTENCE_COLUMNS = [
    "dataset", "sentence_id", "path", "dialect_region", "client_id", "gender",
    "age", "canton", "reference", "dat_hypothesis", "dit_hypothesis",
    "is_praeteritum",
]

OCCURRENCE_COLUMNS = [
    "dataset", "sentence_id", "path", "dialect_region", "client_id", "gender",
    "age", "canton", "reference", "dat_hypothesis", "dit_hypothesis",
    "is_praeteritum", "word", "word_source", "model", "reference_word",
    "hypothesis_word", "reference_index", "hypothesis_index", "similarity",
]


def _clean_word(word: object) -> str:
    """Use the pipeline's token normalization without importing visualization."""
    return clean_word(word)


def _read_metadata(paths: DatasetPaths) -> pd.DataFrame:
    """Load both transcript TSVs into the common sentence-level schema."""
    if paths.metadata_join_key == "path":
        dat = pd.read_csv(paths.dat_tsv, sep="\t", encoding="utf-8-sig")[
            ["path", "dialect_region", "client_id", "gender", "age", "canton",
             "sentence", "fhnw_transcript"]
        ]
        dit = pd.read_csv(paths.dit_tsv, sep="\t", encoding="utf-8-sig")[
            ["path", "whisper_large_v2_transcript", "is_praeteritum"]
        ]
        merged = dat.merge(dit, on="path", how="inner", validate="one_to_one")
        merged["sentence_id"] = merged["path"]
    else:
        dat = pd.read_csv(paths.dat_tsv, sep="\t", encoding="utf-8-sig")[
            ["clip_id", "sentence", "fhnw_transcript", "client_id", "gender", "age", "canton"]
        ]
        dit = pd.read_csv(paths.dit_tsv, sep="\t", encoding="utf-8-sig")[
            ["clip_id", "path", "dialect_region", "whisper_large_v2_transcript",
             "is_praeteritum"]
        ]
        merged = dat.merge(dit, on="clip_id", how="inner", validate="one_to_one")
        merged["sentence_id"] = merged["clip_id"].astype(str)

    merged = merged.rename(columns={
        "sentence": "reference",
        "fhnw_transcript": "dat_hypothesis",
        "whisper_large_v2_transcript": "dit_hypothesis",
    })
    merged["dataset"] = paths.name
    return merged[SENTENCE_COLUMNS].drop_duplicates(
        ["dataset", "sentence_id"], keep="first"
    )


def _read_alignment(path: Path, model: str) -> pd.DataFrame:
    """Load an alignment table and attach the model represented by its hypothesis."""
    required = (
        "path", "reference_word", "hypothesis_word", "reference_index",
        "hypothesis_index", "similarity",
    )
    alignment = pd.read_parquet(path)
    missing = set(required).difference(alignment.columns)
    if missing:
        raise ValueError(f"{path} is missing alignment columns: {sorted(missing)}")
    return alignment[list(required)].assign(model=model)


def _occurrences(
    sentences: pd.DataFrame,
    dat_alignment: pd.DataFrame,
    dit_alignment: pd.DataFrame,
) -> pd.DataFrame:
    """Create one searchable row per reference, DAT, or DIT word occurrence."""
    metadata = sentences.set_index("path", drop=False)
    frames: list[pd.DataFrame] = []

    for alignment, source, model in (
        (dat_alignment, "reference", "dialect-aware"),
        (dat_alignment, "DAT", "dialect-aware"),
        (dit_alignment, "DIT", "dialect-ignorant"),
    ):
        rows = alignment.copy()
        if source == "reference":
            rows = rows[rows["reference_word"].notna()].copy()
            rows["word"] = rows["reference_word"].map(_clean_word)
        else:
            rows = rows[rows["hypothesis_word"].notna()].copy()
            rows["word"] = rows["hypothesis_word"].map(_clean_word)

        rows["word_source"] = source
        rows["model"] = model
        rows = rows[rows["word"] != ""]
        frames.append(rows)

    occurrences = pd.concat(frames, ignore_index=True)
    occurrence_metadata = metadata.reindex(occurrences["path"]).reset_index(drop=True)
    occurrences = occurrences.reset_index(drop=True)
    for column in SENTENCE_COLUMNS:
        occurrences[column] = occurrence_metadata[column]

    return occurrences[OCCURRENCE_COLUMNS].sort_values(
        ["dataset", "sentence_id", "word_source", "reference_index", "hypothesis_index"],
        na_position="last",
    ).reset_index(drop=True)


def prepare_dataset(paths: DatasetPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare one dataset and return its normalized sentences and occurrences."""
    sentences = _read_metadata(paths)
    dat_alignment = _read_alignment(paths.dat_alignment, "dialect-aware")
    dit_alignment = _read_alignment(paths.dit_alignment, "dialect-ignorant")

    dat_paths = set(dat_alignment["path"])
    dit_paths = set(dit_alignment["path"])
    sentence_paths = set(sentences["path"])
    missing = (dat_paths | dit_paths) - sentence_paths
    if missing:
        raise ValueError(
            f"{paths.name}: {len(missing):,} alignment paths have no metadata row"
        )

    return sentences, _occurrences(sentences, dat_alignment, dit_alignment)


def prepare_combined_data(
    selected_datasets: tuple[str, ...] = DATASET_ORDER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare and concatenate selected datasets in a deterministic order."""
    unknown = set(selected_datasets) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown dataset(s): {sorted(unknown)}")

    sentence_frames: list[pd.DataFrame] = []
    occurrence_frames: list[pd.DataFrame] = []
    for name in selected_datasets:
        sentences, occurrences = prepare_dataset(DATASETS[name])
        sentence_frames.append(sentences)
        occurrence_frames.append(occurrences)

    sentences = pd.concat(sentence_frames, ignore_index=True)
    occurrences = pd.concat(occurrence_frames, ignore_index=True)
    sentences = sentences.sort_values(["dataset", "sentence_id"]).reset_index(drop=True)
    occurrences = occurrences.sort_values(
        ["dataset", "sentence_id", "word_source", "reference_index", "hypothesis_index"],
        na_position="last",
    ).reset_index(drop=True)
    return sentences, occurrences


def write_outputs(
    sentences: pd.DataFrame,
    occurrences: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write compressed Parquets with stable schemas and no pandas index."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sentences.to_parquet(
        output_dir / "combined_sentences.parquet",
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    occurrences.to_parquet(
        output_dir / "word_occurrences.parquet",
        engine="pyarrow",
        compression="zstd",
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for combined_sentences.parquet and word_occurrences.parquet.",
    )
    args = parser.parse_args()

    sentences, occurrences = prepare_combined_data()
    write_outputs(sentences, occurrences, args.output_dir)
    print(f"Wrote {len(sentences):,} sentences to {args.output_dir / 'combined_sentences.parquet'}")
    print(f"Wrote {len(occurrences):,} word occurrences to {args.output_dir / 'word_occurrences.parquet'}")
    print("Word sources: " + ", ".join(sorted(occurrences["word_source"].unique())))


if __name__ == "__main__":
    main()
