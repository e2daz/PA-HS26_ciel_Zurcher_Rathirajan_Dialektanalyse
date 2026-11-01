"""Data preparation for interactive reference-word lexicon graphs."""

from itertools import combinations
from typing import Any, Literal

import Levenshtein
import pandas as pd

RankBy = Literal["tfidf", "count", "similarity"]

# Primary sort key first; the remaining keys break ties deterministically.
_RANK_KEYS: dict[str, list[str]] = {
    "tfidf": ["tfidf", "count"],
    "count": ["count", "tfidf"],
    "similarity": ["mean_similarity", "count"],
}


def build_levenshtein_ring_graph(
    rows: pd.DataFrame,
    reference_word: str,
    *,
    top_n: int = 10,
    rank_by: RankBy = "tfidf",
    tfidf_by_hypothesis: dict[str, float] | None = None,
    max_edge_distance: int | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable ring graph from reference-word alignments.

    Candidates are ranked by `rank_by` ("tfidf", "count" = alignment occurrences,
    or "similarity" = mean alignment similarity to the reference); a secondary
    metric and finally the word itself break ties deterministically. Deletions
    and unchanged matches are intentionally omitted from the prototype graph.

    Pairwise edges grow quadratically with `top_n`; pass `max_edge_distance` to
    drop distant pairs server-side when rendering larger word sets.
    """
    required = {"hypothesis_word", "similarity"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"rows is missing required columns: {sorted(missing)}")
    if rank_by not in _RANK_KEYS:
        raise ValueError(f"rank_by must be one of {sorted(_RANK_KEYS)}; got {rank_by!r}")
    if top_n <= 0:
        raise ValueError(f"top_n must be positive; got {top_n}")
    if max_edge_distance is not None and max_edge_distance <= 0:
        raise ValueError(f"max_edge_distance must be positive; got {max_edge_distance}")

    candidates = rows.dropna(subset=["hypothesis_word"]).copy()
    candidates = candidates[candidates["hypothesis_word"] != reference_word]
    if candidates.empty:
        return {
            "reference": reference_word,
            "nodes": [{"id": "reference", "label": reference_word, "kind": "reference", "distance": 0}],
            "edges": [],
            "max_distance": 0,
            "max_edge_distance": 0,
            "shown_count": 0,
            "candidate_count": 0,
            "rank_by": rank_by,
        }

    grouped = (
        candidates.groupby("hypothesis_word", sort=False)
        .agg(count=("hypothesis_word", "size"), mean_similarity=("similarity", "mean"))
        .reset_index()
    )
    tfidf = tfidf_by_hypothesis or {}
    grouped["tfidf"] = grouped["hypothesis_word"].map(tfidf).fillna(0.0)
    candidate_count = len(grouped)
    grouped = grouped.sort_values(
        [*_RANK_KEYS[rank_by], "hypothesis_word"],
        ascending=[False] * len(_RANK_KEYS[rank_by]) + [True],
        kind="mergesort",
    ).head(top_n)

    words = grouped["hypothesis_word"].tolist()
    region_counts = {}
    if "dialect_region" in candidates.columns:
        region_counts = (
            candidates.groupby(["hypothesis_word", "dialect_region"], observed=True)
            .size()
            .unstack(fill_value=0)
            .to_dict(orient="index")
        )
    nodes = [{
        "id": "reference",
        "label": reference_word,
        "kind": "reference",
        "distance": 0,
    }]
    for row in grouped.to_dict("records"):
        word = str(row["hypothesis_word"])
        nodes.append({
            "id": f"hypothesis:{word}",
            "label": word,
            "kind": "hypothesis",
            "distance": int(Levenshtein.distance(reference_word, word)),
            "count": int(row["count"]),
            "mean_similarity": round(float(row["mean_similarity"]), 4),
            "tfidf": round(float(row["tfidf"]), 6),
            "region_counts": {
                str(region): int(count)
                for region, count in region_counts.get(word, {}).items()
                if count
            },
        })

    edges = []
    for left, right in combinations(words, 2):
        distance = int(Levenshtein.distance(left, right))
        if max_edge_distance is not None and distance > max_edge_distance:
            continue
        edges.append({
            "source": f"hypothesis:{left}",
            "target": f"hypothesis:{right}",
            "distance": distance,
        })

    return {
        "reference": reference_word,
        "nodes": nodes,
        "edges": edges,
        "max_distance": max((node["distance"] for node in nodes), default=0),
        "max_edge_distance": max((edge["distance"] for edge in edges), default=0),
        "shown_count": len(grouped),
        "candidate_count": candidate_count,
        "rank_by": rank_by,
    }
