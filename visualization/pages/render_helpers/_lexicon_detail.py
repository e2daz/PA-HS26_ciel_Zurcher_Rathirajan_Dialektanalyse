"""Detail view for the Dialect Word Lexicon page."""
import html
import io
import sys
from collections.abc import Callable
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

_VIS_DIR = Path(__file__).resolve().parents[2]
_REPO_ROOT = _VIS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "domain"))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "utils"))
sys.path.insert(0, str(_VIS_DIR))
from _auth import audio_unlocked, render_audio_gate  # noqa: E402
from _data import (  # noqa: E402
    ALPHA, LAMBDA, REGIONS, USE_GLOBAL_LEXICAL_NORMALIZATION,
    USE_SQUARED_POSITIONAL, resolve_audio_path, tfidf_matrix_pairs,
)
from pages.render_helpers import _levenshtein_graph  # noqa: E402
from bipartite_matching import build_full_bipartite_graph, solve_matching  # noqa: E402
from lexicon_graph import build_levenshtein_ring_graph  # noqa: E402
from plot_helpers import plot_reduced_bipartite_graph_with_matching  # noqa: E402
from preprocessing import tokenize  # noqa: E402
from word_similarity_calculator import WordSimilarityCalculator  # noqa: E402

_HYPOTHESIS_TABLE_COLUMN_CONFIG = {
    "hypothesis_word": st.column_config.TextColumn(
        help="The hypothesis variant the model produced as the alignment of the searched reference word.",
    ),
    "count": st.column_config.NumberColumn(
        help="Number of alignments where the searched reference word was matched with this hypothesis word.",
    ),
    "mean sim": st.column_config.NumberColumn(
        help="Mean similarity per hypothesis word: average similarity across the **count** alignments where the searched "
             "reference word was matched with this specific hypothesis variant. Higher = the variant is lexically and "
             "positionally close to the reference.",
    ),
    "TF-IDF": st.column_config.NumberColumn(
        help="Highest TF-IDF of the (ref, DIT-hyp) pair across the selected "
             "regions. Matches the word-cloud metric.",
        format="%.5f",
    ),
}

_GRAPH_RANK_LABELS = {
    "tfidf": "TF-IDF",
    "count": "occurrence count",
    "similarity": "mean similarity",
}

_LABEL_STYLE = (
    "padding: 4px 12px; font-weight: 600; text-align: center; color: #888; "
    "white-space: nowrap; border-bottom: 1px solid #eee;"
)


@st.cache_data(max_entries=64, show_spinner=True)
def _reduced_graph_png(reference: str, hypothesis: str) -> bytes | None:
    """Run the bipartite solver and return the rendered matching as PNG bytes."""
    ref_words = tokenize(reference)
    hyp_words = tokenize(hypothesis)

    if not ref_words and not hyp_words:
        return None

    calc = WordSimilarityCalculator(
        sent_len=max(len(ref_words), len(hyp_words)),
        alpha=ALPHA, lambda_=LAMBDA,
        use_global_lexical_normalization=USE_GLOBAL_LEXICAL_NORMALIZATION,
        use_squared_positional=USE_SQUARED_POSITIONAL,
    )
    G = build_full_bipartite_graph(ref_words, hyp_words, calc)
    matching = solve_matching(G)

    fig = plot_reduced_bipartite_graph_with_matching(G, matching)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)

    return buf.getvalue()


def render_header(word: str, word_rows: pd.DataFrame, selected_regions: list[str]) -> None:
    """Back button, word title, and total alignment-row / sentence count caption."""
    st.button("<- Back to word cloud", on_click=_back_to_cloud, key="back_btn")
    st.markdown(f"### `{word}`")

    n_total = len(word_rows)
    n_sentences = word_rows["path"].nunique()

    st.caption(f"{n_total:,} alignment rows across {n_sentences:,} sentences "
               f"(in {len(selected_regions)} selected region(s))")


def render_hypothesis_tables(word: str, word_rows: pd.DataFrame, selected_regions: list[str],
                             include_preterite: bool, dataset: str) -> None:
    """Side-by-side DIT/DAT hypothesis-variant tables for the searched reference word."""
    col_dit, col_dat = st.columns(2)

    with col_dit:
        st.markdown("**Dialect-Ignorant-Transcript (DIT, Whisper-large-v2)**")
        st.caption("Hypothesis variants the DIT model produced as the alignment of the searched reference word.")

        dit_table = _hypothesis_table(
            word_rows[word_rows["model"] == "dialect-ignorant"],
            word, "dialect-ignorant", selected_regions, include_preterite, dataset,
        )

        st.dataframe(dit_table, width="stretch", hide_index=True,
                     column_config=_HYPOTHESIS_TABLE_COLUMN_CONFIG)
        st.caption("Default ordering: descending by **TF-IDF**.")

    with col_dat:
        st.markdown("**Dialect-Aware-Transcript (DAT, FHNW STT4SG)**")
        st.caption("Hypothesis variants the DAT model produced as the alignment of the searched reference word.")

        dat_table = _hypothesis_table(
            word_rows[word_rows["model"] == "dialect-aware"],
            word, "dialect-aware", selected_regions, include_preterite, dataset,
        )

        st.dataframe(dat_table, width="stretch", hide_index=True,
                     column_config=_HYPOTHESIS_TABLE_COLUMN_CONFIG)
        st.caption("Default ordering: descending by **count**.")


def render_word_chart(word_rows: pd.DataFrame) -> None:
    """Stacked bar of regional DIT-variant breakdown for the searched reference word."""
    region_order = [r for r in REGIONS if r in word_rows["dialect_region"].unique()]
    if not region_order:
        return

    dit = word_rows[word_rows["model"] == "dialect-ignorant"].copy()
    dit["variant"] = dit["hypothesis_word"].fillna("(deletion)")

    top_n = 6
    top_variants = dit["variant"].value_counts().head(top_n).index.tolist()
    dit["variant_grouped"] = dit["variant"].where(dit["variant"].isin(top_variants), "(other)")

    counts = (
        dit.groupby(["dialect_region", "variant_grouped"], observed=True)
        .size()
        .reset_index(name="count")
    )

    variant_chart = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("dialect_region:N", sort=region_order, title=None,
                    axis=alt.Axis(labelAngle=0, labelOverlap=False)),
            y=alt.Y("count:Q", title="DIT alignments"),
            color=alt.Color("variant_grouped:N",
                            scale=alt.Scale(scheme="category10"),
                            legend=alt.Legend(title="DIT variant")),
            tooltip=[
                alt.Tooltip("dialect_region:N", title="Region"),
                alt.Tooltip("variant_grouped:N", title="Variant"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )
        .properties(height=340, title="DIT variants per region")
    )
    st.altair_chart(variant_chart, width="stretch")


def render_levenshtein_graph(word: str, word_rows: pd.DataFrame, selected_regions: list[str],
                             include_preterite: bool, dataset: str) -> None:
    """Render the top DIT variants on concentric rings by edit distance."""
    dit_rows = word_rows[word_rows["model"] == "dialect-ignorant"]
    lookup = _dit_tfidf_lookup(word, selected_regions, include_preterite, dataset)
    tfidf_by_hypothesis = {hyp: lookup(hyp) for hyp in dit_rows["hypothesis_word"].dropna().unique()}

    st.markdown("### Levenshtein ring graph")
    col_rank, col_top_n = st.columns(2)
    with col_rank:
        rank_by = st.selectbox(
            "Select variants by",
            options=list(_GRAPH_RANK_LABELS),
            format_func=_GRAPH_RANK_LABELS.get,
            key="lev_graph_rank_by",
            help="Which DIT variants make it into the graph: highest TF-IDF, most alignment "
                 "occurrences, or highest mean alignment similarity to the reference.",
        )
    with col_top_n:
        top_n = st.slider("Number of variants", min_value=5, max_value=100, value=10, step=5,
                          key="lev_graph_top_n",
                          help="Edges grow quadratically; use the edge-distance slider to thin them out.")

    graph = build_levenshtein_ring_graph(dit_rows, word, top_n=top_n, rank_by=rank_by,
                                         tfidf_by_hypothesis=tfidf_by_hypothesis)
    st.caption(
        f"Top {graph['shown_count']} of {graph['candidate_count']} DIT variants by "
        f"{_GRAPH_RANK_LABELS[rank_by]}. "
        "The reference word is centered; each ring shows the raw Levenshtein distance "
        "to the reference. Node size represents alignment count."
    )
    _levenshtein_graph.render(graph, dark=st.context.theme.type == "dark")


def render_regional_distribution(word: str, word_rows: pd.DataFrame,
                                 selected_regions: list[str]) -> None:
    """Heatmap of where each DIT variant of the searched word occurs across the dialect regions.
    The exact match (hypothesis == reference) is dropped so the regional variation is visible."""
    dit_rows = word_rows[
        (word_rows["model"] == "dialect-ignorant")
        & word_rows["hypothesis_word"].notna()
        & (word_rows["hypothesis_word"] != word)
    ]
    if dit_rows.empty:
        st.markdown("### Regional distribution")
        st.caption("No DIT variants to show for the selected regions.")
        return

    counts = (
        dit_rows.groupby(["hypothesis_word", "dialect_region"], observed=True)
        .size().reset_index(name="count")
    )
    # `dialect_region` is categorical; make it a plain string so the share division below works.
    counts["dialect_region"] = counts["dialect_region"].astype(str)

    st.markdown("### Regional distribution")
    col_values, col_top_n = st.columns(2)
    with col_values:
        normalize = st.radio(
            "Values",
            options=["Absolute count", "Share within region (%)"],
            horizontal=True,
            key="region_dist_values",
            help="Regions differ in sample size. 'Share within region' normalizes each region to "
                 "100 %, so you compare which variant a region prefers instead of raw counts.",
        )
    with col_top_n:
        top_n = st.slider("Number of variants", min_value=5, max_value=30, value=12, step=1,
                          key="region_dist_top_n",
                          help="The most frequent DIT variants of the searched word.")

    # Keep the most frequent variants so the heatmap stays readable.
    top_variants = (
        counts.groupby("hypothesis_word")["count"].sum()
        .sort_values(ascending=False).head(top_n).index.tolist()
    )
    shown = counts[counts["hypothesis_word"].isin(top_variants)].copy()

    if normalize.startswith("Share"):
        # Denominator = all DIT variants in the region, so columns are comparable across regions.
        region_totals = counts.groupby("dialect_region")["count"].sum()
        shown["value"] = (shown["count"] / shown["dialect_region"].map(region_totals) * 100).round(1)
        value_title, value_format = "Share %", ".1f"
    else:
        shown["value"] = shown["count"]
        value_title, value_format = "Count", "d"

    region_order = [r for r in selected_regions if r in set(shown["dialect_region"])]
    heatmap = (
        alt.Chart(shown)
        .mark_rect()
        .encode(
            x=alt.X("dialect_region:N", sort=region_order, title=None,
                    axis=alt.Axis(labelAngle=0, labelOverlap=False)),
            y=alt.Y("hypothesis_word:N", sort=top_variants, title="DIT variant"),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="blues"),
                            legend=alt.Legend(title=value_title)),
            tooltip=[
                alt.Tooltip("hypothesis_word:N", title="Variant"),
                alt.Tooltip("dialect_region:N", title="Region"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("value:Q", title=value_title, format=value_format),
            ],
        )
        .properties(height=min(460, 26 * len(top_variants) + 40))
    )
    st.altair_chart(heatmap, use_container_width=True)
    st.caption("Darker = more frequent. Absolute counts can mislead because regions have different "
               "sample sizes; switch to 'Share within region' to see which variant each region prefers.")


def render_example_sentences(df_view: pd.DataFrame, word_rows: pd.DataFrame, word: str,
                             selected_regions: list[str], include_preterite: bool,
                             dataset: str) -> None:
    """Sentences grouped by DIT hypothesis (TF-IDF desc).
    Two-level pagination: outer pager selects which variants to show; inner shows sentences.
    Matches and deletions sort to the bottom (TF-IDF = 0).
    """
    dit_rows = word_rows[word_rows["model"] == "dialect-ignorant"]

    # Dedupe on (path, _variant) so sentences with multiple occurrences of the searched word appear under every variant.
    unique_paths = (
        dit_rows
        .assign(
            _variant=lambda d: d["hypothesis_word"].fillna("(deletion)"),
            _region_idx=lambda d: d["dialect_region"].map(
                lambda r: REGIONS.index(r) if r in REGIONS else len(REGIONS)
            ),
        )
        .drop_duplicates(["path", "_variant"])
        [["path", "hypothesis_word", "_variant", "_region_idx",
          "dialect_region", "gender", "age",
          "reference", "dat_hypothesis", "dit_hypothesis"]]
    )

    dit_table = _hypothesis_table(dit_rows, word, "dialect-ignorant", selected_regions, include_preterite, dataset)

    available_variants = set(unique_paths["_variant"].unique())
    variant_order = [
        v for v in dit_table["hypothesis_word"].fillna("(deletion)").tolist()
        if v in available_variants
    ]

    n_unique_sentences = unique_paths["path"].nunique()
    st.markdown(f"**Sentences with word-level alignment**: {n_unique_sentences:,} sentences "
                f"across {len(variant_order)} DIT variants")

    # One-time audio password prompt (the only gated feature); rendered once here, where clips appear.
    render_audio_gate()

    # Outer paginator for variants: 10 per page.
    variant_page_size = 10
    n_variant_pages = max(1, (len(variant_order) + variant_page_size - 1) // variant_page_size)
    if n_variant_pages > 1:
        variant_page = st.selectbox(
            "Variant page",
            options=list(range(1, n_variant_pages + 1)),
            format_func=lambda p, n=n_variant_pages, total=len(variant_order): (
                f"Page {p} of {n} (variants {(p - 1) * variant_page_size + 1}"
                f"–{min(p * variant_page_size, total)})"
            ),
            key=f"variant_page_{word}",
            label_visibility="collapsed",
        )
    else:
        variant_page = 1
    variant_start = (variant_page - 1) * variant_page_size
    variants_on_page = variant_order[variant_start:variant_start + variant_page_size]

    rows_by_path = dict(tuple(
        df_view[df_view["path"].isin(set(unique_paths["path"]))]
        .groupby("path", sort=False)
    ))

    page_size = 15
    for i, variant in enumerate(variants_on_page):
        variant_paths = (
            unique_paths[unique_paths["_variant"] == variant]
            .sort_values(["_region_idx", "path"])
            .reset_index(drop=True)
        )
        n_variant = len(variant_paths)
        variant_idx = variant_start + i

        with st.expander(f"**DIT: `{variant}`** ({n_variant})", expanded=True):
            n_pages = max(1, (n_variant + page_size - 1) // page_size)
            if n_pages > 1:
                page = st.selectbox(
                    "Page",
                    options=list(range(1, n_pages + 1)),
                    format_func=lambda p, n=n_pages, r=n_variant: (
                        f"Page {p} of {n} (sentences {(p - 1) * page_size + 1}–{min(p * page_size, r)})"
                    ),
                    key=f"page_{word}_{variant_idx}",
                    label_visibility="collapsed",
                )
            else:
                page = 1
            start = (page - 1) * page_size
            page_paths = variant_paths.iloc[start:start + page_size]

            for _, row in page_paths.iterrows():
                _render_example_sentence_expander(row, rows_by_path[row["path"]], word, dataset)


def _dit_tfidf_lookup(ref_word: str, selected_regions: list[str], include_preterite: bool,
                      dataset: str) -> Callable[[str], float]:
    """Return hyp -> highest TF-IDF of the (ref, DIT-hyp) pair across the selected regions."""
    matrix, _vocab, word_to_idx, region_order = tfidf_matrix_pairs(include_preterite, "ref_dit", dataset)
    selected_idx = [i for i, r in enumerate(region_order) if r in selected_regions]

    def lookup(hyp):
        if pd.isna(hyp) or hyp == ref_word or not selected_idx:
            return 0.0
        col = word_to_idx.get(f"{ref_word}+{hyp}")
        return float(matrix[selected_idx, col].max()) if col is not None else 0.0

    return lookup


def _hypothesis_table(slice_df: pd.DataFrame, ref_word: str, model: str,
                      selected_regions: list[str], include_preterite: bool,
                      dataset: str) -> pd.DataFrame:
    """Hypothesis variants with count, per-variant mean similarity & TF-IDF score. Sorted by TF-IDF.
    TF-IDF only for DIT-variants, not DAT-variants."""
    out = (
        slice_df.groupby("hypothesis_word", dropna=False)
        .agg(count=("path", "size"), mean_sim=("similarity", "mean"))
        .reset_index()
    )
    out["mean sim"] = out["mean_sim"].round(3)

    if model == "dialect-aware":
        return (
            out[["hypothesis_word", "count", "mean sim"]]
            .sort_values("count", ascending=False)
        )

    lookup = _dit_tfidf_lookup(ref_word, selected_regions, include_preterite, dataset)
    out["TF-IDF"] = out["hypothesis_word"].apply(lookup).round(5)
    return (
        out[["hypothesis_word", "count", "mean sim", "TF-IDF"]]
        .sort_values("TF-IDF", ascending=False)
    )


def _alignment_columns(rows: pd.DataFrame) -> list[dict]:
    """Convert per-(path, model) alignment rows into ordered display columns.

    Substitutions and deletions are placed in reference-index order; insertions
    are interleaved by their hypothesis index relative to surrounding substitutions.
    """
    records = rows.to_dict("records")
    substitutions_deletions = sorted(
        [r for r in records if pd.notna(r["reference_index"])],
        key=lambda r: r["reference_index"],
    )
    insertions = sorted(
        [r for r in records if pd.isna(r["reference_index"])],
        key=lambda r: r["hypothesis_index"],
    )

    columns: list[dict] = []
    insertion_iter = iter(insertions)
    next_insertion = next(insertion_iter, None)

    for row in substitutions_deletions:
        if pd.isna(row["hypothesis_index"]):
            columns.append({"ref": row["reference_word"], "hyp": "ε", "kind": "deletion"})
            continue
        while next_insertion is not None and next_insertion["hypothesis_index"] < row["hypothesis_index"]:
            columns.append({"ref": "ε", "hyp": next_insertion["hypothesis_word"], "kind": "insertion"})
            next_insertion = next(insertion_iter, None)
        kind = "match" if row["reference_word"] == row["hypothesis_word"] else "substitution"
        columns.append({"ref": row["reference_word"], "hyp": row["hypothesis_word"], "kind": kind})

    while next_insertion is not None:
        columns.append({"ref": "ε", "hyp": next_insertion["hypothesis_word"], "kind": "insertion"})
        next_insertion = next(insertion_iter, None)

    return columns


def _render_alignment_html(columns: list[dict], hyp_label: str,
                           searched_word: str | None = None,
                           ref_label: str = "Reference") -> str:
    """Build a 2-row HTML alignment table; highlights the column whose ref word == searched_word."""
    ref_cells = [f'<td style="{_LABEL_STYLE}">{html.escape(ref_label)}</td>']
    hyp_cells = [f'<td style="{_LABEL_STYLE}">{html.escape(hyp_label)}</td>']
    for col in columns:
        is_searched = searched_word is not None and col["ref"] == searched_word
        bg = " background-color: rgba(247, 181, 0, 0.4);" if is_searched else ""
        base = f"padding: 4px 10px;{bg} border-bottom: 1px solid #eee;"
        ref_style = base + (" color: #999; font-style: italic;" if col["kind"] == "insertion" else "")
        hyp_style = base + (" color: #999; font-style: italic;" if col["kind"] == "deletion" else "")
        ref_cells.append(f'<td style="{ref_style}">{html.escape(str(col["ref"]))}</td>')
        hyp_cells.append(f'<td style="{hyp_style}">{html.escape(str(col["hyp"]))}</td>')

    table_style = "border-collapse: collapse; font-family: ui-monospace, monospace; font-size: 0.95em;"
    return (
        f'<div style="overflow-x: auto; margin-bottom: 6px;">'
        f'<table style="{table_style}">'
        f'<tr>{"".join(ref_cells)}</tr>'
        f'<tr>{"".join(hyp_cells)}</tr>'
        f'</table></div>'
    )


def _back_to_cloud():
    """Clear the search selection so the next render returns to the word-cloud overview."""
    st.session_state["selected_word"] = ""


def _render_example_sentence_expander(row: pd.Series, sentence_rows: pd.DataFrame, word: str, dataset: str) -> None:
    """One sentence expander: region + clip metadata, audio player, reference + hypotheses, alignment HTML,
    and an optional technical alignment graph."""
    path = row["path"]
    # SDS-200 demographics are sparse; only show gender/age when present (else they render as "nan").
    demographics = " · ".join(str(v) for v in (row["gender"], row["age"]) if pd.notna(v))
    header = f"**{row['dialect_region']}**" + (f" · {demographics}" if demographics else "") + f" · …{path[-12:]}"

    with st.expander(header):
        st.markdown(f"**Clip ID:** `{path}`")

        audio_file = resolve_audio_path(path, dataset)
        if audio_file is None:
            st.caption(f"Audio file not found locally: `{path}`")
        elif not audio_unlocked():
            # Audio is the only gated feature: without the session unlock we never call st.audio,
            # so no media URL is minted and the clips can't be crawled.
            st.caption("Audio locked, enter the password above to enable playback.")
        else:
            audio_key = f"audio_loaded_{path}"
            loaded = st.session_state.get(audio_key, False)
            if not loaded and st.button("▶ Load audio", key=f"load_{path}"):
                st.session_state[audio_key] = True
                loaded = True
            if loaded:
                st.audio(str(audio_file))

        st.markdown(
            f"**Reference:** {row['reference']}  \n"
            f"**Hypothesis DIT:** {row['dit_hypothesis']}  \n"
            f"**Hypothesis DAT:** {row['dat_hypothesis']}"
        )

        dat_cols = _alignment_columns(sentence_rows[sentence_rows["model"] == "dialect-aware"])
        dit_cols = _alignment_columns(sentence_rows[sentence_rows["model"] == "dialect-ignorant"])
        dat_dit_cols = _alignment_columns(sentence_rows[sentence_rows["model"] == "dat-dit"])

        st.markdown(_render_alignment_html(dit_cols, "Hypothesis DIT", searched_word=word),
                    unsafe_allow_html=True)
        st.markdown(_render_alignment_html(dat_cols, "Hypothesis DAT", searched_word=word),
                    unsafe_allow_html=True)
        st.markdown(_render_alignment_html(dat_dit_cols, "Hypothesis DIT",
                                           searched_word=None, ref_label="Hypothesis DAT"),
                    unsafe_allow_html=True)

        if st.checkbox("Show technical alignment", key=f"graph_{path}"):
            png_dit = _reduced_graph_png(row["reference"], row["dit_hypothesis"])

            if png_dit is not None:
                st.markdown("**REF → DIT: reduced bipartite matching**")
                st.image(png_dit)
            png_dat = _reduced_graph_png(row["reference"], row["dat_hypothesis"])

            if png_dat is not None:
                st.markdown("**REF → DAT: reduced bipartite matching**")
                st.image(png_dat)
            png_dat_dit = _reduced_graph_png(row["dat_hypothesis"], row["dit_hypothesis"])

            if png_dat_dit is not None:
                st.markdown("**DAT → DIT: reduced bipartite matching**")
                st.image(png_dat_dit)
