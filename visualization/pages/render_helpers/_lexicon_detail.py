"""Detail view for the Dialect Word Lexicon page."""
import html
import io
import sys
from collections.abc import Callable
from dataclasses import dataclass
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
from occurrence_table import (  # noqa: E402
    SHOW_ALL, SHOW_DIT_DIFFERS, SHOW_MODELS_DIFFER, build_occurrence_table, filter_occurrences, words_by_frequency,
)
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

# All of these go to the graph at once, so the variant count can change there without a rerun.
_GRAPH_MAX_VARIANTS = 100

_GRAPH_RANK_LABELS = {
    "tfidf": "TF-IDF",
    "count": "occurrence count",
    "similarity": "mean similarity",
}

_VARIANT_TABLE_HEIGHT = 350

# Streamlit groups both step buttons right of the field; this matches the ring graph's "label - n +" control.
_REGION_VARIANTS_CSS = """
<style>
  .st-key-region_dist_top_n [data-testid="stNumberInput"] { display: flex; align-items: center; gap: 8px; }
  .st-key-region_dist_top_n label { margin: 0; width: auto; }
  .st-key-region_dist_top_n [data-testid="stNumberInputContainer"] {
    width: auto; height: auto; gap: 4px; align-items: center; background: transparent; border: none;
  }
  .st-key-region_dist_top_n [data-testid="stNumberInputContainer"] > div { display: contents; }
  .st-key-region_dist_top_n input, .st-key-region_dist_top_n button[data-testid^="stNumberInputStep"] {
    height: 28px; border: 1px solid rgba(128, 128, 128, 0.45); border-radius: 4px;
  }
  .st-key-region_dist_top_n input { flex: none; width: 3.2em; padding: 3px; text-align: center; }
  .st-key-region_dist_top_n button[data-testid^="stNumberInputStep"] { width: 30px; }
  .st-key-region_dist_top_n [data-testid="stNumberInputStepDown"] { order: -1; }
</style>
"""

_OCCURRENCE_VIEWS = {"table": "Table of all occurrences", "by_variant": "Sentences by DIT variant"}

_SHOW_OPTIONS = {
    SHOW_ALL: "All",
    SHOW_DIT_DIFFERS: "DIT ≠ reference",
    SHOW_MODELS_DIFFER: "DIT ≠ DAT",
}

# Also the table's column order; "dataset" is appended when both corpora are shown.
_OCCURRENCE_COLUMN_CONFIG = {
    "dialect_region": st.column_config.TextColumn("region"),
    "dit_word": st.column_config.TextColumn("DIT", help="What Whisper transcribed for this occurrence."),
    "dat_word": st.column_config.TextColumn("DAT", help="What the FHNW model transcribed for this occurrence."),
    "dit_similarity": st.column_config.NumberColumn("DIT sim", format="%.2f",
                                                    help="Alignment similarity of the DIT word to the reference."),
    "dat_similarity": st.column_config.NumberColumn("DAT sim", format="%.2f",
                                                    help="Alignment similarity of the DAT word to the reference."),
    "reference": st.column_config.TextColumn("reference sentence", width="large"),
    "dit_hypothesis": st.column_config.TextColumn("DIT transcript", width="large"),
    "dat_hypothesis": st.column_config.TextColumn("DAT transcript", width="large"),
    "position": st.column_config.NumberColumn("pos", help="Word position in the reference sentence (1 = first)."),
    "dataset": st.column_config.TextColumn("dataset"),
}

_LABEL_STYLE = (
    "padding: 4px 12px; font-weight: 600; text-align: center; color: #888; "
    "white-space: nowrap; border-bottom: 1px solid #eee;"
)


@dataclass(frozen=True)
class WordContext:
    """Everything an analysis tab needs to render one reference word."""
    word: str
    word_rows: pd.DataFrame  # alignments of the word itself, REF -> DIT and REF -> DAT
    sentence_rows: pd.DataFrame  # every alignment of the sentences containing the word
    regions: list[str]
    include_preterite: bool
    dataset: str


def _clamp_region_variants() -> None:
    """Snap the Regions variant count into range, like the ring graph's controls."""
    st.session_state["region_dist_top_n"] = min(30, max(5, st.session_state["region_dist_top_n"]))


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


def render_overview(ctx: WordContext) -> None:
    """General figures for the word and its most frequent DIT and DAT variants."""
    dit_rows = ctx.word_rows[ctx.word_rows["model"] == "dialect-ignorant"]
    dat_rows = ctx.word_rows[ctx.word_rows["model"] == "dialect-aware"]

    def variant_count(rows: pd.DataFrame) -> int:
        hypotheses = rows["hypothesis_word"].dropna()
        return hypotheses[hypotheses != ctx.word].nunique()

    col_occurrences, col_sentences, col_dit, col_dat = st.columns(4)
    col_occurrences.metric("Occurrences", f"{len(dit_rows):,}",
                           help=f"How often the word occurs in the {len(ctx.regions)} selected region(s).")
    col_sentences.metric("Sentences", f"{ctx.word_rows['path'].nunique():,}")
    col_dit.metric("DIT variants", f"{variant_count(dit_rows):,}",
                   help="Different words Whisper produced for it, excluding the word itself.")
    col_dat.metric("DAT variants", f"{variant_count(dat_rows):,}",
                   help="Different words the FHNW model produced for it, excluding the word itself.")

    st.caption("What each model transcribed where the reference has this word, most frequent first.")
    col_dit_table, col_dat_table = st.columns(2)
    with col_dit_table:
        st.markdown("**Dialect-Ignorant-Transcript (DIT, Whisper-large-v2)**")
        st.dataframe(
            _hypothesis_table(dit_rows, ctx.word, "dialect-ignorant", ctx.regions, ctx.include_preterite, ctx.dataset)
            .sort_values("count", ascending=False),
            width="stretch", height=_VARIANT_TABLE_HEIGHT, hide_index=True,
            column_config=_HYPOTHESIS_TABLE_COLUMN_CONFIG,
        )
    with col_dat_table:
        st.markdown("**Dialect-Aware-Transcript (DAT, FHNW STT4SG)**")
        st.dataframe(
            _hypothesis_table(dat_rows, ctx.word, "dialect-aware", ctx.regions, ctx.include_preterite, ctx.dataset),
            width="stretch", height=_VARIANT_TABLE_HEIGHT, hide_index=True,
            column_config=_HYPOTHESIS_TABLE_COLUMN_CONFIG,
        )


def render_levenshtein_graph(ctx: WordContext) -> None:
    """Render the top DIT variants on concentric rings by edit distance."""
    word, word_rows = ctx.word, ctx.word_rows
    dit_rows = word_rows[word_rows["model"] == "dialect-ignorant"]
    lookup = _dit_tfidf_lookup(word, ctx.regions, ctx.include_preterite, ctx.dataset)
    tfidf_by_hypothesis = {hyp: lookup(hyp) for hyp in dit_rows["hypothesis_word"].dropna().unique()}

    col_rank, col_caption = st.columns(2, vertical_alignment="bottom")
    col_caption.caption(
        "The reference word is centered; each ring shows the raw Levenshtein distance "
        "to the reference. Node size represents alignment count."
    )
    with col_rank:
        rank_by = st.selectbox(
            "Select variants by",
            options=list(_GRAPH_RANK_LABELS),
            format_func=_GRAPH_RANK_LABELS.get,
            key="lev_graph_rank_by",
            help="Which DIT variants make it into the graph: highest TF-IDF, most alignment "
                 "occurrences, or highest mean alignment similarity to the reference.",
        )

    graph = build_levenshtein_ring_graph(dit_rows, word, top_n=_GRAPH_MAX_VARIANTS, rank_by=rank_by,
                                         tfidf_by_hypothesis=tfidf_by_hypothesis)
    _levenshtein_graph.render(graph, _GRAPH_RANK_LABELS[rank_by], dark=st.context.theme.type == "dark")


def render_regional_distribution(ctx: WordContext) -> None:
    """Heatmap of where the searched word and its DIT variants occur across the dialect regions.
    The exact match (hypothesis == reference) is the first row, so each variant can be compared
    against how often DIT simply transcribed the reference word."""
    word, word_rows, selected_regions = ctx.word, ctx.word_rows, ctx.regions
    dit_rows = word_rows[(word_rows["model"] == "dialect-ignorant") & word_rows["hypothesis_word"].notna()]
    if dit_rows.empty:
        st.caption("No DIT transcriptions to show for the selected regions.")
        return

    counts = (
        dit_rows.groupby(["hypothesis_word", "dialect_region"], observed=True)
        .size().reset_index(name="count")
    )
    # `dialect_region` is categorical; make it a plain string so the share division below works.
    counts["dialect_region"] = counts["dialect_region"].astype(str)

    col_values, col_top_n = st.columns(2)
    with col_values:
        normalize = st.radio(
            "Values",
            options=["Absolute count", "Share within region (%)"],
            horizontal=True,
            key="region_dist_values",
            help="Regions differ in sample size. 'Share within region' normalizes each region to "
                 "100 % of its DIT transcriptions of the word (exact match + variants), so you "
                 "compare what a region produces instead of raw counts.",
        )
    with col_top_n:
        st.html(_REGION_VARIANTS_CSS)
        # No min/max here: Streamlit would keep an out-of-range entry in the field and silently ignore it.
        top_n = st.number_input("Number of variants", value=12, step=1, key="region_dist_top_n",
                                on_change=_clamp_region_variants,
                                help="The most frequent DIT variants of the searched word, shown below "
                                     "the exact match.")

    # Exact match first, then the most frequent variants so the heatmap stays readable.
    variant_counts = counts[counts["hypothesis_word"] != word]
    top_variants = (
        variant_counts.groupby("hypothesis_word")["count"].sum()
        .sort_values(ascending=False).head(top_n).index.tolist()
    )
    rows = [word] + top_variants
    regions_present = [r for r in selected_regions if r in set(counts["dialect_region"])]

    # Complete the word x region grid so empty combinations show as 0 (lightest), not a dark gap.
    grid = pd.MultiIndex.from_product([rows, regions_present],
                                      names=["hypothesis_word", "dialect_region"])
    shown = (
        counts.set_index(["hypothesis_word", "dialect_region"])["count"]
        .reindex(grid, fill_value=0).reset_index()
    )
    exact_label = f"{word} (exact match)"
    shown["row"] = shown["hypothesis_word"].where(shown["hypothesis_word"] != word, exact_label)
    row_order = [exact_label] + top_variants

    if normalize.startswith("Share"):
        # Denominator = all DIT transcriptions of the word in the region, so columns are comparable.
        region_totals = counts.groupby("dialect_region")["count"].sum()
        shown["value"] = (shown["count"] / shown["dialect_region"].map(region_totals) * 100).round(1)
        value_title, value_format = "Share %", ".1f"
    else:
        shown["value"] = shown["count"]
        value_title, value_format = "Count", "d"

    # The exact match usually dwarfs every variant; scale colours to the variants and clamp, so the
    # variant rows keep their contrast. The saturated exact-match row is labelled with its values.
    is_exact = shown["hypothesis_word"] == word
    color_max = shown.loc[~is_exact, "value"].max() if top_variants else shown["value"].max()
    color_scale = alt.Scale(scheme="blues", domain=[0, max(float(color_max), 1e-9)], clamp=True)

    base = alt.Chart(shown).encode(
        x=alt.X("dialect_region:N", sort=regions_present, title=None,
                axis=alt.Axis(labelAngle=0, labelOverlap=False)),
        y=alt.Y("row:N", sort=row_order, title="DIT transcription"),
    )
    cells = base.mark_rect().encode(
        color=alt.Color("value:Q", scale=color_scale, legend=alt.Legend(title=value_title)),
        tooltip=[
            alt.Tooltip("row:N", title="Transcription"),
            alt.Tooltip("dialect_region:N", title="Region"),
            alt.Tooltip("count:Q", title="Count"),
            alt.Tooltip("value:Q", title=value_title, format=value_format),
        ],
    )
    # Label every non-zero cell so it is easy to spot where a variant occurs. The cell colours come
    # from the same "blues" scale in light and dark mode, so the text colour only depends on the cell:
    # white on dark cells (upper half of the scale; clamped exact-match cells land there too), dark
    # otherwise.
    labels = base.transform_filter(alt.datum.count > 0).mark_text(fontSize=11).encode(
        text=alt.Text("value:Q", format=value_format),
        color=alt.condition(alt.datum.value >= float(color_max) / 2,
                            alt.value("white"), alt.value("#1f2937")),
    )

    st.altair_chart((cells + labels).properties(height=min(486, 26 * len(row_order) + 40)),
                    width="stretch")
    st.caption("Darker = more frequent; empty cells are 0. The first row is the exact match; its "
               "colour is capped at the most frequent variant, so read its values from the labels. "
               "Absolute counts can mislead because regions have different sample sizes; switch to "
               "'Share within region' to see what each region produces.")


def render_example_sentences(ctx: WordContext) -> None:
    """Occurrences tab: a filterable table of every occurrence, or the sentences grouped by DIT variant."""
    view = st.radio("View", options=list(_OCCURRENCE_VIEWS), format_func=_OCCURRENCE_VIEWS.get, horizontal=True,
                    key="occurrence_view", label_visibility="collapsed")
    if view == "table":
        _render_occurrence_table(ctx)
    else:
        _render_sentences_by_variant(ctx)


def _render_occurrence_table(ctx: WordContext) -> None:
    """Every occurrence of the word with its DIT and DAT transcription; selecting a row shows its details."""
    table = build_occurrence_table(ctx.word_rows, ctx.word)
    # Per word, so filter values never refer to another word's transcriptions.
    key = f"occurrences_{ctx.word}"

    col_search, col_show = st.columns([3, 2], vertical_alignment="bottom")
    search = col_search.text_input(
        "Search", key=f"{key}_search", placeholder="Word or sentence text…",
        help="Case-insensitive; matches the reference sentence, both transcripts and both transcribed words.",
    )
    show = col_show.radio("Show", options=list(_SHOW_OPTIONS), format_func=_SHOW_OPTIONS.get, horizontal=True,
                          key=f"{key}_show")
    col_dit, col_dat, col_region = st.columns(3)
    dit_words = col_dit.multiselect("DIT transcription", words_by_frequency(table, "dit_word"),
                                    key=f"{key}_dit", placeholder="All", help="Most frequent first.")
    dat_words = col_dat.multiselect("DAT transcription", words_by_frequency(table, "dat_word"),
                                    key=f"{key}_dat", placeholder="All", help="Most frequent first.")
    regions_present = set(table["dialect_region"])
    regions = col_region.multiselect("Region", [r for r in REGIONS if r in regions_present],
                                     key=f"{key}_regions", placeholder="All",
                                     help="Narrows this table only; the sidebar filter applies to every tab.")

    filtered = filter_occurrences(table, search, dit_words, dat_words, regions, show)
    st.caption(f"{len(filtered):,} of {len(table):,} occurrences. Click a column header to sort; select a row "
               "to see its audio, full transcripts and word alignments.")

    columns = list(_OCCURRENCE_COLUMN_CONFIG)
    if table["dataset"].nunique() > 1:
        columns.append("dataset")
    display = filtered.assign(position=filtered["reference_index"] + 1)[columns]
    # A new key per filter state clears the selection, which would otherwise point at another row.
    filter_state = hash((search, show, tuple(dit_words), tuple(dat_words), tuple(regions)))
    event = st.dataframe(display, width="stretch", height=380, hide_index=True,
                         column_config=_OCCURRENCE_COLUMN_CONFIG, on_select="rerun",
                         selection_mode="single-row", key=f"{key}_table_{filter_state}")

    if not event.selection.rows:
        return
    row = filtered.iloc[event.selection.rows[0]]
    st.markdown(f"#### Selected occurrence: {_sentence_header(row)}")
    render_audio_gate()
    sentence_rows = ctx.sentence_rows[ctx.sentence_rows["path"] == row["path"]]
    _render_sentence_details(row, sentence_rows, ctx.word, ctx.dataset)


def _render_sentences_by_variant(ctx: WordContext) -> None:
    """Sentences grouped by DIT hypothesis (TF-IDF desc).
    Two-level pagination: outer pager selects which variants to show; inner shows sentences.
    Matches and deletions sort to the bottom (TF-IDF = 0).
    """
    df_view, word_rows, word = ctx.sentence_rows, ctx.word_rows, ctx.word
    selected_regions, include_preterite, dataset = ctx.regions, ctx.include_preterite, ctx.dataset
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


def _sentence_header(row: pd.Series) -> str:
    """Region, demographics and the end of the clip ID, e.g. '**ZH** · f · 30-39 · …a1b2c3.flac'."""
    # SDS-200 demographics are sparse; only show gender/age when present (else they render as "nan").
    demographics = " · ".join(str(v) for v in (row["gender"], row["age"]) if pd.notna(v))
    return f"**{row['dialect_region']}**" + (f" · {demographics}" if demographics else "") + f" · …{row['path'][-12:]}"


def _render_example_sentence_expander(row: pd.Series, sentence_rows: pd.DataFrame, word: str, dataset: str) -> None:
    """One sentence as an expander holding its details."""
    with st.expander(_sentence_header(row)):
        _render_sentence_details(row, sentence_rows, word, dataset)


def _render_sentence_details(row: pd.Series, sentence_rows: pd.DataFrame, word: str, dataset: str) -> None:
    """Clip ID, audio player, reference + hypotheses, alignment HTML, and an optional technical alignment graph."""
    path = row["path"]
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
