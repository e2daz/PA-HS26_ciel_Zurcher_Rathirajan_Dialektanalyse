"""Detail view for the Dialect Word Analysis page."""
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
from _help import chart_help  # noqa: E402
from _layout import control_stack  # noqa: E402
from i18n import localized, number, t  # noqa: E402
from _data import (  # noqa: E402
    ALPHA, LAMBDA, REGIONS, USE_GLOBAL_LEXICAL_NORMALIZATION,
    USE_SQUARED_POSITIONAL, SearchSide, resolve_audio_path, tfidf_matrix_pairs,
)
from pages.render_helpers import _levenshtein_graph  # noqa: E402
from bipartite_matching import build_full_bipartite_graph, solve_matching  # noqa: E402
from lexicon_graph import build_levenshtein_ring_graph  # noqa: E402
from occurrence_table import (  # noqa: E402
    DELETION_LABEL, INSERTION_LABEL, SHOW_ALL, SHOW_DIT_DIFFERS, SHOW_MODELS_DIFFER, build_occurrence_table,
    filter_occurrences, words_by_frequency,
)
from plot_helpers import plot_reduced_bipartite_graph_with_matching  # noqa: E402
from preprocessing import tokenize  # noqa: E402
from word_similarity_calculator import WordSimilarityCalculator  # noqa: E402

def _hypothesis_table_columns() -> dict:
    """Overview tables of a reference word: what DIT or DAT wrote for it."""
    return {
        "hypothesis_word": st.column_config.TextColumn(t("overview.col_variant"), help=t("overview.col_variant_help")),
        "count": st.column_config.NumberColumn(t("overview.col_count"), help=t("overview.col_count_help")),
        "mean sim": st.column_config.NumberColumn(t("overview.col_sim"), help=t("overview.col_sim_help")),
        "TF-IDF": st.column_config.NumberColumn(help=t("overview.col_tfidf_help"), format="%.5f"),
    }


# All of these go to the graph at once, so the variant count can change there without a rerun.
_GRAPH_MAX_VARIANTS = 100

def _source_table_columns() -> dict:
    """Overview tables of a searched DIT word: the reference and DAT words it was aligned to."""
    inserted = t("labels.inserted")
    return {
        "reference_word": st.column_config.TextColumn(t("overview.col_reference_word"),
                                                      help=t("overview.col_reference_word_help", inserted=inserted)),
        "dat_word": st.column_config.TextColumn(t("overview.col_dat_word"),
                                                help=t("overview.col_dat_word_help", inserted=inserted)),
        "count": st.column_config.NumberColumn(t("overview.col_count"), help=t("overview.col_source_count_help")),
        "mean sim": st.column_config.NumberColumn(t("overview.col_sim"), help=t("overview.col_source_sim_help")),
        "TF-IDF": st.column_config.NumberColumn(help=t("overview.col_source_tfidf_help"), format="%.5f"),
    }


# Rankings of the ring graph; their names are `levenshtein.rank_<key>` in the locale files.
_GRAPH_RANKINGS = ("tfidf", "count", "similarity")

_VARIANT_TABLE_HEIGHT = 350

# First column of the Overview word tables: an arrow that opens the row's word (DIAL-35).
_OPEN_COLUMN = "open"
_OPEN_ARROW = "→"
_OPEN_CLICKS_KEY = "_overview_open_clicks"

# Streamlit groups both step buttons right of the field; this matches the ring graph's "− n +" steppers.
_REGION_VARIANTS_CSS = """
<style>
  .st-key-region_dist_top_n [data-testid="stNumberInputContainer"] {
    width: auto; height: auto; gap: 4px; align-items: center; background: transparent; border: none; overflow: visible;
  }
  .st-key-region_dist_top_n [data-testid="stNumberInputContainer"] > div { display: contents; }
  .st-key-region_dist_top_n input, .st-key-region_dist_top_n button[data-testid^="stNumberInputStep"] {
    height: 28px; border: 1px solid rgba(128, 128, 128, 0.45); border-radius: 4px;
  }
  .st-key-region_dist_top_n input { flex: none; width: 3.2em; padding: 3px; text-align: center; }
  .st-key-region_dist_top_n button[data-testid^="stNumberInputStep"] { width: 30px; }
  .st-key-region_dist_top_n [data-testid="stNumberInputStepDown"] { order: -1; }
  /* Streamlit pushes a number input's (?) to the far end of the field; keep it next to the label instead. */
  .st-key-region_dist_top_n [data-testid="stWidgetLabel"] > div { flex: none; margin-left: 0.5rem; }
</style>
"""

# Filters of the occurrence table; their names are `occurrences.show_<key>` in the locale files.
_SHOW_OPTIONS = (SHOW_ALL, SHOW_DIT_DIFFERS, SHOW_MODELS_DIFFER)


def _occurrence_columns(is_dit: bool) -> dict:
    """Column config of the occurrence table, also its column order ("dataset" is appended when both corpora are
    shown). One row per reference token for a reference word, or per DIT token for a DIT word."""
    common_start = {"dialect_region": st.column_config.TextColumn(t("occurrences.col_region"))}
    if is_dit:
        words = {
            "ref_word": st.column_config.TextColumn(t("occurrences.col_reference"),
                                                    help=t("occurrences.col_reference_help")),
            "dat_word": st.column_config.TextColumn("DAT", help=t("occurrences.col_dat_of_dit_help")),
            "ref_similarity": st.column_config.NumberColumn(t("occurrences.col_ref_sim"), format="%.2f",
                                                            help=t("occurrences.col_ref_sim_help")),
            "dat_similarity": st.column_config.NumberColumn(t("occurrences.col_dat_sim"), format="%.2f",
                                                            help=t("occurrences.col_dat_sim_of_dit_help")),
        }
        position_help = t("occurrences.col_position_dit_help")
    else:
        words = {
            "dit_word": st.column_config.TextColumn("DIT", help=t("occurrences.col_dit_help")),
            "dat_word": st.column_config.TextColumn("DAT", help=t("occurrences.col_dat_help")),
            "dit_similarity": st.column_config.NumberColumn(t("occurrences.col_dit_sim"), format="%.2f",
                                                            help=t("occurrences.col_dit_sim_help")),
            "dat_similarity": st.column_config.NumberColumn(t("occurrences.col_dat_sim"), format="%.2f",
                                                            help=t("occurrences.col_dat_sim_help")),
        }
        position_help = t("occurrences.col_position_help")
    return {
        **common_start,
        **words,
        "reference": st.column_config.TextColumn(t("occurrences.col_reference_sentence"), width="large"),
        "dit_hypothesis": st.column_config.TextColumn(t("occurrences.col_dit_transcript"), width="large"),
        "dat_hypothesis": st.column_config.TextColumn(t("occurrences.col_dat_transcript"), width="large"),
        "position": st.column_config.NumberColumn(t("occurrences.col_position"), help=position_help),
        "dataset": st.column_config.TextColumn(t("occurrences.col_dataset")),
    }


def _word_label(word: object) -> str:
    """A word as shown in a table or filter: the deletion/insertion labels in the current language."""
    if word == DELETION_LABEL:
        return t("labels.deleted")
    if word == INSERTION_LABEL:
        return t("labels.inserted")
    return word


_LABEL_STYLE = (
    "padding: 4px 12px; font-weight: 600; text-align: center; color: #888; "
    "white-space: nowrap; border-bottom: 1px solid #eee;"
)


@dataclass(frozen=True)
class WordContext:
    """Everything an analysis tab needs to render one searched word."""
    word: str
    # Alignments of the word itself: REF -> DIT and REF -> DAT for a reference word, REF -> DIT and DAT -> DIT
    # for a DIT word (see `_data.word_alignments`).
    word_rows: pd.DataFrame
    sentence_rows: pd.DataFrame  # every alignment of the sentences containing the word
    regions: list[str]
    include_preterite: bool
    dataset: str
    side: SearchSide = "reference"
    # Opens another word's analysis, called as open_word(word, side) from a table click; None disables the clicks.
    open_word: Callable[[object, SearchSide], None] | None = None

    @property
    def is_dit(self) -> bool:
        return self.side == "dit"

    def rows_of(self, model: str) -> pd.DataFrame:
        return self.word_rows[self.word_rows["model"] == model]


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


def _word_table(table: pd.DataFrame, word_column: str, ctx: WordContext, open_side: SearchSide, key: str,
                column_config: dict) -> None:
    """A word table of the Overview tab. Every row with a word starts with an arrow; clicking the arrow opens
    that word as a `open_side` word (DIAL-35).

    Row selection would show a checkbox, which reads as "select this row" rather than "go to this word", so the
    table selects single cells instead and only reacts to the arrow column.
    """
    if ctx.open_word is None:
        st.dataframe(table, width="stretch", height=_VARIANT_TABLE_HEIGHT, hide_index=True,
                     column_config=column_config)
        return

    # Deletion/insertion rows have no word to open (NaN, or a label in brackets such as "(inserted)").
    openable = table[word_column].map(lambda word: isinstance(word, str) and not word.startswith("("))
    shown = table.assign(**{_OPEN_COLUMN: openable.map({True: _OPEN_ARROW, False: ""}),
                            word_column: table[word_column].map(_word_label)})
    shown = shown[[_OPEN_COLUMN, *table.columns]]
    # A new key after every arrow click clears the selection, so the same arrow can be clicked again later.
    key = f"{key}_{st.session_state.get(_OPEN_CLICKS_KEY, 0)}"

    def open_clicked_arrow() -> None:
        cells = st.session_state[key].selection.cells
        if not cells or cells[0][1] != _OPEN_COLUMN:
            return
        row = cells[0][0]
        st.session_state[_OPEN_CLICKS_KEY] = st.session_state.get(_OPEN_CLICKS_KEY, 0) + 1
        if openable.iloc[row]:
            ctx.open_word(table.iloc[row][word_column], open_side)

    st.dataframe(shown, width="stretch", height=_VARIANT_TABLE_HEIGHT, hide_index=True,
                 column_config={_OPEN_COLUMN: st.column_config.TextColumn("", width=40, help=t("overview.open_help")),
                                **column_config},
                 on_select=open_clicked_arrow, selection_mode="single-cell", key=key)


def render_overview(ctx: WordContext) -> None:
    """General figures for the word and its most frequent variants (reference word) or sources (DIT word)."""
    if ctx.is_dit:
        _render_dit_overview(ctx)
        return

    dit_rows = ctx.rows_of("dialect-ignorant")
    dat_rows = ctx.rows_of("dialect-aware")

    def variant_count(rows: pd.DataFrame) -> int:
        hypotheses = rows["hypothesis_word"].dropna()
        return hypotheses[hypotheses != ctx.word].nunique()

    col_occurrences, col_sentences, col_dit, col_dat = st.columns(4)
    col_occurrences.metric(t("overview.occurrences"), number(len(dit_rows)),
                           help=t("overview.occurrences_help", n=len(ctx.regions)))
    col_sentences.metric(t("overview.sentences"), number(ctx.word_rows['path'].nunique()))
    col_dit.metric(t("overview.dit_variants"), number(variant_count(dit_rows)), help=t("overview.dit_variants_help"))
    col_dat.metric(t("overview.dat_variants"), number(variant_count(dat_rows)), help=t("overview.dat_variants_help"))

    col_dit_table, col_dat_table = st.columns(2)
    with col_dit_table:
        st.markdown(f"**{t('overview.dit_table')}**")
        _word_table(
            _hypothesis_table(dit_rows, ctx.word, "dialect-ignorant", ctx.regions, ctx.include_preterite, ctx.dataset)
            .sort_values("count", ascending=False).reset_index(drop=True),
            "hypothesis_word", ctx, "dit", f"overview_dit_table_{ctx.word}", _hypothesis_table_columns(),
        )
    with col_dat_table:
        st.markdown(f"**{t('overview.dat_table')}**")
        _word_table(
            _hypothesis_table(dat_rows, ctx.word, "dialect-aware", ctx.regions, ctx.include_preterite, ctx.dataset)
            .reset_index(drop=True),
            "hypothesis_word", ctx, "reference", f"overview_dat_table_{ctx.word}", _hypothesis_table_columns(),
        )
    chart_help(t("overview.shows"), t("overview.use"))


def _render_dit_overview(ctx: WordContext) -> None:
    """Overview of a searched DIT word: which reference words and which DAT words it was written for."""
    ref_rows = ctx.rows_of("dialect-ignorant")
    dat_rows = ctx.rows_of("dat-dit")

    def source_count(rows: pd.DataFrame) -> int:
        sources = rows["reference_word"].dropna()
        return sources[sources != ctx.word].nunique()

    col_occurrences, col_sentences, col_ref, col_dat = st.columns(4)
    col_occurrences.metric(t("overview.occurrences"), number(len(ref_rows)),
                           help=t("overview.dit_occurrences_help", n=len(ctx.regions)))
    col_sentences.metric(t("overview.sentences"), number(ref_rows['path'].nunique()))
    col_ref.metric(t("overview.reference_words"), number(source_count(ref_rows)),
                   help=t("overview.reference_words_help"))
    col_dat.metric(t("overview.dat_words"), number(source_count(dat_rows)), help=t("overview.dat_words_help"))

    col_ref_table, col_dat_table = st.columns(2)
    with col_ref_table:
        st.markdown(f"**{t('overview.ref_source_table')}**")
        _word_table(_source_table(ref_rows, ctx, with_tfidf=True), "reference_word", ctx, "reference",
                    f"overview_ref_table_dit_{ctx.word}", _source_table_columns())
    with col_dat_table:
        st.markdown(f"**{t('overview.dat_source_table')}**")
        _word_table(_source_table(dat_rows, ctx, with_tfidf=False).rename(columns={"reference_word": "dat_word"}),
                    "dat_word", ctx, "reference", f"overview_dat_table_dit_{ctx.word}", _source_table_columns())
    chart_help(t("overview.dit_shows", inserted=t("labels.inserted")), t("overview.dit_use"))


def _source_table(rows: pd.DataFrame, ctx: WordContext, with_tfidf: bool) -> pd.DataFrame:
    """Words aligned to the searched DIT word (in `reference_word`) with count, mean similarity and, for the
    reference side, TF-IDF; most frequent first. Insertions are one row labelled `INSERTION_LABEL`."""
    out = (
        rows.groupby("reference_word", dropna=False)
        .agg(count=("path", "size"), mean_sim=("similarity", "mean"))
        .reset_index()
    )
    out["mean sim"] = out["mean_sim"].round(3)
    columns = ["reference_word", "count", "mean sim"]
    if with_tfidf:
        lookup = _pair_tfidf_lookup(ctx.regions, ctx.include_preterite, ctx.dataset)
        out["TF-IDF"] = out["reference_word"].apply(lambda ref: lookup(ref, ctx.word)).round(5)
        columns.append("TF-IDF")
    out["reference_word"] = out["reference_word"].fillna(INSERTION_LABEL)
    return out[columns].sort_values("count", ascending=False).reset_index(drop=True)


def render_levenshtein_graph(ctx: WordContext) -> None:
    """Render the top variants (reference word) or reference sources (DIT word) on rings by edit distance."""
    word = ctx.word
    dit_rows = ctx.rows_of("dialect-ignorant")
    lookup = _pair_tfidf_lookup(ctx.regions, ctx.include_preterite, ctx.dataset)
    if ctx.is_dit:
        # Around a DIT word: the reference words DIT wrote it for.
        variant_column = "reference_word"
        labels = {"center_label": "DIT", "variant_label": t("levenshtein.variants_dit")}
        tfidf = {ref: lookup(ref, word) for ref in dit_rows[variant_column].dropna().unique()}
        what = t("levenshtein.what_dit")
    else:
        variant_column = "hypothesis_word"
        labels = {"center_label": t("levenshtein.center_reference"), "variant_label": t("levenshtein.variants_reference")}
        tfidf = {hyp: lookup(word, hyp) for hyp in dit_rows[variant_column].dropna().unique()}
        what = t("levenshtein.what_reference")

    # One graph per ranking, so the ranking can be switched in the graph's own controls without a rerun.
    graphs = {
        rank_by: build_levenshtein_ring_graph(dit_rows, word, top_n=_GRAPH_MAX_VARIANTS, rank_by=rank_by,
                                              tfidf_by_hypothesis=tfidf, variant_column=variant_column, **labels)
        for rank_by in _GRAPH_RANKINGS
    }
    rank_labels = {rank_by: t(f"levenshtein.rank_{rank_by}") for rank_by in _GRAPH_RANKINGS}
    _levenshtein_graph.render(graphs, rank_labels, dark=st.context.theme.type == "dark")
    chart_help(t("levenshtein.shows", what=what), t("levenshtein.use"))


def render_regional_distribution(ctx: WordContext) -> None:
    """Heatmap of where the searched word and its DIT variants occur across the dialect regions.
    The exact match (hypothesis == reference) is the first row, so each variant can be compared
    against how often DIT simply transcribed the reference word.

    For a DIT word the rows are the reference words DIT wrote it for, with the exact match first again."""
    word, selected_regions = ctx.word, ctx.regions
    # Renamed to `hypothesis_word` for a DIT word, so the heatmap code below is shared by both sides.
    variant_column = "reference_word" if ctx.is_dit else "hypothesis_word"
    dit_rows = ctx.rows_of("dialect-ignorant")
    dit_rows = dit_rows[dit_rows[variant_column].notna()]
    if ctx.is_dit:
        dit_rows = dit_rows.drop(columns="hypothesis_word").rename(columns={variant_column: "hypothesis_word"})
        variant_noun, axis_title = t("regions.noun_dit"), t("regions.axis_dit")
    else:
        variant_noun, axis_title = t("regions.noun_reference"), t("regions.axis_reference")
    if dit_rows.empty:
        st.caption(t("regions.empty"))
        return

    counts = (
        dit_rows.groupby(["hypothesis_word", "dialect_region"], observed=True)
        .size().reset_index(name="count")
    )
    # `dialect_region` is categorical; make it a plain string so the share division below works.
    counts["dialect_region"] = counts["dialect_region"].astype(str)

    with control_stack():
        normalize = st.radio(
            t("regions.values"),
            options=["count", "share"],
            format_func=lambda option: t(f"regions.values_{option}"),
            horizontal=True,
            key=localized("region_dist_values"),
            help=t("regions.values_help", variants=variant_noun),
        )
        st.html(_REGION_VARIANTS_CSS)
        # No min/max here: Streamlit would keep an out-of-range entry in the field and silently ignore it.
        top_n = st.number_input(t("regions.top_n"), value=12, step=1, key="region_dist_top_n",
                                on_change=_clamp_region_variants,
                                help=t("regions.top_n_help", variants=variant_noun))

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
    exact_label = t("regions.exact_match", word=word)
    shown["row"] = shown["hypothesis_word"].where(shown["hypothesis_word"] != word, exact_label)
    row_order = [exact_label] + top_variants

    if normalize == "share":
        # Denominator = all DIT transcriptions of the word in the region, so columns are comparable.
        region_totals = counts.groupby("dialect_region")["count"].sum()
        shown["value"] = (shown["count"] / shown["dialect_region"].map(region_totals) * 100).round(1)
        value_title, value_format = t("regions.share_title"), ".1f"
    else:
        shown["value"] = shown["count"]
        value_title, value_format = t("regions.count_title"), "d"

    # The exact match usually dwarfs every variant; scale colours to the variants and clamp, so the
    # variant rows keep their contrast. The saturated exact-match row is labelled with its values.
    is_exact = shown["hypothesis_word"] == word
    color_max = shown.loc[~is_exact, "value"].max() if top_variants else shown["value"].max()
    color_scale = alt.Scale(scheme="blues", domain=[0, max(float(color_max), 1e-9)], clamp=True)

    base = alt.Chart(shown).encode(
        x=alt.X("dialect_region:N", sort=regions_present, title=None,
                axis=alt.Axis(labelAngle=0, labelOverlap=False)),
        y=alt.Y("row:N", sort=row_order, title=axis_title),
    )
    cells = base.mark_rect().encode(
        color=alt.Color("value:Q", scale=color_scale, legend=alt.Legend(title=value_title)),
        tooltip=[
            alt.Tooltip("row:N", title=t("regions.tooltip_dit") if ctx.is_dit else t("regions.tooltip_reference")),
            alt.Tooltip("dialect_region:N", title=t("regions.tooltip_region")),
            alt.Tooltip("count:Q", title=t("regions.count_title")),
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
    what = t("regions.what_dit") if ctx.is_dit else t("regions.what_reference")
    chart_help(t("regions.shows", what=what), t("regions.use"))


def render_occurrences(ctx: WordContext) -> None:
    """Occurrences tab: every occurrence of the word with what the other transcripts have there; selecting a
    row shows its audio and word alignment below the table."""
    controls = control_stack()
    table = build_occurrence_table(ctx.word_rows, ctx.word, anchor=ctx.side)
    # Per word and side, so filter values never refer to another word's transcriptions.
    key = f"occurrences_dit_{ctx.word}" if ctx.is_dit else f"occurrences_{ctx.word}"

    search = controls.text_input(
        t("occurrences.search"), key=f"{key}_search", persist_state="session",
        placeholder=t("occurrences.search_placeholder"), help=t("occurrences.search_help"),
    )
    show = controls.radio(t("occurrences.show"), options=list(_SHOW_OPTIONS),
                          format_func=lambda option: t(f"occurrences.show_{option}"), horizontal=True,
                          key=localized(f"{key}_show"), persist_state="session", help=t("occurrences.show_help"))
    # Three short filters fit side by side in the half-width stack, which keeps the table higher up.
    col_other, col_dat, col_region = controls.columns(3)
    # For a reference word the first filter is what DIT wrote; for a DIT word it is the reference word.
    other_column, other_label = (("ref_word", t("occurrences.filter_reference")) if ctx.is_dit
                                 else ("dit_word", t("occurrences.filter_dit")))
    other_words = col_other.multiselect(other_label, words_by_frequency(table, other_column), format_func=_word_label,
                                        key=localized(f"{key}_{'ref' if ctx.is_dit else 'dit'}"),
                                        persist_state="session",
                                        placeholder=t("occurrences.all"), help=t("occurrences.frequent_first"))
    dat_words = col_dat.multiselect(t("occurrences.filter_dat"), words_by_frequency(table, "dat_word"),
                                    format_func=_word_label, key=localized(f"{key}_dat"), persist_state="session",
                                    placeholder=t("occurrences.all"), help=t("occurrences.frequent_first"))
    regions_present = set(table["dialect_region"])
    regions = col_region.multiselect(t("occurrences.filter_region"), [r for r in REGIONS if r in regions_present],
                                     key=f"{key}_regions", persist_state="session", placeholder=t("occurrences.all"),
                                     help=t("occurrences.filter_region_help"))

    word_filter = {"ref_words": other_words} if ctx.is_dit else {"dit_words": other_words}
    filtered = filter_occurrences(table, search, dat_words=dat_words, regions=regions, show=show, **word_filter)
    st.caption(t("occurrences.count", shown=number(len(filtered)), total=number(len(table))))

    column_config = _occurrence_columns(ctx.is_dit)
    columns = [c for c in column_config if c != "dataset"]
    if table["dataset"].nunique() > 1:
        columns.append("dataset")
    index_column = "hypothesis_index" if ctx.is_dit else "reference_index"
    display = filtered.assign(position=filtered[index_column] + 1)[columns]
    for word_column in ("ref_word", "dit_word", "dat_word"):
        if word_column in display:
            display[word_column] = display[word_column].map(_word_label)
    # A new key per filter state clears the selection, which would otherwise point at another row.
    filter_state = hash((search, show, tuple(other_words), tuple(dat_words), tuple(regions)))
    event = st.dataframe(display, width="stretch", height=380, hide_index=True,
                         column_config=column_config, on_select="rerun",
                         selection_mode="single-row", key=f"{key}_table_{filter_state}")
    what = (t("occurrences.what_dit", inserted=t("labels.inserted")) if ctx.is_dit
            else t("occurrences.what_reference"))
    chart_help(what, t("occurrences.use"))

    if not event.selection.rows:
        return
    row = filtered.iloc[event.selection.rows[0]]
    # Keyed, so the scroll below can find the heading.
    st.container(key="selected_occurrence").markdown(f"#### {t('occurrences.selected')} {_sentence_header(row)}")
    # The details open below the table, out of view; bring them up when a row is selected.
    _scroll_to_selected_occurrence(row["path"])
    render_audio_gate()
    sentence_rows = ctx.sentence_rows[ctx.sentence_rows["path"] == row["path"]]
    _render_sentence_details(row, sentence_rows, ctx.word, ctx.dataset, ctx.side)


def _scroll_to_selected_occurrence(path: str) -> None:
    """Scroll the page to the selected-occurrence heading.

    Streamlit has no scroll command, so an invisible frame does it; it shares the app's origin and can reach
    the page. Streamlit only reloads the frame when its content changes, and the content holds the path, so
    the page scrolls once per newly selected row and not on every rerun (e.g. loading the audio).
    """
    st.iframe(f"""<!-- {html.escape(path)} -->
<script>
const heading = parent.document.querySelector(".st-key-selected_occurrence");
if (heading) {{
  // Stay clear of the analysis tabs pinned to the top of the page.
  heading.style.scrollMarginTop = "80px";
  heading.scrollIntoView({{behavior: "smooth", block: "start"}});
}}
</script>""", height=1)


def _pair_tfidf_lookup(selected_regions: list[str], include_preterite: bool,
                       dataset: str) -> Callable[[object, object], float]:
    """Return (ref, DIT-hyp) -> highest TF-IDF of that pair across the selected regions; 0 for an identical
    pair, a missing word or a pair the matrix doesn't have."""
    matrix, _vocab, word_to_idx, region_order = tfidf_matrix_pairs(include_preterite, "ref_dit", dataset)
    selected_idx = [i for i, r in enumerate(region_order) if r in selected_regions]

    def lookup(ref, hyp):
        if pd.isna(ref) or pd.isna(hyp) or ref == hyp or not selected_idx:
            return 0.0
        col = word_to_idx.get(f"{ref}+{hyp}")
        return float(matrix[selected_idx, col].max()) if col is not None else 0.0

    return lookup


def _dit_tfidf_lookup(ref_word: str, selected_regions: list[str], include_preterite: bool,
                      dataset: str) -> Callable[[str], float]:
    """Return hyp -> highest TF-IDF of the (ref, DIT-hyp) pair across the selected regions."""
    lookup = _pair_tfidf_lookup(selected_regions, include_preterite, dataset)
    return lambda hyp: lookup(ref_word, hyp)


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
                           ref_label: str | None = None, searched_row: str = "ref") -> str:
    """Build a 2-row HTML alignment table; highlights the columns whose `searched_row` ("ref" or "hyp") word
    == searched_word."""
    ref_label = t("details.reference") if ref_label is None else ref_label
    ref_cells = [f'<td style="{_LABEL_STYLE}">{html.escape(ref_label)}</td>']
    hyp_cells = [f'<td style="{_LABEL_STYLE}">{html.escape(hyp_label)}</td>']
    for col in columns:
        is_searched = searched_word is not None and col[searched_row] == searched_word
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


def _render_sentence_details(row: pd.Series, sentence_rows: pd.DataFrame, word: str, dataset: str,
                             side: SearchSide = "reference") -> None:
    """Clip ID, audio player, reference + hypotheses, alignment HTML, and an optional technical alignment graph."""
    path = row["path"]
    st.markdown(f"**{t('details.clip_id')}** `{path}`")

    audio_file = resolve_audio_path(path, dataset)
    if audio_file is None:
        st.caption(t("details.audio_missing", path=path))
    elif not audio_unlocked():
        # Audio is the only gated feature: without the session unlock we never call st.audio,
        # so no media URL is minted and the clips can't be crawled.
        st.caption(t("details.audio_locked"))
    else:
        audio_key = f"audio_loaded_{path}"
        loaded = st.session_state.get(audio_key, False)
        if not loaded and st.button(t("details.load_audio"), key=f"load_{path}", help=t("details.load_audio_help")):
            st.session_state[audio_key] = True
            loaded = True
        if loaded:
            st.audio(str(audio_file))

    st.markdown(
        f"**{t('details.reference')}:** {row['reference']}  \n"
        f"**{t('details.hypothesis_dit')}:** {row['dit_hypothesis']}  \n"
        f"**{t('details.hypothesis_dat')}:** {row['dat_hypothesis']}"
    )

    dat_cols = _alignment_columns(sentence_rows[sentence_rows["model"] == "dialect-aware"])
    dit_cols = _alignment_columns(sentence_rows[sentence_rows["model"] == "dialect-ignorant"])
    dat_dit_cols = _alignment_columns(sentence_rows[sentence_rows["model"] == "dat-dit"])

    # A reference word is on the reference row of REF -> DIT and REF -> DAT; a DIT word is on the DIT row of
    # REF -> DIT and DAT -> DIT.
    dit_word = word if side == "dit" else None
    ref_word = None if side == "dit" else word
    st.markdown(_render_alignment_html(dit_cols, t("details.hypothesis_dit"), searched_word=word,
                                       searched_row="hyp" if side == "dit" else "ref"),
                unsafe_allow_html=True)
    st.markdown(_render_alignment_html(dat_cols, t("details.hypothesis_dat"), searched_word=ref_word),
                unsafe_allow_html=True)
    st.markdown(_render_alignment_html(dat_dit_cols, t("details.hypothesis_dit"), searched_word=dit_word,
                                       ref_label=t("details.hypothesis_dat"), searched_row="hyp"),
                unsafe_allow_html=True)
    chart_help(t("details.alignment_shows"), t("details.alignment_use"))

    if st.checkbox(t("details.technical"), key=f"graph_{path}", help=t("details.technical_help")):
        png_dit = _reduced_graph_png(row["reference"], row["dit_hypothesis"])

        if png_dit is not None:
            st.markdown(f"**REF → DIT: {t('details.matching')}**")
            st.image(png_dit)
        png_dat = _reduced_graph_png(row["reference"], row["dat_hypothesis"])

        if png_dat is not None:
            st.markdown(f"**REF → DAT: {t('details.matching')}**")
            st.image(png_dat)
        png_dat_dit = _reduced_graph_png(row["dat_hypothesis"], row["dit_hypothesis"])

        if png_dat_dit is not None:
            st.markdown(f"**DAT → DIT: {t('details.matching')}**")
            st.image(png_dat_dit)
