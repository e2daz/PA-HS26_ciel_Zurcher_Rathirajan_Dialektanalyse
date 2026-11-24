"""
Dialect Word Analysis: the app's start page.

Word selection first (search, example words, word cloud). Choosing a word opens its analysis tabs.

Run via the entry script: streamlit run visualization/Home.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pages.render_helpers import _lexicon_detail as detail, _lexicon_overview as overview
from _data import (  # noqa: E402
    CloudMode, SearchSide, lexicon_search_index, word_alignments, word_sentence_alignments,
)
import _filters  # noqa: E402
from _filters import render_region_filter, shared_filters  # noqa: E402
from _layout import control_row, sidebar_counts, tab_header  # noqa: E402
from i18n import all_translations, localized, number, t  # noqa: E402

# Reference words offered as one-click examples, grouped by the kind of variation they show
# (DIAL-25). Checked against the DIT alignments of both corpora.
EXAMPLE_WORDS = {
    # Lexical: the dialect uses a different word with the same meaning, which Whisper transcribes
    # literally (krankenhaus -> spital, sprechen -> reden, deshalb -> darum, zuvor -> vorher).
    "lexical": [
        "fahrrad", "krankenhaus", "arbeiten", "hause", "sprechen",
        "junge", "deshalb", "nun", "selbst", "zuvor",
    ],
    # Phonetic: Whisper writes Standard German, so a dialect pronunciation shows up as a
    # sound-alike word rather than a dialect spelling. Strongly regional: lösung -> lesung
    # (Wallis 58 %, elsewhere <= 4 %), abend -> oben (Basel 28 %), haben -> habt (Wallis, Bern).
    # Broader: denn -> dann, wann -> wenn, darin -> drin.
    "phonetic": [
        "lösung", "abend", "haben", "denn", "wann", "darin",
    ],
}

# One entry per tab: its translation key and render function. The tab's title and its description per search
# side are `tabs.<key>` and `tabs.<key>_<side>` in the locale files.
ANALYSIS_VIEWS = {
    "overview": detail.render_overview,
    "levenshtein": detail.render_levenshtein_graph,
    "regions": detail.render_regional_distribution,
    "occurrences": detail.render_occurrences,
}
TAB_KEY = "analysis_tab"

WORD_KEY = "selected_word"
SEARCH_SIDE_KEY = "search_by"
# How the search side is written in the URL; the same in every language, so shared links keep working.
SEARCH_SIDES: dict[SearchSide, str] = {"reference": "Reference", "dit": "DIT"}


def _side_label(side: SearchSide) -> str:
    return t(f"search.side_{side}")


# The (search side, word) last written to the URL, to tell a URL changed by Back/Forward from one the page wrote.
_URL_STATE_KEY = "_search_in_url"


def _search_from_url() -> tuple[SearchSide, str]:
    """The search side ("DIT" or "Reference", any case; reference if missing or unknown) and word in the URL."""
    value = st.query_params.get(SEARCH_SIDE_KEY, "").casefold()
    side = next((side for side, label in SEARCH_SIDES.items() if value in (side, label.casefold())), "reference")
    return side, st.query_params.get(WORD_KEY, "")


def _write_search_to_url(side: SearchSide, word: str) -> None:
    """Put the search side and word into the URL in one update, so the browser gets one history entry."""
    params = {k: v for k, v in st.query_params.to_dict().items() if k not in (SEARCH_SIDE_KEY, WORD_KEY)}
    if side != "reference":  # the default side stays out of the URL
        params[SEARCH_SIDE_KEY] = SEARCH_SIDES[side]
    if word:
        params[WORD_KEY] = word
    st.query_params.from_dict(params)


def _clear_word() -> None:
    st.session_state[WORD_KEY] = ""


def _open_word(word: object, side: SearchSide, dataset: str, regions: tuple[str, ...],
               include_preterite: bool) -> None:
    """Open `word` as a `side` word from a click in an analysis table, if the current filters have it (DIAL-35).

    Runs as a widget callback, so the search side and the search box can still be set before they render.
    """
    if not isinstance(word, str) or word.startswith("("):  # a deletion/insertion row has no word to open
        st.session_state["open_note"] = t("notes.no_word_to_open")
        return
    if word not in lexicon_search_index(dataset, regions, include_preterite, side)[0].index:
        st.session_state["open_note"] = t("notes.no_data_as_side", word=word, side=_side_label(side))
        return
    st.session_state[SEARCH_SIDE_KEY] = side
    st.session_state[WORD_KEY] = word


def _select_example_word() -> None:
    """Load the clicked example into the search box, then clear the chip so it can be clicked again."""
    word = st.session_state.get("example_word_pill")
    if word:
        st.session_state[WORD_KEY] = word
    st.session_state["example_word_pill"] = None


def _open_clicked_pair(known_words: pd.Index, side: SearchSide) -> None:
    """Open a word of the clicked 'word → DIT word' pair: the left one if it is a reference word, or in DIT
    search the DIT word on the right."""
    left, right = st.session_state[overview.WORD_CLOUD_KEY]["chart_event"]["name"].split(" → ", 1)
    word = right if side == "dit" else left
    if word in known_words:
        st.session_state[WORD_KEY] = word
    else:
        st.session_state["cloud_note"] = word


def render_example_words(ref_counts: pd.Series) -> None:
    """Example-type choice and the example chips it offers, plus the user's own additions."""
    st.subheader(t("examples.title"))
    # Let the user pick which kind of variation the example words should illustrate (DIAL-25).
    col_type, *_ = control_row()
    example_type = col_type.radio(
        t("examples.type"),
        options=list(EXAMPLE_WORDS),
        format_func=lambda kind: t(f"examples.type_{kind}"),
        horizontal=True,
        key=localized("example_word_type"),
        persist_state="session",
        help=t("examples.type_help"),
    )

    # Only offer words the current dataset/region/preterite filter can show; the search box rejects others.
    example_words = [
        w for w in dict.fromkeys(EXAMPLE_WORDS[example_type] + st.session_state.get("custom_example_words", []))
        if w in ref_counts.index
    ]
    col_examples, col_add = st.columns([5, 1], vertical_alignment="bottom")
    with col_examples:
        st.pills(
            t("examples.pick"),
            options=example_words,
            key="example_word_pill",
            on_change=_select_example_word,
            wrap=True,
            help=t("examples.pick_help"),
        )
    with col_add:
        with st.popover(t("examples.add"), help=t("examples.add_help")):
            st.multiselect(
                t("examples.yours"),
                options=ref_counts.index.tolist(),
                key="custom_example_words",
                persist_state="session",
                placeholder=t("examples.yours_placeholder"),
                help=t("examples.yours_help"),
            )


def render_word_cloud(word_counts: pd.Series, regions: list[str], include_preterite: bool, dataset: str,
                      side: SearchSide) -> None:
    """Word cloud of regionally distinctive pairs, as a way to discover interesting words."""
    st.subheader(t("cloud.title"))
    col_mode, *_ = control_row()
    mode: CloudMode = col_mode.radio(
        t("cloud.compare"),
        options=["ref_dit", "dat_dit"],
        format_func=lambda option: t(f"cloud.compare_{option}"),
        horizontal=True,
        key=localized("cloud_mode"),
        persist_state="session",
        help=t("cloud.compare_help"),
    )

    table = overview.compute_top_table(regions, include_preterite, mode, dataset)

    if table.empty:
        st.warning(t("cloud.empty"))
        return

    overview.render_word_cloud(table, on_click=lambda: _open_clicked_pair(word_counts.index, side))
    overview.render_caption(mode)
    overview.render_top_candidates_expander(table, mode)


def render_analysis(word: str, side: SearchSide, regions: list[str], include_preterite: bool, dataset: str) -> None:
    """Analysis tabs for one reference or DIT word; only the open tab does any work."""
    open_note = st.session_state.pop("open_note", None)
    if open_note:
        st.toast(open_note)
    with st.spinner(t("search.loading_word")):
        sentence_rows = word_sentence_alignments(word, tuple(regions), dataset, include_preterite, side)
    word_rows = word_alignments(sentence_rows, word, side)
    ctx = detail.WordContext(
        word, word_rows, sentence_rows, regions, include_preterite, dataset, side,
        open_word=lambda w, s: _open_word(w, s, dataset, tuple(regions), include_preterite),
    )

    title = t("search.dit_title", word=word) if side == "dit" else word
    labels = {key: t(f"tabs.{key}") for key in ANALYSIS_VIEWS}
    _keep_open_tab(labels)
    tabs = st.tabs(list(labels.values()), key=TAB_KEY, on_change="rerun", bind="query-params")
    for tab, (key, render_view) in zip(tabs, ANALYSIS_VIEWS.items()):
        if tab.open:
            with tab:
                tab_header(title, t(f"tabs.{key}_{side}"))
                render_view(ctx)


def _keep_open_tab(labels: dict[str, str]) -> None:
    """Keep the open tab when the language changes. The tab titles are also the tab widget's value (and the
    `analysis_tab` URL parameter), so a title stored in the other language is mapped to the same tab here."""
    current = st.session_state.get(TAB_KEY) or st.query_params.get(TAB_KEY)
    for key, label in labels.items():
        if current != label and current in all_translations(f"tabs.{key}"):
            st.session_state[TAB_KEY] = label
            return


# --- Page 1 ---
dataset, include_preterite = shared_filters()
selected_regions = render_region_filter()

# The word search sits at the top of the sidebar, above the filters, so the analysis tabs can start at the
# top of the page. It runs after the filters because its options depend on them.
word_slot = _filters.word_slot()
# The search side and word are synced with the URL by hand rather than with bind="query-params": opening a word
# from an Overview table changes both at once, and two bound widgets would leave two browser history entries,
# so Back would need two clicks. A URL that changed from outside (page load, Back/Forward) wins; otherwise the
# widgets do and the URL follows below the search box.
url_state = _search_from_url()
if st.session_state.get(_URL_STATE_KEY) != url_state:
    st.session_state[SEARCH_SIDE_KEY], st.session_state[WORD_KEY] = url_state
search_side: SearchSide = word_slot.radio(
    t("search.by"),
    options=list(SEARCH_SIDES),
    format_func=_side_label,
    horizontal=True,
    key=localized(SEARCH_SIDE_KEY),
    help=t("search.by_help"),
)

with st.spinner(t("search.loading_index")):
    # Cached per (dataset, regions, preterite, side): word frequencies + sidebar counts.
    word_counts, n_alignments, n_sentences = lexicon_search_index(
        dataset, tuple(selected_regions), include_preterite, search_side
    )

# A filter change, or switching the search side, can remove the selected word from the search index; go back
# and say why.
previous_word = st.session_state.get(WORD_KEY)
side_changed = st.session_state.get("_prev_search_side", search_side) != search_side
st.session_state["_prev_search_side"] = search_side
if previous_word and previous_word not in word_counts.index:
    st.session_state[WORD_KEY] = ""
    st.session_state["filtered_out_word"] = (
        t("notes.not_a_side_word", word=previous_word, side=_side_label(search_side)) if side_changed
        else t("notes.filtered_out", word=previous_word)
    )

analysis_open = st.session_state.get(WORD_KEY, "") in word_counts.index

selected_word = word_slot.selectbox(
    t(f"search.box_{search_side}"),
    options=[""] + word_counts.index.tolist(),
    key=WORD_KEY,
    placeholder=t("search.placeholder"),
    help=t(f"search.box_{search_side}_help"),
)
if (search_side, selected_word) != url_state:
    _write_search_to_url(search_side, selected_word)
st.session_state[_URL_STATE_KEY] = (search_side, selected_word)
if analysis_open:
    word_slot.button(t("search.back"), on_click=_clear_word, width="stretch", help=t("search.back_help"))
word_slot.divider()

if st.session_state.get("_prev_selected_word") != selected_word:
    for k in [k for k in st.session_state if k.startswith("audio_loaded_")]:
        del st.session_state[k]
    st.session_state["_prev_selected_word"] = selected_word

if analysis_open:
    render_analysis(selected_word, search_side, selected_regions, include_preterite, dataset)
else:
    st.title(t("nav.word_analysis"))
    if search_side == "dit":
        st.caption(t("search.start_dit"))
    else:
        st.caption(t("search.start_reference"))
        # The example words are reference words, so they are only offered in reference search.
        render_example_words(word_counts)

    n_words = number(len(word_counts))
    sidebar_counts(t("counts.dit_words", n=n_words) if search_side == "dit" else t("counts.reference_words", n=n_words),
                   t("counts.alignments", n=number(n_alignments)), t("counts.sentences", n=number(n_sentences)))

    filtered_out = st.session_state.pop("filtered_out_word", None)
    if filtered_out:
        st.info(filtered_out)
    cloud_note = st.session_state.pop("cloud_note", None)
    if cloud_note:
        st.info(t("notes.cloud_no_dit", word=cloud_note) if search_side == "dit"
                else t("notes.cloud_dat_word", word=cloud_note))

    render_word_cloud(word_counts, selected_regions, include_preterite, dataset, search_side)
