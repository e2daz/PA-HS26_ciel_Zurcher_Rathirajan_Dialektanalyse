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

# Reference words offered as one-click examples, grouped by the kind of variation they show
# (DIAL-25). Checked against the DIT alignments of both corpora.
EXAMPLE_WORDS = {
    # Lexical: the dialect uses a different word with the same meaning, which Whisper transcribes
    # literally (krankenhaus -> spital, sprechen -> reden, deshalb -> darum, zuvor -> vorher).
    "Lexical": [
        "fahrrad", "krankenhaus", "arbeiten", "hause", "sprechen",
        "junge", "deshalb", "nun", "selbst", "zuvor",
    ],
    # Phonetic: Whisper writes Standard German, so a dialect pronunciation shows up as a
    # sound-alike word rather than a dialect spelling. Strongly regional: lösung -> lesung
    # (Wallis 58 %, elsewhere <= 4 %), abend -> oben (Basel 28 %), haben -> habt (Wallis, Bern).
    # Broader: denn -> dann, wann -> wenn, darin -> drin.
    "Phonetic": [
        "lösung", "abend", "haben", "denn", "wann", "darin",
    ],
}

# One entry per tab: a render function taking a WordContext and, per search side, one sentence on what the
# tab shows.
ANALYSIS_VIEWS = {
    "Overview": (detail.render_overview, {
        "reference": "Which words Whisper (DIT) and the FHNW model (DAT) wrote for this word, and how often.",
        "dit": "For which reference words Whisper (DIT) wrote this word, and which DAT words are aligned to it.",
    }),
    "Levenshtein": (detail.render_levenshtein_graph, {
        "reference": "The DIT variants of this word, arranged by how many letters they differ from it.",
        "dit": "The reference words DIT wrote this word for, arranged by how many letters they differ from it.",
    }),
    "Regions": (detail.render_regional_distribution, {
        "reference": "In which dialect regions DIT writes which variant of this word.",
        "dit": "In which dialect regions DIT writes this word for which reference word.",
    }),
    "Occurrences": (detail.render_occurrences, {
        "reference": "Every occurrence of this word, with both transcripts, the audio and the word alignment.",
        "dit": "Every place DIT wrote this word, with both transcripts, the audio and the word alignment.",
    }),
}

WORD_KEY = "selected_word"
SEARCH_SIDE_KEY = "search_by"
SEARCH_SIDES: dict[SearchSide, str] = {"reference": "Reference", "dit": "DIT"}
# The search box's label and help per side.
SEARCH_BOX = {
    "reference": ("Reference word", "Type a Standard German word to see its dialect-specific variants."),
    "dit": ("DIT word", "Type a word Whisper (DIT) wrote to see which reference words it was written for."),
}


def _clear_word() -> None:
    st.session_state[WORD_KEY] = ""


def _open_word(word: object, side: SearchSide, dataset: str, regions: tuple[str, ...],
               include_preterite: bool) -> None:
    """Open `word` as a `side` word from a click in an analysis table, if the current filters have it (DIAL-35).

    Runs as a widget callback, so the search side and the search box can still be set before they render.
    """
    if not isinstance(word, str) or word.startswith("("):  # a deletion/insertion row has no word to open
        st.session_state["open_note"] = "This row has no word to open."
        return
    if word not in lexicon_search_index(dataset, regions, include_preterite, side)[0].index:
        st.session_state["open_note"] = f"**{word}** has no data as a {SEARCH_SIDES[side]} word with the current filters."
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
    st.subheader("Example words")
    # Let the user pick which kind of variation the example words should illustrate (DIAL-25).
    col_type, *_ = control_row()
    example_type = col_type.radio(
        "Example type",
        options=list(EXAMPLE_WORDS),
        horizontal=True,
        key="example_word_type",
        persist_state="session",
        help="Lexical = a different word with the same meaning (krankenhaus -> spital). "
             "Phonetic = the dialect pronunciation is transcribed as a similar-sounding word "
             "(lösung -> lesung in Wallis, abend -> oben in Basel).",
    )

    # Only offer words the current dataset/region/preterite filter can show; the search box rejects others.
    example_words = [
        w for w in dict.fromkeys(EXAMPLE_WORDS[example_type] + st.session_state.get("custom_example_words", []))
        if w in ref_counts.index
    ]
    col_examples, col_add = st.columns([5, 1], vertical_alignment="bottom")
    with col_examples:
        st.pills(
            "Pick an example word",
            options=example_words,
            key="example_word_pill",
            on_change=_select_example_word,
            wrap=True,
            help="Predefined reference words for the selected example type. Add your own with the button on the right.",
        )
    with col_add:
        with st.popover("+ Add word", help="Add your own reference words to the example list."):
            st.multiselect(
                "Your example words",
                options=ref_counts.index.tolist(),
                key="custom_example_words",
                persist_state="session",
                placeholder="Add a reference word...",
                help="Kept for this browser session only.",
            )


def render_word_cloud(word_counts: pd.Series, regions: list[str], include_preterite: bool, dataset: str,
                      side: SearchSide) -> None:
    """Word cloud of regionally distinctive pairs, as a way to discover interesting words."""
    st.subheader("Regionally typical word pairs")
    col_mode, *_ = control_row()
    mode_label = col_mode.radio(
        "Compare",
        options=["Reference → DIT", "DAT → DIT"],
        horizontal=True,
        key="cloud_mode",
        persist_state="session",
        help="Which two transcripts form the pairs: the reference against Whisper (DIT), or the FHNW model "
             "(DAT) against DIT.",
    )
    mode: CloudMode = "ref_dit" if mode_label.startswith("Reference") else "dat_dit"

    table = overview.compute_top_table(regions, include_preterite, mode, dataset)

    if table.empty:
        st.warning("No regionally distinctive pairs in the selected regions.")
        return

    overview.render_word_cloud(table, on_click=lambda: _open_clicked_pair(word_counts.index, side))
    overview.render_caption(mode)
    overview.render_top_candidates_expander(table, mode)


def render_analysis(word: str, side: SearchSide, regions: list[str], include_preterite: bool, dataset: str) -> None:
    """Analysis tabs for one reference or DIT word; only the open tab does any work."""
    open_note = st.session_state.pop("open_note", None)
    if open_note:
        st.toast(open_note)
    with st.spinner("Loading word data..."):
        sentence_rows = word_sentence_alignments(word, tuple(regions), dataset, include_preterite, side)
    word_rows = word_alignments(sentence_rows, word, side)
    ctx = detail.WordContext(
        word, word_rows, sentence_rows, regions, include_preterite, dataset, side,
        open_word=lambda w, s: _open_word(w, s, dataset, tuple(regions), include_preterite),
    )

    title = f"DIT word: {word}" if side == "dit" else word
    tabs = st.tabs(list(ANALYSIS_VIEWS), key="analysis_tab", on_change="rerun", bind="query-params")
    for tab, (render_view, descriptions) in zip(tabs, ANALYSIS_VIEWS.values()):
        if tab.open:
            with tab:
                tab_header(title, descriptions[side])
                render_view(ctx)


# --- Page 1 ---
dataset, include_preterite = shared_filters()
selected_regions = render_region_filter()

# The word search sits at the top of the sidebar, above the filters, so the analysis tabs can start at the
# top of the page. It runs after the filters because its options depend on them.
word_slot = _filters.word_slot
search_side: SearchSide = word_slot.radio(
    "Search by",
    options=list(SEARCH_SIDES),
    format_func=SEARCH_SIDES.get,
    horizontal=True,
    key=SEARCH_SIDE_KEY,
    bind="query-params",
    help="Reference = a word of the reference sentence, to see what Whisper (DIT) and DAT wrote for it. "
         "DIT = a word Whisper wrote, to see which reference words it was written for.",
)
if search_side not in SEARCH_SIDES:  # e.g. a hand-edited URL
    search_side = "reference"

with st.spinner("Loading alignment data..."):
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
        f"**{previous_word}** is not a {SEARCH_SIDES[search_side]} word with the current filters." if side_changed
        else f"**{previous_word}** has no data with the current filters."
    )

# On a fresh page load, a word from the URL isn't in the session state until the search box renders.
requested_word = st.session_state.get(WORD_KEY, st.query_params.get(WORD_KEY, ""))
analysis_open = requested_word in word_counts.index

search_label, search_help = SEARCH_BOX[search_side]
selected_word = word_slot.selectbox(
    search_label,
    options=[""] + word_counts.index.tolist(),
    key=WORD_KEY,
    bind="query-params",
    placeholder="Type a word…",
    help=search_help,
)
if analysis_open:
    word_slot.button("← Back to word selection", on_click=_clear_word, width="stretch",
                      help="Close this word's analysis and return to the example words and the word cloud.")
word_slot.divider()

if st.session_state.get("_prev_selected_word") != selected_word:
    for k in [k for k in st.session_state if k.startswith("audio_loaded_")]:
        del st.session_state[k]
    st.session_state["_prev_selected_word"] = selected_word

if analysis_open:
    render_analysis(selected_word, search_side, selected_regions, include_preterite, dataset)
else:
    st.title("Dialect Word Analysis")
    if search_side == "dit":
        st.caption("Pick a DIT word in the sidebar or click a pair in the word cloud to see which reference words "
                   "Whisper wrote it for.")
    else:
        st.caption("Pick a word in the sidebar, click an example word or click a pair in the word cloud to "
                   "analyse it.")
        # The example words are reference words, so they are only offered in reference search.
        render_example_words(word_counts)

    if search_side == "dit":
        sidebar_counts(dit_words=len(word_counts), alignments=n_alignments, sentences=n_sentences)
    else:
        sidebar_counts(reference_words=len(word_counts), alignments=n_alignments, sentences=n_sentences)

    filtered_out = st.session_state.pop("filtered_out_word", None)
    if filtered_out:
        st.info(filtered_out)
    cloud_note = st.session_state.pop("cloud_note", None)
    if cloud_note:
        st.info(f"**{cloud_note}** has no data as a DIT word with the current filters." if search_side == "dit"
                else f"**{cloud_note}** is a DAT transcription, not a reference word, so it has no analysis.")

    render_word_cloud(word_counts, selected_regions, include_preterite, dataset, search_side)
