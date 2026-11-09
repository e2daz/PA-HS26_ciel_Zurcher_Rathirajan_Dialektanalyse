"""
Dialect Word Lexicon: the app's start page.

Word selection first (search, example words, word cloud). Choosing a word opens its analysis tabs.

Run via the entry script: streamlit run visualization/Home.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pages.render_helpers import _lexicon_detail as detail, _lexicon_overview as overview
from _data import CloudMode, lexicon_search_index, word_sentence_alignments  # noqa: E402
from _filters import render_region_filter, shared_filters  # noqa: E402

# Reference words offered as one-click examples, grouped by the kind of variation they show
# (DIAL-25). Checked against the DIT alignments of both corpora.
EXAMPLE_WORDS = {
    # Lexical: the dialect uses a different word with the same meaning, which Whisper transcribes
    # literally (krankenhaus -> spital, sprechen -> reden, deshalb -> darum, zuvor -> vorher).
    "Lexical (different word)": [
        "fahrrad", "krankenhaus", "arbeiten", "hause", "sprechen",
        "junge", "deshalb", "nun", "selbst", "zuvor",
    ],
    # Phonetic: Whisper writes Standard German, so a dialect pronunciation shows up as a
    # sound-alike word rather than a dialect spelling. Strongly regional: lösung -> lesung
    # (Wallis 58 %, elsewhere <= 4 %), abend -> oben (Basel 28 %), haben -> habt (Wallis, Bern).
    # Broader: denn -> dann, wann -> wenn, darin -> drin.
    "Phonetic (sound-alike word)": [
        "lösung", "abend", "haben", "denn", "wann", "darin",
    ],
}

# One entry per tab; a new analysis only needs a render function taking a WordContext.
ANALYSIS_VIEWS = {
    "Overview": detail.render_overview,
    "Levenshtein": detail.render_levenshtein_graph,
    "Regions": detail.render_regional_distribution,
    "Occurrences": detail.render_example_sentences,
}

WORD_KEY = "selected_word"


def _clear_word() -> None:
    st.session_state[WORD_KEY] = ""


def _select_example_word() -> None:
    """Load the clicked example into the search box, then clear the chip so it can be clicked again."""
    word = st.session_state.get("example_word_pill")
    if word:
        st.session_state[WORD_KEY] = word
    st.session_state["example_word_pill"] = None


def _open_clicked_pair(known_words: pd.Index) -> None:
    """Open the word on the left of the clicked 'word → DIT word' pair, if it is a reference word."""
    word = st.session_state[overview.WORD_CLOUD_KEY]["chart_event"]["name"].split(" → ", 1)[0]
    if word in known_words:
        st.session_state[WORD_KEY] = word
    else:
        st.session_state["cloud_note"] = word


def render_example_words(ref_counts: pd.Series) -> None:
    """Example-type choice and the example chips it offers, plus the user's own additions."""
    # Let the user pick which kind of variation the example words should illustrate (DIAL-25).
    example_type = st.radio(
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
            "Or pick an example word",
            options=example_words,
            key="example_word_pill",
            on_change=_select_example_word,
            wrap=True,
            help="Predefined reference words for the selected example type. Add your own with the button on the right.",
        )
    with col_add:
        with st.popover("+ Add"):
            st.multiselect(
                "Your example words",
                options=ref_counts.index.tolist(),
                key="custom_example_words",
                persist_state="session",
                placeholder="Add a reference word...",
                help="Kept for this browser session only.",
            )


def render_word_cloud(ref_counts: pd.Series, regions: list[str], include_preterite: bool, dataset: str) -> None:
    """Word cloud of regionally distinctive pairs, as a way to discover interesting words."""
    mode_label = st.radio(
        "Compare",
        options=["Reference → DIT (Whisper)", "DAT (FHNW) → DIT (Whisper)"],
        horizontal=True,
        key="cloud_mode",
        persist_state="session",
    )
    mode: CloudMode = "ref_dit" if mode_label.startswith("Reference") else "dat_dit"

    table = overview.compute_top_table(regions, include_preterite, mode, dataset)

    if table.empty:
        st.warning("No regionally distinctive pairs in the selected regions.")
        return

    overview.render_word_cloud(table, on_click=lambda: _open_clicked_pair(ref_counts.index))
    overview.render_caption(mode)
    overview.render_top_candidates_expander(table, mode)


def render_analysis(word: str, regions: list[str], include_preterite: bool, dataset: str) -> None:
    """Analysis tabs for one reference word; only the open tab does any work."""
    with st.spinner("Loading word data..."):
        sentence_rows = word_sentence_alignments(word, tuple(regions), dataset, include_preterite)
    word_rows = sentence_rows[(sentence_rows["reference_word"] == word) & (sentence_rows["model"] != "dat-dit")]
    ctx = detail.WordContext(word, word_rows, sentence_rows, regions, include_preterite, dataset)

    tabs = st.tabs(list(ANALYSIS_VIEWS), key="analysis_tab", on_change="rerun", bind="query-params")
    for tab, render_view in zip(tabs, ANALYSIS_VIEWS.values()):
        if tab.open:
            with tab:
                render_view(ctx)


# --- Page 1 ---
dataset, include_preterite = shared_filters()
selected_regions = render_region_filter()

with st.spinner("Loading alignment data..."):
    # Cached per (dataset, regions, preterite): ref-word frequencies + sidebar counts.
    ref_counts, n_alignments, n_sentences = lexicon_search_index(
        dataset, tuple(selected_regions), include_preterite
    )

# A filter change can remove the selected word from the search index; go back and say why.
previous_word = st.session_state.get(WORD_KEY)
if previous_word and previous_word not in ref_counts.index:
    st.session_state[WORD_KEY] = ""
    st.session_state["filtered_out_word"] = previous_word

# On a fresh page load, a word from the URL isn't in the session state until the search box renders.
requested_word = st.session_state.get(WORD_KEY, st.query_params.get(WORD_KEY, ""))
analysis_open = requested_word in ref_counts.index

if analysis_open:
    col_back, col_word = st.columns([1, 3], vertical_alignment="bottom")
    col_back.button("← Back to word selection", on_click=_clear_word, width="stretch")
else:
    st.title("Dialect Word Lexicon")
    col_word, col_examples = st.columns([2, 3])

selected_word = col_word.selectbox(
    "Reference word",
    options=[""] + ref_counts.index.tolist(),
    key=WORD_KEY,
    bind="query-params",
    help="Type a Standard German word to see its dialect-specific variants.",
)

if st.session_state.get("_prev_selected_word") != selected_word:
    for k in [k for k in st.session_state if k.startswith("audio_loaded_")]:
        del st.session_state[k]
    st.session_state["_prev_selected_word"] = selected_word

if analysis_open:
    render_analysis(selected_word, selected_regions, include_preterite, dataset)
else:
    with col_examples:
        render_example_words(ref_counts)

    st.sidebar.metric("Reference words", f"{len(ref_counts):,}")
    st.sidebar.metric("Alignments", f"{n_alignments:,}")
    st.sidebar.metric("Unique sentences", f"{n_sentences:,}")

    filtered_out = st.session_state.pop("filtered_out_word", None)
    if filtered_out:
        st.info(f"**{filtered_out}** has no data with the current filters.")
    cloud_note = st.session_state.pop("cloud_note", None)
    if cloud_note:
        st.info(f"**{cloud_note}** is a DAT transcription, not a reference word, so it has no analysis.")

    render_word_cloud(ref_counts, selected_regions, include_preterite, dataset)
