"""
Dialect Word Lexicon: Page 1.

Search a Standard German reference word, compare DIT and DAT across regions.

Run via the entry script: streamlit run visualization/Home.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pages.render_helpers import _lexicon_detail as detail, _lexicon_overview as overview
from _data import (  # noqa: E402
    DATASET_CHOICES, DEFAULT_DATASET, REGION_COLORS, REGIONS,
    CloudMode, filter_preterite, load_region_alignments_and_metadata, lexicon_search_index,
)

# Reference words with interesting DIT/DAT differences, offered as one-click examples.
# Phonetic (abend → oben, haben → hat/habt), lexical (sprechen → reden, deshalb → darum,
# nun → jetzt, zuvor → vorher, selbst → selber, hause → daheim) and grammatical
# (habe → hat/hätte: Swiss German avoids Konjunktiv I). Chosen from words with >= 300
# alignments where DIT deviates from the reference far more often than DAT.
EXAMPLE_WORDS = ["abend", "haben", "sprechen", "deshalb", "nun", "zuvor", "selbst", "hause", "habe"]


# --- Page 1 ---
st.set_page_config(page_title="Dialect Word Lexicon", layout="wide")
st.title("Dialect Word Lexicon")

# Sidebar filters
dataset = st.sidebar.selectbox(
    "Dataset",
    DATASET_CHOICES,
    index=DATASET_CHOICES.index(DEFAULT_DATASET),
    key="selected_dataset",
    help="Switch between STT4SG-350, SDS-200, or All (both corpora combined).",
)

region_filter = st.sidebar.multiselect(
    "Filter regions",
    REGIONS,
    default=[],
    help="Leave empty to include all 7 regions. Pick specific regions to narrow the view.",
    placeholder="All regions",
)
selected_regions = region_filter if region_filter else REGIONS

_chip_css = "\n".join(
    f'.stMultiSelect [data-baseweb="tag"][aria-label*="{region}"],'
    f'.stMultiSelect [data-baseweb="tag"]:has([aria-label*="{region}"])'
    f' {{ background-color: {color} !important; color: white !important; }}'
    for region, color in REGION_COLORS.items()
)
st.markdown(f"<style>{_chip_css}</style>", unsafe_allow_html=True)

include_preterite = st.sidebar.toggle(
    "Include Preterite sentences",
    value=True,
    help="As Swiss German does not have the Preterite tense and replaces it with the Perfect, including such cases "
         "can introduce noise in the transcripts. Exclude them to focus on dialect specifics; include for more data.")

with st.spinner("Loading alignment data..."):
    # Cached per (dataset, regions, preterite): ref-word frequencies + sidebar counts.
    ref_counts, n_alignments, n_sentences = lexicon_search_index(
        dataset, tuple(selected_regions), include_preterite
    )

st.sidebar.metric("Alignments", f"{n_alignments:,}")
st.sidebar.metric("Unique sentences", f"{n_sentences:,}")

# Search box
search_options = [""] + ref_counts.index.tolist()
selected_word = st.selectbox(
    f"Search a reference word ({len(ref_counts):,} unique) to see its dialect specific variants.",
    options=search_options,
    key="selected_word",
    placeholder="Start typing a Standard German word...",
)


def _select_example_word() -> None:
    """Load the clicked example into the search box, then clear the chip so it can be clicked again."""
    word = st.session_state.get("example_word_pill")
    if word:
        st.session_state["selected_word"] = word
    st.session_state["example_word_pill"] = None


# Only offer words the current dataset/region/preterite filter can show; the search box rejects others.
example_words = [
    w for w in dict.fromkeys(EXAMPLE_WORDS + st.session_state.get("custom_example_words", []))
    if w in ref_counts.index
]
col_examples, col_add = st.columns([6, 1], vertical_alignment="bottom")
with col_examples:
    st.pills(
        "Or pick an example word",
        options=example_words,
        key="example_word_pill",
        on_change=_select_example_word,
        help="Predefined reference words with interesting phonetic and lexical differences "
             "between the transcripts. Add your own with the button on the right.",
    )
with col_add:
    with st.popover("+ Add"):
        st.multiselect(
            "Your example words",
            options=ref_counts.index.tolist(),
            key="custom_example_words",
            placeholder="Add a reference word...",
            help="Kept for this browser session only.",
        )

if st.session_state.get("_prev_selected_word") != selected_word:
    for k in [k for k in st.session_state if k.startswith("audio_loaded_")]:
        del st.session_state[k]
    st.session_state["_prev_selected_word"] = selected_word


def render_detail(df_view: pd.DataFrame, word: str, regions: list[str],
                  include_preterite: bool, dataset: str) -> None:
    """Per-reference-word detail view: header, hypothesis tables, charts, regional table, examples."""
    word_rows = df_view[(df_view["reference_word"] == word) & (df_view["model"] != "dat-dit")]

    detail.render_header(word, word_rows, regions)
    detail.render_word_chart(word_rows)
    detail.render_levenshtein_graph(word, word_rows, regions, include_preterite, dataset)
    detail.render_hypothesis_tables(word, word_rows, regions, include_preterite, dataset)
    detail.render_example_sentences(df_view, word_rows, word, regions, include_preterite, dataset)


def render_overview(regions: list[str], include_preterite: bool, dataset: str) -> None:
    """Word-cloud overview of regionally distinctive substitution pairs."""
    mode_label = st.radio(
        "Compare",
        options=["Reference → DIT (Whisper)", "DAT (FHNW) → DIT (Whisper)"],
        horizontal=True,
        key="cloud_mode",
    )
    mode: CloudMode = "ref_dit" if mode_label.startswith("Reference") else "dat_dit"

    overview.render_caption(mode)
    table = overview.compute_top_table(regions, include_preterite, mode, dataset)

    if table.empty:
        st.warning("No regionally distinctive pairs in the selected regions.")
        return

    overview.render_word_cloud(table)
    overview.render_top_candidates_expander(table, mode)


if selected_word:
    # Full frame needed only for the detail view; load it lazily here.
    df = load_region_alignments_and_metadata(tuple(selected_regions), dataset, include_dat_dit=True)
    df = filter_preterite(df, include_preterite)
    render_detail(df, selected_word, selected_regions, include_preterite, dataset)
else:
    render_overview(selected_regions, include_preterite, dataset)