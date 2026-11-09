"""Sidebar filters shared across the app."""
import streamlit as st

from _data import DATASET_CHOICES, DEFAULT_DATASET, REGION_COLORS, REGIONS

_DATASET_KEY = "selected_dataset"
_PRETERITE_KEY = "include_preterite"


def render_shared_filters() -> None:
    """Dataset and Preterite controls, rendered once by the entry script so every page sees the same values.

    `persist_state` keeps the values while a page that hides the filters (About) is open.
    """
    st.sidebar.selectbox(
        "Dataset",
        DATASET_CHOICES,
        index=DATASET_CHOICES.index(DEFAULT_DATASET),
        key=_DATASET_KEY,
        persist_state="session",
        help="Switch between STT4SG-350, SDS-200, or All (both corpora combined).",
    )
    st.sidebar.toggle(
        "Include Preterite sentences",
        value=False,
        key=_PRETERITE_KEY,
        persist_state="session",
        help="As Swiss German does not have the Preterite tense and replaces it with the Perfect, including such "
             "cases can introduce noise in the transcripts. Exclude them to focus on dialect specifics; include "
             "for more data.",
    )


def shared_filters() -> tuple[str, bool]:
    """The current (dataset, include_preterite) selection."""
    return st.session_state[_DATASET_KEY], st.session_state[_PRETERITE_KEY]


def render_region_filter() -> list[str]:
    """Region multiselect for the lexicon; returns the selected regions, or all of them when empty."""
    region_filter = st.sidebar.multiselect(
        "Filter regions",
        REGIONS,
        default=[],
        key="region_filter",
        persist_state="session",
        help="Leave empty to include all 7 regions. Pick specific regions to narrow the view.",
        placeholder="All regions",
    )
    chip_css = "\n".join(
        f'.stMultiSelect [data-baseweb="tag"][aria-label*="{region}"],'
        f'.stMultiSelect [data-baseweb="tag"]:has([aria-label*="{region}"])'
        f' {{ background-color: {color} !important; color: white !important; }}'
        for region, color in REGION_COLORS.items()
    )
    st.markdown(f"<style>{chip_css}</style>", unsafe_allow_html=True)
    return region_filter if region_filter else REGIONS
