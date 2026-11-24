"""Sidebar filters shared across the app."""
import streamlit as st

from _data import ALL_DATASETS, DATASET_CHOICES, DEFAULT_DATASET, REGION_COLORS, REGIONS
from i18n import localized, t

_DATASET_KEY = "selected_dataset"
_PRETERITE_KEY = "include_preterite"

_WORD_SLOT_KEY = "_word_slot"


def reserve_word_slot() -> None:
    """Reserve the sidebar place above the filters for the word search; the entry script calls this before the
    filters render.

    Kept in the session state, not in a module variable: modules are shared by every user of the server, so a
    second user loading the page at the same moment would otherwise swap the containers.
    """
    st.session_state[_WORD_SLOT_KEY] = st.sidebar.container()


def word_slot():
    """The sidebar place reserved by `reserve_word_slot`, or the sidebar itself if none was reserved."""
    return st.session_state.get(_WORD_SLOT_KEY, st.sidebar)


def render_shared_filters() -> None:
    """Dataset and Preterite controls, rendered once by the entry script so every page sees the same values.

    `persist_state` keeps the values while a page that hides the filters (About) is open.
    """
    st.sidebar.selectbox(
        t("filters.dataset"),
        DATASET_CHOICES,
        index=DATASET_CHOICES.index(DEFAULT_DATASET),
        format_func=dataset_label,
        key=localized(_DATASET_KEY),
        persist_state="session",
        help=t("filters.dataset_help"),
    )
    st.sidebar.toggle(
        t("filters.preterite"),
        value=False,
        key=_PRETERITE_KEY,
        persist_state="session",
        help=t("filters.preterite_help"),
    )


def dataset_label(dataset: str) -> str:
    """The corpus name, or "All" in the current language for both corpora combined."""
    return t("filters.all_datasets") if dataset == ALL_DATASETS else dataset


def shared_filters() -> tuple[str, bool]:
    """The current (dataset, include_preterite) selection."""
    return st.session_state[_DATASET_KEY], st.session_state[_PRETERITE_KEY]


def render_region_filter() -> list[str]:
    """Region multiselect for the lexicon; returns the selected regions, or all of them when empty."""
    region_filter = st.sidebar.multiselect(
        t("filters.regions"),
        REGIONS,
        default=[],
        key="region_filter",
        persist_state="session",
        help=t("filters.regions_help"),
        placeholder=t("filters.all_regions"),
    )
    chip_css = "\n".join(
        f'.stMultiSelect [data-baseweb="tag"][aria-label*="{region}"],'
        f'.stMultiSelect [data-baseweb="tag"]:has([aria-label*="{region}"])'
        f' {{ background-color: {color} !important; color: white !important; }}'
        for region, color in REGION_COLORS.items()
    )
    st.markdown(f"<style>{chip_css}</style>", unsafe_allow_html=True)
    return region_filter if region_filter else REGIONS
