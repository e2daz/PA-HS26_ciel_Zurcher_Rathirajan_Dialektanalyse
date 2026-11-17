"""Layout pieces shared by every page and tab, so controls and headings sit in the same place everywhere."""
import streamlit as st

# Every row of controls above a chart or table uses this many equally wide slots.
CONTROL_COLUMNS = 3


def control_row() -> list:
    """One row of equally wide control slots; a row with fewer controls leaves the remaining slots empty."""
    return st.columns(CONTROL_COLUMNS, gap="medium", vertical_alignment="bottom")


def tab_header(word: str, description: str) -> None:
    """The analysed word and one sentence on what the open tab shows."""
    st.markdown(f"### {word}")
    st.caption(description)


def sidebar_counts(**counts: int) -> None:
    """Data size for the current filters as one muted line in the sidebar, e.g. sidebar_counts(alignments=1200)."""
    st.sidebar.caption(" · ".join(f"{value:,} {name.replace('_', ' ')}" for name, value in counts.items()))
