"""Layout pieces shared by every page and tab, so controls and headings sit in the same place everywhere."""
import streamlit as st

# Every row of controls above a chart or table uses this many equally wide slots.
CONTROL_COLUMNS = 3


def control_row() -> list:
    """One row of equally wide control slots; a row with fewer controls leaves the remaining slots empty.

    Aligned at the top, so the labels of a row sit on one line even when the controls differ in height.
    """
    return st.columns(CONTROL_COLUMNS, gap="medium", vertical_alignment="top")


def control_stack():
    """Left half of the page for an analysis tab's controls, stacked one below the other.

    Use as `with control_stack():`; the chart or table below it keeps the full width.
    """
    column, _ = st.columns(2, gap="medium")
    return column


def tab_header(word: str, description: str) -> None:
    """The analysed word and one sentence on what the open tab shows."""
    st.markdown(f"### {word}")
    st.caption(description)


def sidebar_counts(*counts: str) -> None:
    """Data size for the current filters as one muted line in the sidebar, e.g.
    sidebar_counts(t("counts.alignments", n=number(1200)))."""
    st.sidebar.caption(" · ".join(counts))
