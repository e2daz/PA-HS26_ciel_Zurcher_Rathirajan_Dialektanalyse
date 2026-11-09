"""Short help text placed directly under a chart or table, in the same format everywhere."""
import streamlit as st


def chart_help(shows: str, use: str | None = None) -> None:
    """What the element above shows and, optionally, how to use it. Keep each part to one or two sentences."""
    text = f"**What it shows:** {shows}"
    if use:
        text += f"  \n**How to use:** {use}"
    st.caption(text)
