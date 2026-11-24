"""Short help text placed directly under a chart or table, in the same format everywhere."""
import streamlit as st

from i18n import t


def chart_help(shows: str, use: str | None = None) -> None:
    """What the element above shows and, optionally, how to use it. Keep each part to one or two sentences."""
    text = f"**{t('help.shows')}** {shows}"
    if use:
        text += f"  \n**{t('help.use')}** {use}"
    st.caption(text)
