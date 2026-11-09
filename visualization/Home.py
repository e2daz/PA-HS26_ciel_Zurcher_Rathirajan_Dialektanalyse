"""
Swiss German Dialect Analysis: entry script. Sets up navigation and the filters shared by all pages.

Run: streamlit run visualization/Home.py
"""
import streamlit as st

from _filters import render_shared_filters

st.set_page_config(page_title="Swiss German Dialect Analysis", layout="wide")

about = st.Page("pages/3_About.py", title="About")
page = st.navigation([
    st.Page("pages/1_Dialect_Word_Lexicon.py", title="Dialect Word Lexicon", default=True),
    st.Page("pages/2_Regional_Distance.py", title="Regional Distance"),
    about,
])

if page.title != about.title:
    render_shared_filters()

page.run()
