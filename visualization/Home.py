"""
Swiss German Dialect Analysis: entry script. Sets up navigation and the filters shared by all pages.

Run: streamlit run visualization/Home.py
"""
import streamlit as st

import _filters

st.set_page_config(page_title="Swiss German Dialect Analysis", layout="wide")

# Streamlit reserves an empty header row above the page and 10rem below it. Instead, the header becomes
# see-through (only its buttons take clicks), so the analysis tabs sit in that row and stay there while
# scrolling, and a page without more content ends where its content ends.
# The sticky tab bar needs an opaque background so content scrolls behind it, in the app's theme colour.
tab_bar_background = "#0e1117" if st.context.theme.type == "dark" else "#ffffff"
st.markdown(f"""<style>
[data-testid='stMainBlockContainer'] {{ padding-top: 0.75rem; padding-bottom: 2rem; }}
[data-testid='stHeader'] {{ background: transparent; }}
[data-testid='stHeader'], [data-testid='stHeader'] * {{ pointer-events: none !important; }}
[data-testid='stHeader'] button, [data-testid='stHeader'] button *, [data-testid='stHeader'] a {{
  pointer-events: auto !important;
}}
/* Blocks that only carry CSS would still take a row gap each, pushing the page down. */
[data-testid='stElementContainer']:has(> [data-testid='stMarkdown'] [data-testid='stMarkdownContainer'] > style:only-child),
[data-testid='stElementContainer']:has(> [data-testid='stHtml'] > style:only-child) {{ display: none; }}
.stTabs [role='tablist'] {{
  position: sticky; top: 18px; z-index: 100; background: {tab_bar_background};
}}
</style>""", unsafe_allow_html=True)

about = st.Page("pages/3_About.py", title="About")
# The pages are listed above the filters in the sidebar, so the top of the page is free for the analysis tabs.
page = st.navigation([
    st.Page("pages/1_Dialect_Word_Analysis.py", title="Dialect Word Analysis", default=True),
    st.Page("pages/2_Regional_Distance.py", title="Regional Distance"),
    about,
], position="sidebar")

if page.title != about.title:
    # The word search goes first in the sidebar; the analysis page fills this slot after the filters exist.
    _filters.word_slot = st.sidebar.container()
    _filters.render_shared_filters()

page.run()
