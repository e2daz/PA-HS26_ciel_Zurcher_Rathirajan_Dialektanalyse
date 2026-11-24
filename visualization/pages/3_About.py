"""About: what the tool does and which data it uses."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from i18n import t  # noqa: E402

st.title(t("about.title"))

st.warning(t("about.audio_note"))

st.markdown(t("about.body"))
