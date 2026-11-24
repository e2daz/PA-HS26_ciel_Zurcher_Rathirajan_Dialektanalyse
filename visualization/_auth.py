"""Audio-playback gate (anti-crawl, not real authentication)."""
import hmac

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from i18n import t


def audio_unlocked() -> bool:
    """True if this session entered the audio password. Pure read — safe to call repeatedly."""
    return bool(st.session_state.get("audio_authed"))


def render_audio_gate() -> None:
    """Render the one-time audio password prompt where audio playback appears. Call once per render."""
    if st.session_state.get("audio_authed"):
        return
    st.caption(t("audio.gate_note"))
    try:
        expected = st.secrets["password"]
    except (KeyError, StreamlitSecretNotFoundError):
        expected = None
    if not isinstance(expected, str) or not expected:
        st.warning(t("audio.no_password"))
        return
    with st.form("audio_gate", clear_on_submit=True):
        password = st.text_input(t("audio.password"), type="password", help=t("audio.password_help"))
        submitted = st.form_submit_button(t("audio.unlock"))
    if submitted and hmac.compare_digest(password, expected):
        st.session_state["audio_authed"] = True
        st.rerun()
    elif submitted:
        st.error(t("audio.wrong_password"))
