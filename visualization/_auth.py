"""Audio-playback gate (anti-crawl, not real authentication)."""
import hmac

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


def audio_unlocked() -> bool:
    """True if this session entered the audio password. Pure read — safe to call repeatedly."""
    return bool(st.session_state.get("audio_authed"))


def render_audio_gate() -> None:
    """Render the one-time audio password prompt where audio playback appears. Call once per render."""
    if st.session_state.get("audio_authed"):
        return
    st.caption("Audio is password-protected because the corpora are licensed; automated downloading or crawling "
               "is not permitted.")
    try:
        expected = st.secrets["password"]
    except (KeyError, StreamlitSecretNotFoundError):
        expected = None
    if not isinstance(expected, str) or not expected:
        st.warning("Audio is locked: no password is configured (set `password` in `.streamlit/secrets.toml`).")
        return
    with st.form("audio_gate", clear_on_submit=True):
        password = st.text_input("Enter the password to enable audio playback", type="password")
        submitted = st.form_submit_button("Unlock audio")
    if submitted and hmac.compare_digest(password, expected):
        st.session_state["audio_authed"] = True
        st.rerun()
    elif submitted:
        st.error("Wrong password")
