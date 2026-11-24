"""Translations of the user-facing texts (DIAL-30).

The texts live in `locales/<language>.json`, nested by page and topic. `t("overview.occurrences")` looks the
dotted key up in the language chosen in the sidebar, falls back to English when the key is missing there, and
fills in `{placeholders}` from its keyword arguments. A key missing in English too comes back as the key itself,
so a forgotten text shows up on the page instead of crashing it.
"""
import json
import string
from functools import cache
from pathlib import Path

import streamlit as st

LOCALES_DIR = Path(__file__).resolve().parent / "locales"
LANGUAGES = {"en": "English", "de": "Deutsch"}
DEFAULT_LANGUAGE = "en"
LANGUAGE_KEY = "language"


def _flatten(tree: dict, prefix: str = "") -> dict[str, str]:
    """{"a": {"b": "text"}} -> {"a.b": "text"}."""
    flat = {}
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{path}."))
        else:
            flat[path] = value
    return flat


@cache
def translations(language: str) -> dict[str, str]:
    """All texts of `language` by dotted key; empty for a language without a file. Read once per process."""
    path = LOCALES_DIR / f"{language}.json"
    if not path.exists():
        return {}
    return _flatten(json.loads(path.read_text(encoding="utf-8")))


def placeholders(text: str) -> set[str]:
    """The `{name}` placeholders in a text."""
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def current_language() -> str:
    """The language chosen in the sidebar, or English before the selector has rendered."""
    language = st.session_state.get(LANGUAGE_KEY, DEFAULT_LANGUAGE)
    return language if language in LANGUAGES else DEFAULT_LANGUAGE


def translate(key: str, language: str, **params) -> str:
    """`key` in `language`, falling back to English and then to the key itself."""
    for candidate in (language, DEFAULT_LANGUAGE):
        text = translations(candidate).get(key)
        if text is None:
            continue
        try:
            return text.format(**params) if params else text
        except (KeyError, IndexError, ValueError):
            continue  # a broken placeholder in a translation: use the English text instead
    return key


def t(key: str, **params) -> str:
    """`key` in the current language, e.g. t("counts.alignments", n=number(120))."""
    return translate(key, current_language(), **params)


def all_translations(key: str) -> set[str]:
    """`key` in every language, to recognise a value stored while another language was active."""
    return {translate(key, language) for language in LANGUAGES}


def number(value: int) -> str:
    """A whole number with thousands separators: 12,345 in English, 12'345 (Swiss style) in German."""
    text = f"{value:,}"
    return text.replace(",", "'") if current_language() == "de" else text


_LOCALIZED_KEYS = "_localized_widget_keys"


def localized(key: str) -> str:
    """Register the key of a widget whose options are shown through a translated `format_func`; returns the key.

    The browser remembers such a widget's choice by its displayed text, so after a language switch a radio would
    show nothing selected and a multiselect its old labels, although the value itself is unchanged. The language
    switch therefore sends these widgets' values to the browser again (`_resend_localized_widgets`).
    """
    st.session_state.setdefault(_LOCALIZED_KEYS, set()).add(key)
    return key


def _resend_localized_widgets() -> None:
    """Language switch callback: assigning a widget's own value makes Streamlit send it to the browser again."""
    for key in st.session_state.get(_LOCALIZED_KEYS, ()):
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def render_language_selector() -> None:
    """English / Deutsch switch at the top of the sidebar. English is the default; the choice is kept in the
    session state, so it stays the same on every page."""
    st.sidebar.segmented_control(
        "Language / Sprache",
        options=list(LANGUAGES),
        format_func=LANGUAGES.get,
        default=DEFAULT_LANGUAGE,
        required=True,
        key=LANGUAGE_KEY,
        persist_state="session",
        on_change=_resend_localized_widgets,
    )
