"""D3 renderer for the Dialect Word Lexicon Levenshtein ring graph."""
import json
from functools import cache
from pathlib import Path

import streamlit as st

# Vendored (pinned) so the graph renders without network access.
_D3_PATH = Path(__file__).resolve().parent / "vendor" / "d3-7.9.0.min.js"
_SCRIPT_PATH = Path(__file__).with_suffix(".js")

_LIGHT = {
    "text": "#1f2937", "muted": "#64748b", "ring": "#cbd5e1", "edge": "#2563eb",
    "ref_fill": "#fca5a5", "ref_stroke": "#991b1b", "hyp_fill": "#bfdbfe", "hyp_stroke": "#1d4ed8",
    "tooltip_bg": "#ffffff", "tooltip_border": "#bbbbbb",
    "highlight": "#ea580c", "label_halo": "#ffffff",
}
_DARK = {
    "text": "#e5e7eb", "muted": "#94a3b8", "ring": "#475569", "edge": "#60a5fa",
    "ref_fill": "#7f1d1d", "ref_stroke": "#fca5a5", "hyp_fill": "#1e3a8a", "hyp_stroke": "#93c5fd",
    "tooltip_bg": "#1f2937", "tooltip_border": "#4b5563",
    "highlight": "#fb923c", "label_halo": "#0e1117",
}


# Not part of the f-string below, so the CSS braces stay single; colours arrive as CSS variables.
_CONTROL_STYLE = """
<style>
  #ring-graph button, #ring-graph input, #ring-graph summary {
    background: var(--control-bg); color: var(--control-text); border: 1px solid var(--control-border);
    border-radius: 4px; padding: 3px 9px; font: inherit;
  }
  #ring-graph button { cursor: pointer; }
  #ring-graph button:disabled { opacity: 0.45; cursor: default; }
  #ring-graph summary { cursor: pointer; }
  #ring-graph #hidden-menu { position: relative; }
  #ring-graph .hidden-panel {
    position: absolute; z-index: 10; top: calc(100% + 4px); left: 0; min-width: 170px; max-height: 300px;
    overflow-y: auto; padding: 8px; background: var(--control-bg); border: 1px solid var(--control-border);
    border-radius: 4px; box-shadow: 0 2px 6px #0003;
  }
  #ring-graph #hidden-list { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }
  #ring-graph .stepper { display: flex; align-items: center; gap: 4px; }
  #ring-graph .stepper span { margin-right: 4px; }
  #ring-graph .stepper input { width: 3.2em; text-align: center; appearance: textfield; -moz-appearance: textfield; }
  #ring-graph .stepper input::-webkit-outer-spin-button,
  #ring-graph .stepper input::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
</style>
"""


@cache
def _d3_source() -> str:
    return _D3_PATH.read_text(encoding="utf-8")


def _graph_script() -> str:
    return _SCRIPT_PATH.read_text(encoding="utf-8")


def _to_js(value) -> str:
    """JSON for embedding inside a <script> tag."""
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")


def render(graph: dict, dark: bool = False) -> None:
    """Render a concentric-ring graph in an isolated component.

    Nodes are held on their ring by `forceRadial`; pairwise link forces pull
    lexically close variants together along the ring. The simulation runs to
    completion before drawing, so the layout is static and deterministic.
    """
    max_edge = max(1, graph.get("max_edge_distance", 1))
    default_edge = graph["default_edge_distance"]
    colors = _DARK if dark else _LIGHT

    st.iframe(
        f"""
{_CONTROL_STYLE}
<div id="ring-graph" style="font-family: sans-serif; width: 100%; color: {colors['text']};
     --control-bg: {colors['tooltip_bg']}; --control-border: {colors['tooltip_border']}; --control-text: {colors['text']};">
  <div style="display:flex; flex-wrap:wrap; gap:16px; align-items:center; margin-bottom:8px;">
    <div class="stepper" role="group" aria-label="Show edges up to distance">
      <span>Show edges up to distance</span>
      <button id="edge-decrease" type="button" aria-label="Decrease edge distance">−</button>
      <input id="edge-distance" type="number" min="1" max="{max_edge}" value="{default_edge}" step="1">
      <button id="edge-increase" type="button" aria-label="Increase edge distance">+</button>
    </div>
    <button id="hide-selected" type="button" disabled>Hide selected</button>
    <details id="hidden-menu" hidden>
      <summary id="hidden-summary"></summary>
      <div class="hidden-panel">
        <button id="show-all-hidden" type="button">Show all</button>
        <div id="hidden-list"></div>
      </div>
    </details>
    <button id="reset-layout" type="button">Reset layout</button>
    <button id="export-png" type="button">Export PNG</button>
  </div>
  <div id="graph-message" style="color:{colors['muted']};"></div>
  <svg id="graph-svg" width="100%" height="620" viewBox="0 0 900 620"
       role="img" aria-label="Levenshtein ring graph"></svg>
</div>
<script>{_d3_source()}</script>
<script>
const graphData = {_to_js(graph)};
const colors = {_to_js(colors)};
</script>
<script>{_graph_script()}</script>
""",
        height=750,
    )
