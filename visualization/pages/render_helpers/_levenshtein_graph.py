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
    edge_distances = sorted(edge["distance"] for edge in graph["edges"])
    default_edge = edge_distances[len(edge_distances) // 2] if edge_distances else max_edge
    colors = _DARK if dark else _LIGHT

    st.iframe(
        f"""
<div id="ring-graph" style="font-family: sans-serif; width: 100%; color: {colors['text']};">
  <div style="display:flex; flex-wrap:wrap; gap:16px; align-items:center; margin-bottom:8px;">
    <label>Show edges up to distance
      <input id="edge-distance" type="range" min="1" max="{max_edge}" value="{default_edge}" step="1">
      <strong id="edge-distance-value">{default_edge}</strong>
    </label>
    <button id="reset-view" type="button">Reset view</button>
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
        height=680,
    )
