"""D3 renderer for the Dialect Word Analysis Levenshtein ring graph."""
import html
import json
from functools import cache
from pathlib import Path

import streamlit as st

from i18n import t

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
  /* No default page margin, so the graph lines up with the Streamlit content outside the frame. */
  body { margin: 0; }
  #ring-graph { position: relative; font-size: 14px; }
  #ring-graph button, #ring-graph input, #ring-graph select, #ring-graph summary {
    background: var(--control-bg); color: var(--control-text); border: 1px solid var(--control-border);
    border-radius: 4px; padding: 3px 9px; font: inherit;
  }
  /* The controls float over the left of the graph, which the centred rings leave empty on wide screens. */
  #ring-graph .controls {
    position: absolute; top: 0; left: 0; z-index: 5; width: 250px;
    display: flex; flex-direction: column; gap: 12px; align-items: flex-start;
  }
  #ring-graph .controls label { display: flex; flex-direction: column; gap: 4px; }
  #ring-graph .button-row { display: flex; flex-wrap: wrap; gap: 6px; }
  #ring-graph button { cursor: pointer; }
  #ring-graph button:disabled { opacity: 0.45; cursor: default; }
  #ring-graph summary { cursor: pointer; }
  /* Same look as Streamlit's (?) help icon; the text shows on hover. */
  #ring-graph .help {
    display: inline-flex; align-items: center; justify-content: center; width: 14px; height: 14px; margin-left: 6px;
    border: 1px solid var(--control-text); border-radius: 50%; font-size: 10px; line-height: 1; opacity: 0.6;
    cursor: help; vertical-align: middle;
  }
  #ring-graph #hidden-menu { position: relative; }
  #ring-graph .hidden-panel {
    position: absolute; z-index: 10; top: calc(100% + 4px); left: 0; min-width: 170px; max-height: 300px;
    overflow-y: auto; padding: 8px; background: var(--control-bg); border: 1px solid var(--control-border);
    border-radius: 4px; box-shadow: 0 2px 6px #0003;
  }
  #ring-graph #hidden-list { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }
  /* Label above "− n +". */
  #ring-graph .stepper { display: grid; grid-template-columns: auto auto 1fr; gap: 4px; justify-items: start; }
  #ring-graph .stepper span { grid-column: 1 / -1; }
  #ring-graph .stepper input { width: 3.2em; text-align: center; appearance: textfield; -moz-appearance: textfield; }
  #ring-graph .stepper input::-webkit-outer-spin-button,
  #ring-graph .stepper input::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
  #ring-graph svg { display: block; width: 100%; height: GRAPH_HEIGHTpx; }
  /* Too narrow to float the controls beside the rings: put them above the graph instead. */
  @media (max-width: 1000px) {
    #ring-graph .controls { position: static; width: auto; flex-direction: row; flex-wrap: wrap; align-items: flex-end;
                            margin-bottom: 8px; }
  }
</style>
"""

# Height of the graph itself; small enough that the graph and its help text fit on a laptop screen.
_GRAPH_HEIGHT = 540
# Extra room for the controls, which sit above the graph on narrow screens.
_FRAME_HEIGHT = _GRAPH_HEIGHT + 20


@cache
def _d3_source() -> str:
    return _D3_PATH.read_text(encoding="utf-8")


def _graph_script() -> str:
    return _SCRIPT_PATH.read_text(encoding="utf-8")


def _to_js(value) -> str:
    """JSON for embedding inside a <script> tag."""
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")


# Texts the graph script writes itself; `{name}` placeholders are filled in there.
_SCRIPT_TEXTS = ("no_variants", "summary", "edge_tooltip", "hidden", "show_again", "png_failed", "ring_distance",
                 "node_distance", "node_alignments", "node_similarity", "node_regions")


def _attr(key: str) -> str:
    """A translated text, escaped for an HTML attribute or element."""
    return html.escape(t(key), quote=True)


def _help_icon(key: str) -> str:
    """Streamlit-like (?) icon that shows the translated text on hover."""
    return f'<span class="help" title="{_attr(key)}" aria-label="{_attr(key)}">?</span>'


def render(graphs: dict[str, dict], rank_labels: dict[str, str], dark: bool = False) -> None:
    """Render a concentric-ring graph in an isolated component.

    Nodes are held on their ring by `forceRadial`; pairwise link forces pull
    lexically close variants together along the ring. The simulation runs to
    completion before drawing, so the layout is static and deterministic.

    `graphs` holds one graph per ranking (keys of `rank_labels`), each with every
    selectable variant; the iframe picks the ranking and how many variants to show,
    so neither needs a Streamlit rerun.
    """
    colors = _DARK if dark else _LIGHT
    rank_options = "".join(f'<option value="{key}">{html.escape(label)}</option>'
                           for key, label in rank_labels.items())

    st.iframe(
        f"""
{_CONTROL_STYLE.replace("GRAPH_HEIGHT", str(_GRAPH_HEIGHT))}
<div id="ring-graph" style="font-family: sans-serif; width: 100%; color: {colors['text']};
     --control-bg: {colors['tooltip_bg']}; --control-border: {colors['tooltip_border']}; --control-text: {colors['text']};">
  <div class="controls">
    <label>
      <span>{_attr("graph.rank_by")}{_help_icon("graph.rank_by_help")}</span>
      <select id="rank-by">{rank_options}</select>
    </label>
    <div class="stepper" role="group" aria-label="{_attr("graph.variant_count")}">
      <span>{_attr("graph.variant_count")}{_help_icon("graph.variant_count_help")}</span>
      <button id="variant-count-decrease" type="button" aria-label="{_attr("graph.decrease")}">−</button>
      <input id="variant-count" type="number" min="1" max="1" value="1" step="1">
      <button id="variant-count-increase" type="button" aria-label="{_attr("graph.increase")}">+</button>
    </div>
    <div class="stepper" role="group" aria-label="{_attr("graph.edge_distance")}">
      <span>{_attr("graph.edge_distance")}{_help_icon("graph.edge_distance_help")}</span>
      <button id="edge-distance-decrease" type="button" aria-label="{_attr("graph.decrease")}">−</button>
      <input id="edge-distance" type="number" min="1" max="1" value="1" step="1">
      <button id="edge-distance-increase" type="button" aria-label="{_attr("graph.increase")}">+</button>
    </div>
    <div class="button-row">
      <button id="hide-selected" type="button" disabled title="{_attr("graph.hide_help")}">{_attr("graph.hide")}</button>
      <details id="hidden-menu" hidden>
        <summary id="hidden-summary"></summary>
        <div class="hidden-panel">
          <button id="show-all-hidden" type="button">{_attr("graph.show_all")}</button>
          <div id="hidden-list"></div>
        </div>
      </details>
    </div>
    <div class="button-row">
      <button id="reset-layout" type="button" title="{_attr("graph.reset_help")}">{_attr("graph.reset")}</button>
      <button id="export-png" type="button" title="{_attr("graph.export_help")}">{_attr("graph.export")}</button>
    </div>
    <div id="variant-summary" style="color:{colors['muted']};"></div>
    <div id="graph-message" style="color:{colors['muted']};"></div>
  </div>
  <svg id="graph-svg" viewBox="0 0 900 620" role="img" aria-label="{_attr("graph.aria")}"></svg>
</div>
<script>{_d3_source()}</script>
<script>
const graphsByRank = {_to_js(graphs)};
const rankLabels = {_to_js(rank_labels)};
const texts = {_to_js({key: t(f"graph.{key}") for key in _SCRIPT_TEXTS})};
const colors = {_to_js(colors)};
</script>
<script>{_graph_script()}</script>
""",
        height=_FRAME_HEIGHT,
    )
