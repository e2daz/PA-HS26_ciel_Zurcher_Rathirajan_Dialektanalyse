"""D3 renderer for the Dialect Word Lexicon Levenshtein ring graph."""
import json
from functools import cache
from pathlib import Path

import streamlit as st

# Vendored (pinned) so the graph renders without network access.
_D3_PATH = Path(__file__).resolve().parent / "vendor" / "d3-7.9.0.min.js"

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
const svg = d3.select("#graph-svg");
const width = 900, height = 620, center = {{x: width / 2, y: height / 2}};
const root = svg.append("g");
const zoom = d3.zoom().scaleExtent([0.45, 4]).on("zoom", (event) => root.attr("transform", event.transform));
svg.call(zoom);
const message = d3.select("#graph-message");
const edgeSlider = d3.select("#edge-distance");
const edgeValue = d3.select("#edge-distance-value");
let selectedId = null;

function showTooltip(event, lines) {{
  hideTooltip();
  const tooltip = d3.select("body").append("div").attr("id", "graph-tooltip")
    .style("position", "fixed").style("left", (event.clientX + 12) + "px")
    .style("top", (event.clientY + 12) + "px").style("background", colors.tooltip_bg)
    .style("color", colors.text).style("border", "1px solid " + colors.tooltip_border)
    .style("border-radius", "4px").style("padding", "7px 9px").style("font-size", "12px")
    .style("box-shadow", "0 2px 6px #0003").style("pointer-events", "none");
  // .text() per line: labels come from ASR output and must not be parsed as HTML.
  lines.forEach(line => tooltip.append("div").text(line));
}}

function hideTooltip() {{
  d3.select("#graph-tooltip").remove();
}}

function nodeLines(node) {{
  if (node.kind === "reference") return ["Reference: " + node.label];
  const regions = Object.entries(node.region_counts || {{}})
    .map(([region, count]) => region + " (" + count + ")").join(", ");
  return [
    node.label,
    "Reference distance: " + node.distance,
    "Alignments: " + node.count,
    "Mean similarity: " + node.mean_similarity,
    "TF-IDF: " + node.tfidf,
    "Regions: " + (regions || "–"),
  ];
}}

function edgeOpacity(distance, maximumVisibleDistance) {{
  if (distance > maximumVisibleDistance) return 0.04;
  return Math.max(0.18, 0.92 - (distance - 1) * 0.16);
}}

function edgeWidth(distance) {{
  return Math.max(0.7, 2.5 - (distance - 1) * 0.28);
}}

function edgeDash(distance) {{
  return distance === 1 ? null : (Math.max(3, distance + 2) + " " + Math.max(3, distance + 1));
}}

// Shrink hypothesis nodes and labels as more variants are shown so rings stay readable.
const nodeScale = Math.max(0.5, Math.min(1, Math.sqrt(20 / Math.max(1, graphData.nodes.length - 1))));

function nodeRadius(node) {{
  return node.kind === "reference" ? 29 : (10 + Math.min(12, Math.sqrt(node.count || 1) * 1.4)) * nodeScale;
}}

function layout(nodes, edges, ringStep) {{
  // Deterministic start: evenly spaced angles per ring, then let forces settle.
  const byDistance = d3.group(nodes, d => d.distance);
  for (const [distance, ringNodes] of byDistance) {{
    ringNodes.forEach((node, index) => {{
      const angle = -Math.PI / 2 + 2 * Math.PI * index / ringNodes.length + distance * 0.5;
      node.x = center.x + distance * ringStep * Math.cos(angle);
      node.y = center.y + distance * ringStep * Math.sin(angle);
    }});
  }}
  const reference = nodes.find(d => d.kind === "reference");
  if (reference) {{ reference.fx = center.x; reference.fy = center.y; }}

  const maxEdge = Math.max(1, graphData.max_edge_distance);
  const links = edges.map(e => ({{source: e.source, target: e.target, distance: e.distance}}));
  const simulation = d3.forceSimulation(nodes)
    .force("radial", d3.forceRadial(d => d.distance * ringStep, center.x, center.y).strength(1))
    .force("link", d3.forceLink(links).id(d => d.id)
      .distance(d => d.distance * ringStep * 0.6)
      .strength(d => 0.08 * Math.pow(1 - (d.distance - 1) / maxEdge, 2)))
    // Repulsion spreads variants around the full ring; links only decide who neighbours whom.
    .force("charge", d3.forceManyBody().strength(d => d.kind === "reference" ? 0 : -420))
    .force("collide", d3.forceCollide(d => nodeRadius(d) + 6))
    .stop();
  for (let i = 0; i < 300; i++) simulation.tick();
  // Snap back onto the ring so ring membership stays exact.
  for (const node of nodes) {{
    if (node.kind === "reference") continue;
    const angle = Math.atan2(node.y - center.y, node.x - center.x);
    node.x = center.x + node.distance * ringStep * Math.cos(angle);
    node.y = center.y + node.distance * ringStep * Math.sin(angle);
  }}
  return new Map(nodes.map(d => [d.id, {{x: d.x, y: d.y}}]));
}}

function render() {{
  root.selectAll("*").remove();
  if (graphData.nodes.length <= 1) {{
    message.text("No hypothesis variants are available for this reference word.");
    return;
  }}
  message.text("Solid edges are smaller pairwise distances. Hover a node or edge for details; click a node to highlight its edges (click empty space to clear).");
  const maxDistance = Math.max(1, graphData.max_distance);
  const ringStep = Math.min(115, 270 / maxDistance);
  const nodes = graphData.nodes.map(d => ({{...d}}));
  const positions = layout(nodes, graphData.edges, ringStep);

  const rings = root.append("g").attr("class", "rings");
  for (let distance = 1; distance <= maxDistance; distance++) {{
    const radius = distance * ringStep;
    rings.append("circle").attr("cx", center.x).attr("cy", center.y).attr("r", radius)
      .attr("fill", "none").attr("stroke", colors.ring).attr("stroke-dasharray", "3 4");
    rings.append("text").attr("x", center.x + 5).attr("y", center.y - radius + 16)
      .attr("fill", colors.muted).attr("font-size", 12).text("distance " + distance);
  }}

  const edgeLayer = root.append("g").attr("class", "edges");
  const edgeSelection = edgeLayer.selectAll("line").data(graphData.edges).join("line")
    .attr("x1", d => positions.get(d.source).x).attr("y1", d => positions.get(d.source).y)
    .attr("x2", d => positions.get(d.target).x).attr("y2", d => positions.get(d.target).y)
    .attr("stroke", colors.edge)
    .attr("stroke-dasharray", d => edgeDash(d.distance))
    .on("mouseenter", (event, d) => showTooltip(event, ["Pairwise Levenshtein distance: " + d.distance]))
    .on("mouseleave", hideTooltip);
  // Distance labels for the selected node's edges; drawn above edges, below nodes.
  const edgeLabelLayer = root.append("g").attr("class", "edge-labels").style("pointer-events", "none");

  const isIncident = d => selectedId !== null && (d.source === selectedId || d.target === selectedId);

  function updateEdges() {{
    const visible = +edgeSlider.node().value;
    const highlighted = d => isIncident(d) && d.distance <= visible;
    edgeSelection
      .attr("stroke", d => highlighted(d) ? colors.highlight : colors.edge)
      .attr("opacity", d => {{
        if (selectedId === null) return edgeOpacity(d.distance, visible);
        if (highlighted(d)) return 1;
        return d.distance <= visible ? 0.07 : 0.02;
      }})
      .attr("stroke-width", d => highlighted(d) ? 3 : edgeWidth(d.distance));
    edgeSelection.filter(highlighted).raise();

    // Fade every node that is not the selection, one of its visible neighbours, or the reference.
    const neighbours = new Set([selectedId]);
    graphData.edges.filter(highlighted).forEach(d => {{ neighbours.add(d.source); neighbours.add(d.target); }});
    nodeSelection
      .attr("opacity", n => selectedId === null || n.kind === "reference" || neighbours.has(n.id) ? 1 : 0.2);
    nodeSelection.select("circle")
      .attr("stroke", n => n.id === selectedId ? colors.highlight
        : (n.kind === "reference" ? colors.ref_stroke : colors.hyp_stroke))
      .attr("stroke-width", n => n.id === selectedId ? 3.5 : 1.5);

    edgeLabelLayer.selectAll("text").data(graphData.edges.filter(highlighted)).join("text")
      .attr("x", d => (positions.get(d.source).x + positions.get(d.target).x) / 2)
      .attr("y", d => (positions.get(d.source).y + positions.get(d.target).y) / 2)
      .attr("text-anchor", "middle").attr("dy", 4).attr("font-size", 11).attr("font-weight", 700)
      .attr("fill", colors.highlight)
      .attr("stroke", colors.label_halo).attr("stroke-width", 3).attr("paint-order", "stroke")
      .text(d => d.distance);
  }}

  const nodeLayer = root.append("g").attr("class", "nodes");
  const nodeSelection = nodeLayer.selectAll("g").data(nodes).join("g")
    .attr("transform", d => "translate(" + positions.get(d.id).x + "," + positions.get(d.id).y + ")")
    .style("cursor", "pointer")
    .on("mouseenter", (event, d) => showTooltip(event, nodeLines(d)))
    .on("mouseleave", hideTooltip)
    .on("click", (event, d) => {{
      event.stopPropagation();
      // The reference has no pairwise edges, so selecting it would only fade the graph.
      if (d.kind === "reference") return;
      selectedId = selectedId === d.id ? null : d.id;
      updateEdges();
    }});
  nodeSelection.append("circle").attr("r", nodeRadius)
    .attr("fill", d => d.kind === "reference" ? colors.ref_fill : colors.hyp_fill)
    .attr("stroke", d => d.kind === "reference" ? colors.ref_stroke : colors.hyp_stroke)
    .attr("stroke-width", 1.5);
  nodeSelection.append("text").attr("text-anchor", "middle").attr("dy", 4)
    .attr("fill", colors.text)
    .attr("font-size", d => d.kind === "reference" ? 14 : Math.max(9, 12 * nodeScale)).text(d => d.label);

  svg.on("click", () => {{
    selectedId = null;
    updateEdges();
  }});
  edgeSlider.on("input", function() {{
    edgeValue.text(this.value);
    updateEdges();
  }});
  updateEdges();
}}

d3.select("#reset-view").on("click", () => svg.transition().duration(250).call(zoom.transform, d3.zoomIdentity));
render();
</script>
""",
        height=680,
    )
