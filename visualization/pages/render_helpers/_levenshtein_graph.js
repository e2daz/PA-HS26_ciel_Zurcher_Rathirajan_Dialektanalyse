const svg = d3.select("#graph-svg");
const width = 900, height = 620, center = {x: width / 2, y: height / 2};
const root = svg.append("g");
const zoom = d3.zoom().scaleExtent([0.45, 4]).on("zoom", (event) => root.attr("transform", event.transform));
svg.call(zoom);
const message = d3.select("#graph-message");
const edgeInput = d3.select("#edge-distance");
const edgeDecrease = d3.select("#edge-decrease");
const edgeIncrease = d3.select("#edge-increase");
const maxEdgeThreshold = +edgeInput.attr("max");
let edgeThreshold = +edgeInput.attr("value");
let selectedId = null;

function showTooltip(event, lines) {
  hideTooltip();
  const tooltip = d3.select("body").append("div").attr("id", "graph-tooltip")
    .style("position", "fixed").style("left", (event.clientX + 12) + "px")
    .style("top", (event.clientY + 12) + "px").style("background", colors.tooltip_bg)
    .style("color", colors.text).style("border", "1px solid " + colors.tooltip_border)
    .style("border-radius", "4px").style("padding", "7px 9px").style("font-size", "12px")
    .style("box-shadow", "0 2px 6px #0003").style("pointer-events", "none");
  // .text() per line: labels come from ASR output and must not be parsed as HTML.
  lines.forEach(line => tooltip.append("div").text(line));
}

function hideTooltip() {
  d3.select("#graph-tooltip").remove();
}

function nodeLines(node) {
  if (node.kind === "reference") return ["Reference: " + node.label];
  const regions = Object.entries(node.region_counts || {})
    .map(([region, count]) => region + " (" + count + ")").join(", ");
  return [
    node.label,
    "Reference distance: " + node.distance,
    "Alignments: " + node.count,
    "Mean similarity: " + node.mean_similarity,
    "TF-IDF: " + node.tfidf,
    "Regions: " + (regions || "–"),
  ];
}

function edgeOpacity(distance, maximumVisibleDistance) {
  if (distance > maximumVisibleDistance) return 0.04;
  return Math.max(0.18, 0.92 - (distance - 1) * 0.16);
}

function edgeWidth(distance) {
  return Math.max(0.7, 2.5 - (distance - 1) * 0.28);
}

function edgeDash(distance) {
  return distance === 1 ? null : (Math.max(3, distance + 2) + " " + Math.max(3, distance + 1));
}

// Shrink hypothesis nodes and labels as more variants are shown so rings stay readable.
const nodeScale = Math.max(0.5, Math.min(1, Math.sqrt(20 / Math.max(1, graphData.nodes.length - 1))));

function nodeRadius(node) {
  return node.kind === "reference" ? 29 : (10 + Math.min(12, Math.sqrt(node.count || 1) * 1.4)) * nodeScale;
}

function layout(nodes, edges, ringStep) {
  // Deterministic start: evenly spaced angles per ring, then let forces settle.
  const byDistance = d3.group(nodes, d => d.distance);
  for (const [distance, ringNodes] of byDistance) {
    ringNodes.forEach((node, index) => {
      const angle = -Math.PI / 2 + 2 * Math.PI * index / ringNodes.length + distance * 0.5;
      node.x = center.x + distance * ringStep * Math.cos(angle);
      node.y = center.y + distance * ringStep * Math.sin(angle);
    });
  }
  const reference = nodes.find(d => d.kind === "reference");
  if (reference) { reference.fx = center.x; reference.fy = center.y; }

  const maxEdge = Math.max(1, graphData.max_edge_distance);
  const links = edges.map(e => ({source: e.source, target: e.target, distance: e.distance}));
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
  for (const node of nodes) {
    if (node.kind === "reference") continue;
    const angle = Math.atan2(node.y - center.y, node.x - center.x);
    node.x = center.x + node.distance * ringStep * Math.cos(angle);
    node.y = center.y + node.distance * ringStep * Math.sin(angle);
  }
  return new Map(nodes.map(d => [d.id, {x: d.x, y: d.y}]));
}

function render() {
  root.selectAll("*").remove();
  if (graphData.nodes.length <= 1) {
    message.text("No hypothesis variants are available for this reference word.");
    d3.selectAll("#ring-graph button, #ring-graph input").property("disabled", true);
    return;
  }
  message.text("Solid edges are smaller pairwise distances. Hover a node or edge for details; click a node to highlight its edges (click empty space to clear).");
  const maxDistance = Math.max(1, graphData.max_distance);
  const ringStep = Math.min(115, 270 / maxDistance);
  const nodes = graphData.nodes.map(d => ({...d}));
  const positions = layout(nodes, graphData.edges, ringStep);

  const rings = root.append("g").attr("class", "rings");
  for (let distance = 1; distance <= maxDistance; distance++) {
    const radius = distance * ringStep;
    rings.append("circle").attr("cx", center.x).attr("cy", center.y).attr("r", radius)
      .attr("fill", "none").attr("stroke", colors.ring).attr("stroke-dasharray", "3 4");
  }

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

  function updateEdges() {
    const visible = edgeThreshold;
    const highlighted = d => isIncident(d) && d.distance <= visible;
    edgeSelection
      .attr("stroke", d => highlighted(d) ? colors.highlight : colors.edge)
      .attr("opacity", d => {
        if (selectedId === null) return edgeOpacity(d.distance, visible);
        if (highlighted(d)) return 1;
        return d.distance <= visible ? 0.07 : 0.02;
      })
      .attr("stroke-width", d => highlighted(d) ? 3 : edgeWidth(d.distance));
    edgeSelection.filter(highlighted).raise();

    // Fade every node that is not the selection, one of its visible neighbours, or the reference.
    const neighbours = new Set([selectedId]);
    graphData.edges.filter(highlighted).forEach(d => { neighbours.add(d.source); neighbours.add(d.target); });
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
  }

  const nodeLayer = root.append("g").attr("class", "nodes");
  const nodeSelection = nodeLayer.selectAll("g").data(nodes).join("g")
    .attr("transform", d => "translate(" + positions.get(d.id).x + "," + positions.get(d.id).y + ")")
    .style("cursor", "pointer")
    .on("mouseenter", (event, d) => showTooltip(event, nodeLines(d)))
    .on("mouseleave", hideTooltip)
    .on("click", (event, d) => {
      event.stopPropagation();
      // The reference has no pairwise edges, so selecting it would only fade the graph.
      if (d.kind === "reference") return;
      selectedId = selectedId === d.id ? null : d.id;
      updateEdges();
    });
  nodeSelection.append("circle").attr("r", nodeRadius)
    .attr("fill", d => d.kind === "reference" ? colors.ref_fill : colors.hyp_fill)
    .attr("stroke", d => d.kind === "reference" ? colors.ref_stroke : colors.hyp_stroke)
    .attr("stroke-width", 1.5);
  nodeSelection.append("text").attr("text-anchor", "middle").attr("dy", 4)
    .attr("fill", colors.text)
    .attr("font-size", d => d.kind === "reference" ? 14 : Math.max(9, 12 * nodeScale)).text(d => d.label);

  // Above the nodes so a node can never cover a ring's distance; ignores the pointer so nodes stay clickable.
  const ringLabelLayer = root.append("g").attr("class", "ring-labels").style("pointer-events", "none");
  for (let distance = 1; distance <= maxDistance; distance++) {
    ringLabelLayer.append("text").attr("x", center.x).attr("y", center.y - distance * ringStep)
      .attr("text-anchor", "middle").attr("dy", 4).attr("font-size", 12).attr("font-weight", 600)
      .attr("fill", colors.muted)
      .attr("stroke", colors.label_halo).attr("stroke-width", 3).attr("paint-order", "stroke")
      .text(distance === maxDistance ? "distance " + distance : distance);
  }

  svg.on("click", () => {
    selectedId = null;
    updateEdges();
  });
  function applyEdgeThreshold(value) {
    edgeThreshold = value;
    edgeDecrease.property("disabled", value <= 1);
    edgeIncrease.property("disabled", value >= maxEdgeThreshold);
    updateEdges();
  }

  function setEdgeThreshold(value) {
    const clamped = Math.min(maxEdgeThreshold, Math.max(1, Math.round(value)));
    edgeInput.property("value", clamped);
    applyEdgeThreshold(clamped);
  }

  edgeDecrease.on("click", () => setEdgeThreshold(edgeThreshold - 1));
  edgeIncrease.on("click", () => setEdgeThreshold(edgeThreshold + 1));
  // While typing, apply valid values but leave the text alone so "1" can still become "12".
  edgeInput.on("input", function() {
    const value = this.valueAsNumber;
    if (Number.isInteger(value) && value >= 1 && value <= maxEdgeThreshold) applyEdgeThreshold(value);
  });
  // On Enter or blur, snap whatever was typed back into range; an empty field keeps the last value.
  edgeInput.on("change", function() {
    setEdgeThreshold(Number.isNaN(this.valueAsNumber) ? edgeThreshold : this.valueAsNumber);
  });
  setEdgeThreshold(edgeThreshold);
}

d3.select("#reset-view").on("click", () => svg.transition().duration(250).call(zoom.transform, d3.zoomIdentity));
render();
