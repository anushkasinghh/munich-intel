"""CLI entrypoint: python scripts/visualize_graph.py

Renders the graph from graph.py as a self-contained, pannable/zoomable SVG page —
a sanity-check visual for VISION.md step 4, not the step-8 "graph viz for the
actual post" (which comes after the eval harness, per the build order). Writes a
real, standalone file into static/, next to the v1 chat UI's index.html, but
nothing here wires it into api/main.py's routing yet.
"""

import html
import sys
from collections import Counter
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from munich_intel.graph import build_graph

OUTPUT_PATH = Path("static/graph.html")

KIND_LABELS = {
    "company": "Company",
    "funding": "Funding round",
    "investor": "Investor",
    "job": "Job posting",
    "news": "News mention",
}
# Order matches the dataviz skill's validated categorical slots 1-5 (blue, orange,
# aqua, yellow, magenta) — passes CVD + normal-vision separation in both themes.
KIND_ORDER = ["company", "funding", "investor", "job", "news"]
BASE_RADIUS = {"company": 9, "funding": 6, "investor": 6, "job": 4, "news": 3}


def _node_label(kind: str, attrs: dict) -> str:
    if kind == "company":
        return attrs["name"]
    if kind == "funding":
        amount = attrs.get("amount_eur")
        amount_text = f"€{amount:,.0f}" if amount else "amount undisclosed"
        return f"{attrs['round_type'].replace('-', ' ').title()} · {amount_text}"
    if kind == "job":
        return attrs["title"]
    return attrs["title"] if kind == "news" else attrs["name"]  # news / investor


def _node_subtitle(kind: str, attrs: dict) -> str:
    if kind == "company":
        return f"{attrs['category']} · {attrs['hq']}"
    if kind == "funding":
        return attrs.get("announced_on") or ""
    if kind == "job":
        return attrs.get("location") or ""
    if kind == "news":
        return f"{attrs.get('source', '')} · {attrs.get('published_on', '')}"
    return "Investor"


def _build_fragment(graph: nx.DiGraph) -> str:
    # Force-directed layout: nodes that share more connections end up closer
    # together, so each company and its funding/job/news rows cluster visibly.
    positions = nx.spring_layout(graph, seed=42, k=0.6 / (len(graph) ** 0.5), iterations=60)

    xs, ys = [p[0] for p in positions.values()], [p[1] for p in positions.values()]
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
    width, height, pad = 1800, 1100, 40

    def to_screen(x: float, y: float) -> tuple[float, float]:
        sx = pad + (x - x_min) / (x_max - x_min) * (width - 2 * pad)
        sy = pad + (y - y_min) / (y_max - y_min) * (height - 2 * pad)
        return sx, sy

    screen_pos = {n: to_screen(x, y) for n, (x, y) in positions.items()}
    degree = dict(graph.degree())

    edge_lines = []
    for u, v, edge_type in graph.edges(data="edge_type"):
        x1, y1 = screen_pos[u]
        x2, y2 = screen_pos[v]
        edge_lines.append(
            f'<line class="edge" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'data-from-kind="{graph.nodes[u]["kind"]}" data-to-kind="{graph.nodes[v]["kind"]}" '
            f'data-edge-type="{edge_type}"></line>'
        )

    node_circles = []
    for node_id, attrs in graph.nodes(data=True):
        kind = attrs["kind"]
        x, y = screen_pos[node_id]
        hub_bonus = (1.6 if kind == "company" else 0.6) * (min(degree[node_id], 40) ** 0.5)
        radius = BASE_RADIUS[kind] + hub_bonus
        label = html.escape(_node_label(kind, attrs))
        subtitle = html.escape(_node_subtitle(kind, attrs))
        node_circles.append(
            f'<circle class="node node-{kind}" cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" '
            f'data-kind="{kind}" data-label="{label}" data-subtitle="{subtitle}"></circle>'
        )

    node_kind_counts = Counter(kind for _, kind in graph.nodes(data="kind"))
    top_companies = sorted(
        ((n, d) for n, d in degree.items() if graph.nodes[n]["kind"] == "company"),
        key=lambda item: item[1],
        reverse=True,
    )[:8]

    legend_rows = "\n".join(
        f'<label class="legend-row">'
        f'<input type="checkbox" class="kind-toggle" data-kind="{kind}" checked>'
        f'<span class="swatch swatch-{kind}"></span>'
        f'<span class="legend-text">{KIND_LABELS[kind]}</span>'
        f'<span class="legend-count">{node_kind_counts.get(kind, 0)}</span>'
        f"</label>"
        for kind in KIND_ORDER
    )

    top_rows = "\n".join(
        f'<li><span class="rank-name">{html.escape(graph.nodes[n]["name"])}</span>'
        f'<span class="rank-value">{d}</span></li>'
        for n, d in top_companies
    )

    return _PAGE_TEMPLATE.format(
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        company_count=node_kind_counts.get("company", 0),
        edges="\n".join(edge_lines),
        nodes="\n".join(node_circles),
        legend_rows=legend_rows,
        top_rows=top_rows,
        width=width,
        height=height,
    )


_PAGE_TEMPLATE = """\
<style>
  .graph-root {{
    color-scheme: light;
    --page:           #f9f9f7;
    --surface:        #fcfcfb;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --text-muted:     #898781;
    --border:         rgba(11,11,11,0.10);
    --company:        #2a78d6;
    --funding:        #eb6834;
    --investor:       #1baf7a;
    --job:            #eda100;
    --news:           #e87ba4;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) .graph-root {{
      color-scheme: dark;
      --page:           #0d0d0d;
      --surface:        #1a1a19;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted:     #898781;
      --border:         rgba(255,255,255,0.10);
      --company:        #3987e5;
      --funding:        #d95926;
      --investor:       #199e70;
      --job:            #c98500;
      --news:           #d55181;
    }}
  }}
  :root[data-theme="dark"] .graph-root {{
    color-scheme: dark;
    --page:           #0d0d0d;
    --surface:        #1a1a19;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #898781;
    --border:         rgba(255,255,255,0.10);
    --company:        #3987e5;
    --funding:        #d95926;
    --investor:       #199e70;
    --job:            #c98500;
    --news:           #d55181;
  }}

  .graph-root {{
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    background: var(--page);
    color: var(--text-primary);
    display: flex;
    flex-direction: column;
    height: 100vh;
    box-sizing: border-box;
  }}
  .graph-root * {{ box-sizing: border-box; }}

  header.gh {{
    display: flex;
    align-items: baseline;
    gap: 1.25rem;
    flex-wrap: wrap;
    padding: 0.9rem 1.25rem;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }}
  header.gh h1 {{
    font-size: 1.05rem;
    font-weight: 600;
    margin: 0;
    letter-spacing: -0.01em;
  }}
  header.gh p {{
    margin: 0;
    font-size: 0.85rem;
    color: var(--text-secondary);
    flex: 1 1 260px;
    min-width: 0;
  }}
  .stat-chip {{
    font-size: 0.8rem;
    color: var(--text-secondary);
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }}
  .stat-chip b {{ color: var(--text-primary); font-weight: 600; }}

  .gbody {{
    flex: 1;
    display: flex;
    min-height: 0;
  }}
  @media (max-width: 900px) {{
    .gbody {{ flex-direction: column; }}
  }}

  .canvas-wrap {{
    position: relative;
    flex: 1;
    min-width: 0;
    overflow: hidden;
    background: var(--surface);
    cursor: grab;
  }}
  .canvas-wrap.dragging {{ cursor: grabbing; }}
  .canvas-wrap svg {{ width: 100%; height: 100%; display: block; }}

  line.edge {{
    stroke: var(--text-muted);
    stroke-opacity: 0.35;
    stroke-width: 1;
    vector-effect: non-scaling-stroke;
  }}
  line.edge.hidden {{ display: none; }}

  circle.node {{
    stroke: var(--surface);
    stroke-width: 1.5;
    vector-effect: non-scaling-stroke;
    cursor: pointer;
  }}
  circle.node.hidden {{ display: none; }}
  circle.node:hover {{ stroke: var(--text-primary); stroke-width: 2; }}
  .node-company  {{ fill: var(--company); }}
  .node-funding  {{ fill: var(--funding); }}
  .node-investor {{ fill: var(--investor); }}
  .node-job      {{ fill: var(--job); }}
  .node-news     {{ fill: var(--news); }}

  .swatch {{
    width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex: none;
  }}
  .swatch-company  {{ background: var(--company); }}
  .swatch-funding  {{ background: var(--funding); }}
  .swatch-investor {{ background: var(--investor); }}
  .swatch-job      {{ background: var(--job); }}
  .swatch-news     {{ background: var(--news); }}

  .zoom-controls {{
    position: absolute;
    left: 12px;
    bottom: 12px;
    display: flex;
    gap: 6px;
  }}
  .zoom-controls button {{
    width: 28px; height: 28px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--page);
    color: var(--text-primary);
    font-size: 0.95rem;
    cursor: pointer;
  }}
  .zoom-controls button:hover {{ background: var(--surface); }}
  .zoom-controls button:focus-visible {{ outline: 2px solid var(--company); outline-offset: 1px; }}

  aside.panel {{
    width: 260px;
    flex: none;
    overflow-y: auto;
    border-left: 1px solid var(--border);
    background: var(--page);
    padding: 1rem 1.1rem;
  }}
  @media (max-width: 900px) {{
    aside.panel {{ width: 100%; border-left: none; border-top: 1px solid var(--border); max-height: 40vh; }}
  }}
  aside.panel h2 {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    margin: 0 0 0.6rem;
  }}
  .legend-row {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0;
    font-size: 0.85rem;
    cursor: pointer;
  }}
  .legend-row input {{ accent-color: var(--company); }}
  .legend-text {{ flex: 1; color: var(--text-primary); }}
  .legend-count {{ color: var(--text-muted); font-variant-numeric: tabular-nums; font-size: 0.8rem; }}

  ul.top-list {{
    list-style: none;
    margin: 0 0 1.4rem;
    padding: 0;
  }}
  ul.top-list li {{
    display: flex;
    justify-content: space-between;
    gap: 0.5rem;
    font-size: 0.82rem;
    padding: 0.28rem 0;
    border-bottom: 1px solid var(--border);
  }}
  .rank-name {{ color: var(--text-primary); }}
  .rank-value {{ color: var(--text-muted); font-variant-numeric: tabular-nums; }}

  .tooltip {{
    position: fixed;
    pointer-events: none;
    background: var(--text-primary);
    color: var(--surface);
    padding: 0.4rem 0.6rem;
    border-radius: 6px;
    font-size: 0.78rem;
    line-height: 1.35;
    max-width: 240px;
    opacity: 0;
    transform: translate(-50%, -100%);
    transition: opacity 0.08s ease;
    z-index: 10;
  }}
  .tooltip.visible {{ opacity: 1; }}
  .tooltip .t-label {{ font-weight: 600; }}
  .tooltip .t-subtitle {{ opacity: 0.75; }}
</style>

<div class="graph-root">
  <header class="gh">
    <h1>Munich Intel — entity graph</h1>
    <p>{company_count} companies and their funding rounds, investors, job postings, and news
      mentions, drawn from data/entities/. Drag to pan, scroll to zoom, hover a node for detail.</p>
    <span class="stat-chip"><b>{node_count}</b> nodes</span>
    <span class="stat-chip"><b>{edge_count}</b> edges</span>
  </header>
  <div class="gbody">
    <div class="canvas-wrap" id="canvasWrap">
      <svg viewBox="0 0 {width} {height}" id="svgRoot" role="img"
           aria-label="Force-directed graph of Munich AI/deep-tech companies and their funding, investor, job posting, and news entities">
        <g id="viewport">
          <g id="edgeLayer">
{edges}
          </g>
          <g id="nodeLayer">
{nodes}
          </g>
        </g>
      </svg>
      <div class="zoom-controls">
        <button type="button" id="zoomIn" aria-label="Zoom in">+</button>
        <button type="button" id="zoomOut" aria-label="Zoom out">&minus;</button>
        <button type="button" id="zoomReset" aria-label="Reset view">&#8634;</button>
      </div>
      <div class="tooltip" id="tooltip"><div class="t-label"></div><div class="t-subtitle"></div></div>
    </div>
    <aside class="panel">
      <h2>Node kinds</h2>
{legend_rows}
      <h2 style="margin-top:1.2rem">Top companies by degree</h2>
      <ul class="top-list">
{top_rows}
      </ul>
    </aside>
  </div>
</div>

<script>
(function () {{
  var svg = document.getElementById('svgRoot');
  var viewport = document.getElementById('viewport');
  var wrap = document.getElementById('canvasWrap');
  var tooltip = document.getElementById('tooltip');

  var scale = 1, tx = 0, ty = 0;
  function applyTransform() {{
    viewport.setAttribute('transform', 'translate(' + tx + ',' + ty + ') scale(' + scale + ')');
  }}

  wrap.addEventListener('wheel', function (e) {{
    e.preventDefault();
    var rect = wrap.getBoundingClientRect();
    var mx = (e.clientX - rect.left) / rect.width * {width};
    var my = (e.clientY - rect.top) / rect.height * {height};
    var factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    var newScale = Math.min(8, Math.max(0.4, scale * factor));
    tx = mx - (mx - tx) * (newScale / scale);
    ty = my - (my - ty) * (newScale / scale);
    scale = newScale;
    applyTransform();
  }}, {{ passive: false }});

  var dragging = false, lastX = 0, lastY = 0;
  wrap.addEventListener('mousedown', function (e) {{
    dragging = true;
    lastX = e.clientX; lastY = e.clientY;
    wrap.classList.add('dragging');
  }});
  window.addEventListener('mousemove', function (e) {{
    if (!dragging) return;
    var rect = wrap.getBoundingClientRect();
    tx += (e.clientX - lastX) / rect.width * {width};
    ty += (e.clientY - lastY) / rect.height * {height};
    lastX = e.clientX; lastY = e.clientY;
    applyTransform();
  }});
  window.addEventListener('mouseup', function () {{
    dragging = false;
    wrap.classList.remove('dragging');
  }});

  document.getElementById('zoomIn').addEventListener('click', function () {{
    scale = Math.min(8, scale * 1.3); applyTransform();
  }});
  document.getElementById('zoomOut').addEventListener('click', function () {{
    scale = Math.max(0.4, scale / 1.3); applyTransform();
  }});
  document.getElementById('zoomReset').addEventListener('click', function () {{
    scale = 1; tx = 0; ty = 0; applyTransform();
  }});

  var tLabel = tooltip.querySelector('.t-label');
  var tSubtitle = tooltip.querySelector('.t-subtitle');
  svg.addEventListener('mouseover', function (e) {{
    var el = e.target;
    if (!el.classList || !el.classList.contains('node')) return;
    tLabel.textContent = el.dataset.label;
    tSubtitle.textContent = el.dataset.subtitle;
    tooltip.classList.add('visible');
  }});
  svg.addEventListener('mousemove', function (e) {{
    if (!tooltip.classList.contains('visible')) return;
    tooltip.style.left = e.clientX + 'px';
    tooltip.style.top = (e.clientY - 10) + 'px';
  }});
  svg.addEventListener('mouseout', function (e) {{
    if (e.target.classList && e.target.classList.contains('node')) {{
      tooltip.classList.remove('visible');
    }}
  }});

  var toggles = document.querySelectorAll('.kind-toggle');
  toggles.forEach(function (toggle) {{
    toggle.addEventListener('change', function () {{
      var hiddenKinds = {{}};
      toggles.forEach(function (t) {{ if (!t.checked) hiddenKinds[t.dataset.kind] = true; }});
      document.querySelectorAll('circle.node').forEach(function (n) {{
        n.classList.toggle('hidden', !!hiddenKinds[n.dataset.kind]);
      }});
      document.querySelectorAll('line.edge').forEach(function (l) {{
        var hide = hiddenKinds[l.dataset.fromKind] || hiddenKinds[l.dataset.toKind];
        l.classList.toggle('hidden', !!hide);
      }});
    }});
  }});
}})();
</script>
"""


def main() -> None:
    graph = build_graph()
    fragment = _build_fragment(graph)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Munich Intel — Entity Graph</title>\n</head>\n<body>\n"
        + fragment
        + "\n</body>\n</html>\n"
    )
    print(f"Wrote {OUTPUT_PATH} ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")


if __name__ == "__main__":
    main()
