#!/usr/bin/env python3
"""Generate an HTML comparison grid from inference output directories.

Scans for .json config files alongside .mp4 videos, extracts parameters,
and produces a self-contained HTML page with video playback and filtering.

Usage:
    python scripts/compare.py outputs/open_h/04_2026_03_20/nonexpert-2026-01-15_16-50-48
    python scripts/compare.py outputs/open_h  # scan all runs recursively
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CONTROL_KEYS = ["edge", "depth", "vis", "seg"]


def collect_samples(root: Path) -> list[dict]:
    """Walk root for .json configs and pair with matching .mp4 videos."""
    samples = []
    for json_path in sorted(root.rglob("*.json")):
        if json_path.name in ("config.yaml", "config.json"):
            continue
        video_path = json_path.with_suffix(".mp4")
        if not video_path.exists():
            continue

        with open(json_path) as f:
            try:
                cfg = json.load(f)
            except json.JSONDecodeError:
                continue

        # Extract key parameters
        controls = {}
        for key in CONTROL_KEYS:
            ctrl = cfg.get(key)
            if ctrl is not None:
                controls[key] = {
                    "weight": ctrl.get("control_weight", 1.0),
                    "has_control_path": ctrl.get("control_path") is not None,
                }
                if key == "vis":
                    controls[key]["blur_strength"] = ctrl.get("preset_blur_strength", "medium")

        # Collect control video paths
        control_videos = {}
        for key in CONTROL_KEYS:
            ctrl_vid = json_path.with_name(f"{json_path.stem}_control_{key}.mp4")
            if ctrl_vid.exists():
                control_videos[key] = str(ctrl_vid.relative_to(root))

        # Run ID from directory structure (e.g. "04_2026_03_20")
        rel = json_path.relative_to(root)
        run_id = str(rel.parts[0]) if len(rel.parts) > 1 else ""

        samples.append({
            "name": cfg.get("name", json_path.stem),
            "run_id": run_id,
            "video": str(video_path.relative_to(root)),
            "control_videos": control_videos,
            "guidance": cfg.get("guidance"),
            "seed": cfg.get("seed"),
            "sigma_max": cfg.get("sigma_max"),
            "resolution": cfg.get("resolution"),
            "num_steps": cfg.get("num_steps"),
            "controls": controls,
            "prompt": cfg.get("prompt", ""),
            "size_kb": round(video_path.stat().st_size / 1024, 1),
        })

    return samples


def generate_html(samples: list[dict], root: Path) -> str:
    """Generate a self-contained HTML comparison page."""
    samples_json = json.dumps(samples, indent=2)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Experiment Comparison — {root.name}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; background: #1a1a2e; color: #e0e0e0; padding: 16px; }}
h1 {{ font-size: 18px; margin-bottom: 12px; color: #a0a0d0; }}
.toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; padding: 12px; background: #16213e; border-radius: 8px; align-items: center; }}
.toolbar label {{ font-size: 12px; color: #8888aa; margin-right: 2px; }}
.toolbar select, .toolbar input {{ font-size: 12px; padding: 4px 8px; background: #0f3460; color: #e0e0e0; border: 1px solid #333366; border-radius: 4px; }}
.toolbar .sep {{ width: 1px; height: 24px; background: #333366; margin: 0 4px; }}
.toolbar button {{ font-size: 12px; padding: 4px 12px; background: #533483; color: #e0e0e0; border: none; border-radius: 4px; cursor: pointer; }}
.toolbar button:hover {{ background: #6a44a0; }}
.toolbar button.active {{ background: #e94560; }}
.count {{ font-size: 12px; color: #8888aa; margin-left: auto; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 12px; }}
.card {{ background: #16213e; border-radius: 8px; overflow: hidden; border: 2px solid transparent; transition: border-color 0.2s; }}
.card:hover {{ border-color: #533483; }}
.card.selected {{ border-color: #e94560; }}
.card video {{ width: 100%; display: block; cursor: pointer; background: #000; }}
.card .info {{ padding: 10px; }}
.card .name {{ font-size: 13px; font-weight: 600; color: #a0a0d0; margin-bottom: 6px; }}
.card .params {{ display: flex; flex-wrap: wrap; gap: 4px; }}
.card .tag {{ font-size: 11px; padding: 2px 6px; border-radius: 3px; background: #0f3460; }}
.tag.seg {{ background: #1a5c3a; }}
.tag.vis {{ background: #5c3a1a; }}
.tag.depth {{ background: #1a3a5c; }}
.tag.edge {{ background: #5c1a5c; }}
.tag.guidance {{ background: #533483; }}
.tag.sigma {{ background: #834533; }}
.card .size {{ font-size: 10px; color: #666688; margin-top: 4px; }}
.controls-row {{ display: flex; gap: 4px; margin-top: 6px; }}
.controls-row button {{ font-size: 10px; padding: 2px 6px; background: #0f3460; color: #aaa; border: 1px solid #333366; border-radius: 3px; cursor: pointer; }}
.controls-row button:hover {{ background: #1a4a80; color: #fff; }}
.compare-bar {{ display: none; position: fixed; bottom: 0; left: 0; right: 0; background: #16213e; border-top: 2px solid #e94560; padding: 8px 16px; z-index: 100; }}
.compare-bar.visible {{ display: flex; align-items: center; gap: 12px; }}
.compare-view {{ position: fixed; inset: 0; background: #1a1a2e; z-index: 200; display: none; flex-direction: column; }}
.compare-view.visible {{ display: flex; }}
.compare-header {{ padding: 12px 16px; background: #16213e; display: flex; align-items: center; gap: 12px; }}
.compare-header button {{ font-size: 13px; padding: 6px 16px; background: #e94560; color: #fff; border: none; border-radius: 4px; cursor: pointer; }}
.compare-grid {{ flex: 1; display: grid; gap: 8px; padding: 12px; overflow: auto; }}
.compare-grid video {{ width: 100%; border-radius: 4px; }}
.compare-grid .compare-label {{ font-size: 11px; color: #a0a0d0; padding: 4px; text-align: center; }}
.sync-btn {{ font-size: 11px; padding: 4px 10px; background: #0f3460; color: #aaa; border: 1px solid #333366; border-radius: 4px; cursor: pointer; }}
.sync-btn.active {{ background: #1a5c3a; color: #fff; border-color: #1a5c3a; }}
</style>
</head>
<body>

<h1>Experiment Comparison — {root.name}</h1>

<div class="toolbar" id="toolbar">
  <label>Sort:</label>
  <select id="sort-by">
    <option value="name">Name</option>
    <option value="guidance">Guidance</option>
    <option value="seg_w">Seg Weight</option>
    <option value="vis_w">Vis Weight</option>
    <option value="size">File Size</option>
  </select>
  <div class="sep"></div>
  <label>Guidance:</label>
  <select id="filter-guidance"><option value="">All</option></select>
  <label>Controls:</label>
  <select id="filter-controls"><option value="">All</option></select>
  <div class="sep"></div>
  <label>Search:</label>
  <input type="text" id="search" placeholder="filter by name..." size="20">
  <div class="sep"></div>
  <button id="select-mode-btn" onclick="toggleSelectMode()">Select to Compare</button>
  <span class="count" id="count"></span>
</div>

<div class="grid" id="grid"></div>

<div class="compare-bar" id="compare-bar">
  <span id="compare-count">0 selected</span>
  <button onclick="openCompare()">Compare Side by Side</button>
  <button onclick="clearSelection()" style="background:#333366">Clear</button>
</div>

<div class="compare-view" id="compare-view">
  <div class="compare-header">
    <button onclick="closeCompare()">Back</button>
    <span id="compare-title" style="color:#a0a0d0; font-size:14px;"></span>
    <button class="sync-btn" id="sync-btn" onclick="toggleSync()">Sync Playback</button>
    <button onclick="playAll()" style="background:#1a5c3a; border:none; color:#fff; padding:6px 12px; border-radius:4px; cursor:pointer; font-size:12px;">Play All</button>
  </div>
  <div class="compare-grid" id="compare-grid"></div>
</div>

<script>
const SAMPLES = {samples_json};

let selectMode = false;
let selected = new Set();
let syncPlayback = false;

function getControlSummary(s) {{
  const parts = [];
  for (const key of ["seg", "vis", "depth", "edge"]) {{
    if (s.controls[key]) parts.push(key);
  }}
  return parts.join("+") || "none";
}}

function populateFilters() {{
  const guidances = [...new Set(SAMPLES.map(s => s.guidance))].sort((a, b) => a - b);
  const sel = document.getElementById("filter-guidance");
  guidances.forEach(g => {{
    const opt = document.createElement("option");
    opt.value = g;
    opt.textContent = g;
    sel.appendChild(opt);
  }});

  const ctrlSets = [...new Set(SAMPLES.map(getControlSummary))].sort();
  const sel2 = document.getElementById("filter-controls");
  ctrlSets.forEach(c => {{
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = c;
    sel2.appendChild(opt);
  }});
}}

function getFiltered() {{
  let items = [...SAMPLES];
  const g = document.getElementById("filter-guidance").value;
  if (g) items = items.filter(s => String(s.guidance) === g);
  const c = document.getElementById("filter-controls").value;
  if (c) items = items.filter(s => getControlSummary(s) === c);
  const q = document.getElementById("search").value.toLowerCase();
  if (q) items = items.filter(s => s.name.toLowerCase().includes(q));

  const sortBy = document.getElementById("sort-by").value;
  items.sort((a, b) => {{
    if (sortBy === "name") return a.name.localeCompare(b.name);
    if (sortBy === "guidance") return (a.guidance || 0) - (b.guidance || 0);
    if (sortBy === "seg_w") return ((a.controls.seg?.weight) || 0) - ((b.controls.seg?.weight) || 0);
    if (sortBy === "vis_w") return ((a.controls.vis?.weight) || 0) - ((b.controls.vis?.weight) || 0);
    if (sortBy === "size") return a.size_kb - b.size_kb;
    return 0;
  }});
  return items;
}}

function renderGrid() {{
  const items = getFiltered();
  const grid = document.getElementById("grid");
  grid.innerHTML = "";

  items.forEach(s => {{
    const card = document.createElement("div");
    card.className = "card" + (selected.has(s.name) ? " selected" : "");
    card.dataset.name = s.name;

    let tags = "";
    for (const [key, ctrl] of Object.entries(s.controls)) {{
      tags += `<span class="tag ${{key}}">${{key}}: ${{ctrl.weight}}${{ctrl.blur_strength && ctrl.blur_strength !== "medium" ? " (" + ctrl.blur_strength + ")" : ""}}</span>`;
    }}
    if (s.guidance != null) tags += `<span class="tag guidance">g=${{s.guidance}}</span>`;
    if (s.sigma_max != null) tags += `<span class="tag sigma">&sigma;=${{s.sigma_max}}</span>`;

    let ctrlBtns = "";
    for (const [key, path] of Object.entries(s.control_videos)) {{
      ctrlBtns += `<button onclick="event.stopPropagation(); swapVideo(this, '${{s.name}}', '${{path}}')">${{key}} ctrl</button>`;
    }}
    if (ctrlBtns) {{
      ctrlBtns = `<button onclick="event.stopPropagation(); swapVideo(this, '${{s.name}}', '${{s.video}}')">output</button>` + ctrlBtns;
    }}

    card.innerHTML = `
      <video src="${{s.video}}" loop muted playsinline preload="metadata"
             onmouseenter="this.play()" onmouseleave="this.pause();this.currentTime=0"
             id="vid-${{s.name}}"></video>
      <div class="info">
        <div class="name">${{s.name}}</div>
        <div class="params">${{tags}}</div>
        ${{ctrlBtns ? '<div class="controls-row">' + ctrlBtns + '</div>' : ''}}
        <div class="size">${{s.size_kb}} KB</div>
      </div>
    `;

    card.addEventListener("click", () => {{
      if (selectMode) {{
        if (selected.has(s.name)) selected.delete(s.name);
        else selected.add(s.name);
        renderGrid();
        updateCompareBar();
      }}
    }});

    grid.appendChild(card);
  }});

  document.getElementById("count").textContent = `${{items.length}} / ${{SAMPLES.length}} samples`;
}}

function swapVideo(btn, name, src) {{
  const vid = document.getElementById("vid-" + name);
  vid.src = src;
  vid.load();
  // highlight active button
  btn.parentElement.querySelectorAll("button").forEach(b => b.style.background = "#0f3460");
  btn.style.background = "#533483";
}}

function toggleSelectMode() {{
  selectMode = !selectMode;
  const btn = document.getElementById("select-mode-btn");
  btn.classList.toggle("active", selectMode);
  btn.textContent = selectMode ? "Done Selecting" : "Select to Compare";
  if (!selectMode && selected.size === 0) {{
    document.getElementById("compare-bar").classList.remove("visible");
  }}
}}

function updateCompareBar() {{
  const bar = document.getElementById("compare-bar");
  document.getElementById("compare-count").textContent = selected.size + " selected";
  bar.classList.toggle("visible", selected.size > 0);
}}

function clearSelection() {{
  selected.clear();
  renderGrid();
  updateCompareBar();
}}

function openCompare() {{
  const view = document.getElementById("compare-view");
  const grid = document.getElementById("compare-grid");
  const items = SAMPLES.filter(s => selected.has(s.name));
  const cols = Math.min(items.length, 4);
  grid.style.gridTemplateColumns = `repeat(${{cols}}, 1fr)`;
  grid.innerHTML = "";

  items.forEach(s => {{
    let tags = "";
    for (const [key, ctrl] of Object.entries(s.controls)) {{
      tags += `${{key}}=${{ctrl.weight}} `;
    }}
    const div = document.createElement("div");
    div.innerHTML = `
      <video src="${{s.video}}" loop muted playsinline preload="auto" class="compare-vid"></video>
      <div class="compare-label">${{s.name}}<br><small>g=${{s.guidance}} ${{tags}}</small></div>
    `;
    grid.appendChild(div);
  }});

  document.getElementById("compare-title").textContent = `Comparing ${{items.length}} samples`;
  view.classList.add("visible");
}}

function closeCompare() {{
  document.getElementById("compare-view").classList.remove("visible");
}}

function toggleSync() {{
  syncPlayback = !syncPlayback;
  const btn = document.getElementById("sync-btn");
  btn.classList.toggle("active", syncPlayback);

  if (syncPlayback) {{
    const videos = document.querySelectorAll(".compare-vid");
    videos.forEach(v => {{
      v.addEventListener("play", syncHandler);
      v.addEventListener("pause", syncHandler);
      v.addEventListener("seeked", syncHandler);
    }});
  }}
}}

function syncHandler(e) {{
  if (!syncPlayback) return;
  const videos = document.querySelectorAll(".compare-vid");
  videos.forEach(v => {{
    if (v === e.target) return;
    if (e.type === "play") {{ v.currentTime = e.target.currentTime; v.play(); }}
    if (e.type === "pause") v.pause();
    if (e.type === "seeked") v.currentTime = e.target.currentTime;
  }});
}}

function playAll() {{
  const videos = document.querySelectorAll(".compare-vid");
  videos.forEach(v => {{ v.currentTime = 0; v.play(); }});
}}

// Wire up filters
["sort-by", "filter-guidance", "filter-controls"].forEach(id =>
  document.getElementById(id).addEventListener("change", renderGrid));
document.getElementById("search").addEventListener("input", renderGrid);

populateFilters();
renderGrid();
</script>
</body>
</html>"""


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output_dir>", file=sys.stderr)
        sys.exit(1)

    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    samples = collect_samples(root)
    if not samples:
        print(f"No samples found in {root}", file=sys.stderr)
        sys.exit(1)

    html = generate_html(samples, root)
    out_path = root / "compare.html"
    out_path.write_text(html)
    print(f"Generated {out_path} ({len(samples)} samples)")


if __name__ == "__main__":
    main()
