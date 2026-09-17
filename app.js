/* Adaptive HR Tile Selection — static viewer for the Kaggle run artifacts. */
(() => {
"use strict";

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : (+v).toFixed(d);
const pct = v => `${(v * 100).toFixed(0)}%`;
// matchMedia is universal in browsers but absent in some embedded/headless contexts;
// never let a theme query be the thing that stops the map from rendering.
const prefersDark = () => {
  try { return !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches); }
  catch { return false; }
};
const isDark = () => {
  const t = document.documentElement.dataset.theme;
  return t ? t === "dark" : prefersDark();
};

let DATA = null, MAP = null, LAYER = null, SEL = null;

/* ───────────────────────────── theme & tabs ───────────────────────────── */
const savedTheme = (() => { try { return localStorage.getItem("povrl-theme"); } catch { return null; } })();
if (savedTheme) document.documentElement.dataset.theme = savedTheme;

$("#themeToggle").addEventListener("click", () => {
  const next = isDark() ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("povrl-theme", next); } catch {}
  redrawAll();
});

/* "report" is prerendered static HTML and needs no data; the other three do. The spinner is
   therefore only shown when the reader asks for a JS-rendered tab before the fetch lands. */
const VIEWS = ["report", "map", "results", "method"];
let ACTIVE = "report";
function showView(v) {
  ACTIVE = v;
  VIEWS.forEach(x => { const el = $(`#view-${x}`); if (el) el.hidden = x !== v; });
  $$(".tab").forEach(t => t.classList.toggle("is-active", t.dataset.view === v));
  const needsData = v !== "report";
  const failed = !$("#nodata").hidden;
  $("#loading").hidden = !(needsData && !DATA && !failed);
  if (v === "map" && DATA) { ensureMap(); if (MAP) setTimeout(() => MAP.invalidateSize(), 50); }
  if (v === "results" && DATA) redrawAll();
}
$$(".tab").forEach(t => t.addEventListener("click", () => showView(t.dataset.view)));

/* ───────────────────────────── tooltip ────────────────────────────────── */
const tip = Object.assign(document.createElement("div"), { className: "tip" });
tip.style.display = "none"; document.body.appendChild(tip);
const showTip = (e, html) => {
  tip.innerHTML = html; tip.style.display = "block";
  const r = tip.getBoundingClientRect();
  tip.style.left = `${Math.min(e.clientX + 14, innerWidth - r.width - 8)}px`;
  tip.style.top  = `${Math.max(8, e.clientY - r.height - 12)}px`;
};
const hideTip = () => { tip.style.display = "none"; };
const hoverable = (el, html) => {
  el.addEventListener("mousemove", e => showTip(e, html));
  el.addEventListener("mouseleave", hideTip);
};

/* ───────────────────────────── svg helpers ────────────────────────────── */
const NS = "http://www.w3.org/2000/svg";
const mk = (tag, attrs = {}) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
};
const svgRoot = (host, h) => {
  host.innerHTML = "";
  const w = Math.max(280, host.clientWidth || 600);
  const s = mk("svg", { viewBox: `0 0 ${w} ${h}`, width: w, height: h,
                        role: "img", style: "max-width:100%" });
  host.appendChild(s);
  return { s, w, h };
};
const niceMax = v => {
  if (v <= 0) return 1;
  const p = 10 ** Math.floor(Math.log10(v)), n = v / p;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * p;
};
const yAxis = (s, x0, x1, yTop, yBot, max, ticks = 4, unit = "") => {
  for (let i = 0; i <= ticks; i++) {
    const v = max * i / ticks, y = yBot - (yBot - yTop) * i / ticks;
    s.appendChild(mk("line", { x1: x0, x2: x1, y1: y, y2: y, class: "ax" }));
    const t = mk("text", { x: x0 - 7, y: y + 3.5, class: "axtext", "text-anchor": "end" });
    t.textContent = (max <= 2 ? v.toFixed(2) : v.toFixed(0)) + unit;
    s.appendChild(t);
  }
};
const legendRow = (host, items) => {
  const d = document.createElement("div"); d.className = "legend-row";
  d.innerHTML = items.map(i =>
    `<span><i class="swatch" style="background:${i.color}"></i>${i.name}</span>`).join("");
  host.parentNode.insertBefore(d, host);
};

/* ───────────────────────────── charts ─────────────────────────────────── */
/* Grouped bars: categories x 2 series. 2px surface gap between adjacent fills. */
function groupedBar(host, cats, series, opts = {}) {
  const PADL = 46, PADR = 12, PADT = 10, PADB = 74;
  const { s, w } = svgRoot(host, opts.height || 280);
  const h = opts.height || 280, yTop = PADT, yBot = h - PADB;
  const max = niceMax(Math.max(...series.flatMap(x => x.values.filter(Number.isFinite)), 0.001));
  yAxis(s, PADL, w - PADR, yTop, yBot, max, 4);
  const bw = (w - PADL - PADR) / cats.length;
  const gap = 2, inner = Math.min(34, (bw - 14) / series.length);
  cats.forEach((c, i) => {
    const cx = PADL + bw * i + bw / 2;
    series.forEach((se, k) => {
      const v = se.values[i];
      const x = cx - (series.length * inner + gap) / 2 + k * (inner + gap);
      if (Number.isFinite(v)) {
        const bh = Math.max(1, (yBot - yTop) * v / max);
        const r = mk("rect", { x, y: yBot - bh, width: inner, height: bh,
                               rx: Math.min(4, inner / 2), fill: se.color });
        hoverable(r, `<b>${c}</b><br>${se.name} &middot; <b>${fmt(v, 3)}</b>`);
        s.appendChild(r);
      }
    });
    const t = mk("text", { x: cx, y: yBot + 14, class: "axtext",
                           "text-anchor": "end", transform: `rotate(-38 ${cx} ${yBot + 14})` });
    t.textContent = c; s.appendChild(t);
  });
  s.appendChild(mk("line", { x1: PADL, x2: w - PADR, y1: yBot, y2: yBot, class: "ax" }));
  if (opts.yLabel) {
    const t = mk("text", { x: 10, y: (yTop + yBot) / 2, class: "lbl", "text-anchor": "middle",
                           transform: `rotate(-90 10 ${(yTop + yBot) / 2})` });
    t.textContent = opts.yLabel; s.appendChild(t);
  }
}

/* Horizontal bars — one series, so no legend box; values direct-labelled. */
function hBar(host, rows, opts = {}) {
  const PADL = opts.padLeft || 120, PADR = 54, PADT = 6, rowH = opts.rowH || 22;
  const h = PADT + rows.length * rowH + 22;
  const { s, w } = svgRoot(host, h);
  const max = niceMax(Math.max(...rows.map(r => r.value), 1e-9));
  const x0 = PADL, x1 = w - PADR;
  rows.forEach((r, i) => {
    const y = PADT + i * rowH;
    const bw = Math.max(1, (x1 - x0) * r.value / max);
    const rect = mk("rect", { x: x0, y: y + 3, width: bw, height: rowH - 8,
                              rx: 4, fill: r.color || css("--series-1") });
    hoverable(rect, `<b>${r.label}</b><br>${opts.unit || ""}<b>${fmt(r.value, opts.dp ?? 2)}</b>`);
    s.appendChild(rect);
    const lt = mk("text", { x: x0 - 8, y: y + rowH / 2 + 1, class: "lbl", "text-anchor": "end" });
    lt.textContent = r.label; s.appendChild(lt);
    const vt = mk("text", { x: x0 + bw + 7, y: y + rowH / 2 + 1, class: "vlbl" });
    vt.textContent = fmt(r.value, opts.dp ?? 2); s.appendChild(vt);
  });
  s.appendChild(mk("line", { x1: x0, x2: x0, y1: PADT, y2: PADT + rows.length * rowH, class: "ax" }));
}

/* Line chart with markers; optional per-point annotation. */
function lineChart(host, series, opts = {}) {
  const PADL = 48, PADR = 18, PADT = 12, PADB = 42;
  const h = opts.height || 260;
  const { s, w } = svgRoot(host, h);
  const yTop = PADT, yBot = h - PADB;
  const pts = series.flatMap(x => x.points);
  const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
  const xmin = opts.xmin ?? Math.min(...xs), xmax = opts.xmax ?? Math.max(...xs);
  const ymax = niceMax(Math.max(...ys, 1e-9)), ymin = 0;
  const X = v => PADL + (w - PADL - PADR) * (xmax === xmin ? .5 : (v - xmin) / (xmax - xmin));
  const Y = v => yBot - (yBot - yTop) * (v - ymin) / (ymax - ymin);
  yAxis(s, PADL, w - PADR, yTop, yBot, ymax, 4);
  for (let i = 0; i <= 4; i++) {
    const v = xmin + (xmax - xmin) * i / 4;
    const t = mk("text", { x: X(v), y: yBot + 16, class: "axtext", "text-anchor": "middle" });
    t.textContent = opts.xFmt ? opts.xFmt(v) : fmt(v, 2); s.appendChild(t);
  }
  series.forEach(se => {
    const d = se.points.map((p, i) => `${i ? "L" : "M"}${X(p.x)},${Y(p.y)}`).join(" ");
    s.appendChild(mk("path", { d, fill: "none", stroke: se.color,
                               "stroke-width": se.width || 2, "stroke-opacity": se.opacity ?? 1,
                               "stroke-linejoin": "round", "stroke-linecap": "round" }));
    if (!se.noMarkers) se.points.forEach(p => {
      const c = mk("circle", { cx: X(p.x), cy: Y(p.y), r: 4.5, fill: se.color,
                               stroke: css("--surface-1"), "stroke-width": 2 });
      hoverable(c, `<b>${se.name}</b><br>${opts.xLabel || "x"} <b>${fmt(p.x, 2)}</b>` +
                   `<br>${opts.yLabel || "y"} <b>${fmt(p.y, 3)}</b>` +
                   (p.note ? `<br><span class="r">${p.note}</span>` : ""));
      s.appendChild(c);
      if (p.label) {
        const t = mk("text", { x: X(p.x) + 6, y: Y(p.y) - 8, class: "axtext" });
        t.textContent = p.label; s.appendChild(t);
      }
    });
  });
  s.appendChild(mk("line", { x1: PADL, x2: w - PADR, y1: yBot, y2: yBot, class: "ax" }));
  if (opts.xLabel) {
    const t = mk("text", { x: (PADL + w - PADR) / 2, y: h - 8, class: "lbl", "text-anchor": "middle" });
    t.textContent = opts.xLabel; s.appendChild(t);
  }
}

function scatter(host, pts, opts = {}) {
  const PADL = 46, PADR = 16, PADT = 12, PADB = 42, h = opts.height || 260;
  const { s, w } = svgRoot(host, h);
  const yTop = PADT, yBot = h - PADB;
  const m = niceMax(Math.max(...pts.flatMap(p => [p.x, p.y]), 1e-9));
  const X = v => PADL + (w - PADL - PADR) * v / m;
  const Y = v => yBot - (yBot - yTop) * v / m;
  yAxis(s, PADL, w - PADR, yTop, yBot, m, 4);
  for (let i = 0; i <= 4; i++) {
    const v = m * i / 4;
    const t = mk("text", { x: X(v), y: yBot + 16, class: "axtext", "text-anchor": "middle" });
    t.textContent = fmt(v, 1); s.appendChild(t);
  }
  s.appendChild(mk("line", { x1: X(0), y1: Y(0), x2: X(m), y2: Y(m),
                             stroke: css("--text-muted"), "stroke-width": 1.25, "stroke-opacity": .55 }));
  pts.forEach(p => {
    const c = mk("circle", { cx: X(p.x), cy: Y(p.y), r: 3.6, fill: css("--series-1"),
                             "fill-opacity": .62, stroke: css("--surface-1"), "stroke-width": 1.5 });
    hoverable(c, `<b>Cluster ${p.cid}</b><br>true <b>${fmt(p.x)}</b> &middot; predicted <b>${fmt(p.y)}</b>`);
    c.style.cursor = "pointer";
    c.addEventListener("click", () => selectCluster(p.cid));
    s.appendChild(c);
  });
  s.appendChild(mk("line", { x1: PADL, x2: w - PADR, y1: yBot, y2: yBot, class: "ax" }));
  [[opts.xLabel, (PADL + w - PADR) / 2, h - 8, ""],
   [opts.yLabel, 11, (yTop + yBot) / 2, `rotate(-90 11 ${(yTop + yBot) / 2})`]]
    .forEach(([lab, x, y, tr]) => {
      if (!lab) return;
      const t = mk("text", { x, y, class: "lbl", "text-anchor": "middle" });
      if (tr) t.setAttribute("transform", tr);
      t.textContent = lab; s.appendChild(t);
    });
}

/* ───────────────────────────── colour scales ──────────────────────────── */
const SEQ = () => [1,2,3,4,5,6,7].map(i => css(`--seq-${i}`));
const seqColor = t => { const r = SEQ(); return r[Math.max(0, Math.min(r.length - 1, Math.round(t * (r.length - 1))))]; };
const divColor = t => {                    // t in [-1,1]
  const lo = css("--div-lo"), mid = css("--div-mid"), hi = css("--div-hi");
  return t < -0.15 ? lo : t > 0.15 ? hi : mid;
};
const METRICS = {
  y_true:  { label: "True consumption", unit: "", get: c => c.y_true,  kind: "seq" },
  y_pred:  { label: "Predicted consumption", unit: "", get: c => c.y_pred, kind: "seq" },
  err:     { label: "Prediction error", unit: "", get: c => c.y_pred - c.y_true, kind: "div" },
  hr_frac: { label: "HR tiles acquired", unit: "", get: c => c.hr_frac, kind: "seq" },
};

function quantiles(vals) {
  const s = [...vals].sort((a, b) => a - b);
  const q = p => s[Math.max(0, Math.min(s.length - 1, Math.floor(p * (s.length - 1))))];
  return { lo: q(0.05), hi: q(0.95) };
}

/* ───────────────────────────── map ────────────────────────────────────── */
/* The map depends on Leaflet and on remote basemap tiles; the rest of the page does not.
   Never let a map failure hide the results. */
function ensureMap() {
  if (MAP || !DATA) return;
  try {
    if (typeof L === "undefined") throw new Error("Leaflet did not load");
    initMap();
  } catch (e) {
    window.__mapError = e;              // keep the stack reachable for debugging
    console.warn("map init failed:", e);
    const m = $("#map");
    if (m) m.innerHTML =
      `<div style="padding:24px;color:var(--text-secondary);font-size:13.5px">
         <b>Map unavailable</b><br>${e.message}.<br>
         Cluster details are still reachable from the scatter plot on the Results tab.
       </div>`;
  }
}
function initMap() {
  MAP = L.map("map", { zoomControl: true, scrollWheelZoom: true });
  refreshBasemap();
  MAP.fitBounds(DATA.clusters.map(c => [c.lat, c.lon]), { padding: [30, 30] });
  drawMarkers();
}
let TILES = null;
function refreshBasemap() {
  const dark = isDark();
  if (TILES) MAP.removeLayer(TILES);
  TILES = L.tileLayer(
    `https://{s}.basemaps.cartocdn.com/${dark ? "dark_all" : "light_all"}/{z}/{x}/{y}{r}.png`,
    { maxZoom: 19, subdomains: "abcd",
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/attributions">CARTO</a>' }
  ).addTo(MAP);
  TILES.setZIndex(0);
}

function drawMarkers() {
  if (LAYER) MAP.removeLayer(LAYER);
  const key = $("#colorBy").value, urb = $("#filterUrban").value;
  const M = METRICS[key];
  const shown = DATA.clusters.filter(c => urb === "all" || String(c.urban) === urb);
  const vals = shown.map(M.get);
  const { lo, hi } = quantiles(vals);
  const amp = Math.max(...vals.map(v => Math.abs(v)), 1e-9);

  LAYER = L.layerGroup(shown.map(c => {
    const v = M.get(c);
    const fill = M.kind === "div" ? divColor(v / amp)
                                  : seqColor(hi === lo ? .5 : (v - lo) / (hi - lo));
    const m = L.circleMarker([c.lat, c.lon], {
      radius: c.cid === (SEL && SEL.cid) ? 9 : 6,
      fillColor: fill, fillOpacity: .9,
      color: c.cid === (SEL && SEL.cid) ? css("--text-primary") : css("--surface-1"),
      weight: 2,
    });
    m.on("click", () => selectCluster(c.cid));
    m.on("mouseover", e => showTip(e.originalEvent,
      `<b>Cluster ${c.cid}</b> <span class="r">${c.urban ? "urban" : "rural"}</span><br>` +
      `true <b>${fmt(c.y_true)}</b> &middot; pred <b>${fmt(c.y_pred)}</b><br>` +
      `<span class="r">HR acquired ${pct(c.hr_frac)}</span>`));
    m.on("mousemove", e => showTip(e.originalEvent, tip.innerHTML));
    m.on("mouseout", hideTip);
    return m;
  })).addTo(MAP);

  const ramp = M.kind === "div"
    ? [css("--div-lo"), css("--div-mid"), css("--div-hi")]
    : SEQ();
  $("#legend").innerHTML =
    `<div><b>${M.label}</b></div>
     <div class="ramp">${ramp.map(c => `<i style="background:${c}"></i>`).join("")}</div>
     <div class="ends"><span>${fmt(M.kind === "div" ? -amp : lo)}</span>
     <span>${fmt(M.kind === "div" ? amp : hi)}</span></div>`;

  const errs = shown.map(c => c.y_pred - c.y_true);
  const rmse = Math.sqrt(errs.reduce((a, e) => a + e * e, 0) / Math.max(errs.length, 1));
  $("#railStats").innerHTML = tiles([
    ["Clusters", shown.length, ""],
    ["Mean HR bought", pct(shown.reduce((a, c) => a + c.hr_frac, 0) / Math.max(shown.length, 1)), ""],
    ["RMSE", fmt(rmse), ""],
  ]);
}
const tiles = rows => rows.map(([k, v, s]) =>
  `<div class="tile"><div class="k">${k}</div><div class="v">${v}${s ? ` <small>${s}</small>` : ""}</div></div>`).join("");

/* ───────────────────────────── detail panel ───────────────────────────── */
function selectCluster(cid) {
  const c = DATA.clusters.find(x => x.cid === cid);
  if (!c) return;
  SEL = c;
  $("#detail").hidden = false;
  $("#dTitle").textContent = `Cluster ${c.cid}`;
  $("#dSub").innerHTML =
    `${c.lat.toFixed(4)}, ${c.lon.toFixed(4)} &middot; ${c.urban ? "urban" : "rural"} &middot; ` +
    `${c.households} households &middot; HR native ${c.hr_gsd.toFixed(2)} m/px ` +
    `&middot; ${c.out_of_fold ? `held out in ${c.n_oof} split${c.n_oof > 1 ? "s" : ""}` :
      '<span style="color:var(--warn)">in-sample prediction</span>'}`;

  const G = DATA.grid.side, nsel = c.mask.reduce((a, b) => a + b, 0);
  $("#dStats").innerHTML = tiles([
    ["True pc_cons", fmt(c.y_true), ""],
    ["Predicted", fmt(c.y_pred), ""],
    ["Error", (c.y_pred - c.y_true >= 0 ? "+" : "") + fmt(c.y_pred - c.y_true), ""],
    ["HR bought", `${nsel}`, `/ ${G * G}`],
    ["Objects found", fmt(c.counts_selected.reduce((a, b) => a + b, 0), 0),
      `of ${fmt(c.counts_full.reduce((a, b) => a + b, 0), 0)}`],
  ]);

  const ch = $("#dCountsHead");
  if (ch) ch.textContent = DATA.detector === "xview"
    ? "(xView parent classes, YOLOv3)" : "(COCO classes, YOLOv3)";
  $("#dLR").src = `data/${c.lr_img}`;
  $("#dHR").src = `data/${c.hr_img}`;
  gridOverlay($("#dProbGrid"), c, "prob");
  gridOverlay($("#dMaskGrid"), c, "mask");

  const rows = DATA.classes.map((n, i) => ({
    label: n, value: c.counts_selected[i],
    color: css("--series-1"),
  })).filter(r => r.value > 0 || c.counts_full[DATA.classes.indexOf(r.label)] > 0);
  if (rows.length) hBar($("#dCounts"), rows, { padLeft: 104, dp: 0, rowH: 21 });
  else $("#dCounts").innerHTML = `<p class="muted">No objects detected in the acquired tiles.</p>`;

  $("#dCountsTable").innerHTML =
    `<thead><tr><th>Class</th><th>Acquired tiles</th><th>All 64 tiles</th><th>Recall</th></tr></thead><tbody>` +
    DATA.classes.map((n, i) => {
      const a = c.counts_selected[i], f = c.counts_full[i];
      return `<tr><td>${n}</td><td>${fmt(a, 0)}</td><td>${fmt(f, 0)}</td>` +
             `<td>${f > 0 ? pct(a / f) : "—"}</td></tr>`;
    }).join("") + `</tbody>`;

  try { if (MAP) { drawMarkers(); MAP.panTo([c.lat, c.lon]); } } catch (e) { console.warn(e); }
}
$("#detailClose").addEventListener("click", () => {
  $("#detail").hidden = true; SEL = null;
  try { if (MAP) drawMarkers(); } catch (e) { console.warn(e); }
});

/* 8x8 overlay: acquisition probabilities, or the binary mask as a scrim. */
function gridOverlay(svg, c, mode) {
  const G = DATA.grid.side;
  svg.setAttribute("viewBox", `0 0 ${G} ${G}`);
  svg.innerHTML = "";
  const gap = 0.035;
  for (let i = 0; i < G * G; i++) {
    const r = Math.floor(i / G), col = i % G;
    const objs = c.counts_full.length ? null : null;
    const rect = mk("rect", { x: col + gap, y: r + gap, width: 1 - 2 * gap, height: 1 - 2 * gap, rx: .06 });
    if (mode === "prob") {
      rect.setAttribute("fill", css("--accent"));
      rect.setAttribute("fill-opacity", (0.06 + 0.72 * c.probs[i]).toFixed(3));
    } else if (c.mask[i]) {
      rect.setAttribute("fill", "none");
      rect.setAttribute("stroke", css("--series-2"));
      rect.setAttribute("stroke-width", .07);
    } else {
      rect.setAttribute("fill", css("--surface-0"));
      rect.setAttribute("fill-opacity", .82);
    }
    const label = `<b>Subtile ${i}</b> <span class="r">(row ${r}, col ${col})</span><br>` +
      `acquisition p = <b>${fmt(c.probs[i], 3)}</b><br>` +
      `<b>${c.mask[i] ? "acquired" : "dropped"}</b>` +
      `<br><span class="r">LR excess-green ${fmt(c.lr_exg[i], 1)}</span>`;
    hoverable(rect, label);
    svg.appendChild(rect);
  }
}

/* ───────────────────────────── results view ───────────────────────────── */
const METHOD_LABEL = {
  rl: "RL policy (ours)", rl_matched: "RL policy (matched training)",
  no_dropping: "No dropping", fixed_center: "Fixed (centre)",
  random: "Random", stochastic: "Stochastic", green_tiles: "Green tiles",
  counts_pred: "Counts prediction", lr_only: "LR only (no HR)",
};

const TARGET_LABEL = { pc_cons: "consumption (pc_cons)", log_pc_cons: "log consumption" };
const targets = () => (DATA.targets || ["pc_cons"]).filter(t =>
  t === "pc_cons" || (DATA.results_pivot_log || []).length);
const curTarget = () => {
  const sel = $("#targetSel");
  return (sel && sel.value) || "pc_cons";
};

/* One <select> to switch the whole Results view between regression targets. Built once, from
   whatever targets the run actually produced -- an older run with a single target gets none. */
function buildTargetPicker() {
  if ($("#targetSel") || targets().length < 2) return;
  const host = $("#view-results").querySelector(".hero");
  const d = document.createElement("div");
  d.className = "field";
  d.style.cssText = "margin:12px 0 0;max-width:280px";
  d.innerHTML = `<label for="targetSel">Regression target</label>
    <select id="targetSel">${targets().map(t =>
      `<option value="${t}">${TARGET_LABEL[t] || t}</option>`).join("")}</select>`;
  host.appendChild(d);
  const sel = d.querySelector("select");     // not $("#targetSel"): never trust a re-query
  if (sel) sel.addEventListener("change", () => {
    $$(".legend-row").forEach(e => e.remove());
    renderResults();
  });
}

function renderResults() {
  buildTargetPicker();
  const TGT = curTarget();
  const piv = (TGT === "log_pc_cons" ? DATA.results_pivot_log : DATA.results_pivot) || [];
  const get = (m, v) => { const r = piv.find(x => x.method === m); return r ? r[v] : null; };
  const rl_a = get("rl", "counts_only"), rl_b = get("rl", "counts_plus_lr");
  const rl_c = get("rl", "countsall_plus_lr");
  const nd_b = get("no_dropping", "counts_plus_lr");
  const T = DATA.results_table.filter(r => (r.target || "pc_cons") === TGT);
  const lr_o = get("lr_only", "lr_only");
  const hr = (T.find(r => r.method === "rl" && r.variant === "counts_plus_lr") || {}).hr;

  $("#heroTiles").innerHTML = tiles([
    ["RL, counts only (Eq. 1 literal)", fmt(rl_a, 3), "r²"],
    ["RL, counts + LR (fixed)", fmt(rl_b, 3), "r²"],
    ["Improvement", (rl_b - rl_a >= 0 ? "+" : "") + fmt(rl_b - rl_a, 3), "r²"],
    ["LR only, nothing bought", fmt(lr_o, 3), "r²"],
    ["HR imagery purchased", hr != null ? pct(hr) : "—", ""],
    ["Counts over every subtile", fmt(get("no_dropping", "counts_only"), 3), "r²"],
    ["All-HR upper bound", fmt(nd_b, 3), "r²"],
  ]);

  const order = ["rl", "rl_matched", "no_dropping", "counts_pred", "green_tiles",
                 "fixed_center", "stochastic", "random"];
  const cats = order.filter(m => piv.some(x => x.method === m)).map(m => METHOD_LABEL[m] || m);
  const keys = order.filter(m => piv.some(x => x.method === m));
  const S1 = css("--series-1"), S2 = css("--series-2"), S3 = css("--series-3");
  // v_all == v whenever every detector class is already a feature class, and a third series
  // of identical bars is just clutter -- only show it when it actually differs.
  const has80 = piv.some(x => x.countsall_plus_lr != null && x.counts_plus_lr != null &&
                              Math.abs(x.countsall_plus_lr - x.counts_plus_lr) > 1e-9);
  const seriesDefs = [
    { name: "counts only (Eq. 1 literal)", short: "counts only", color: S1, key: "counts_only" },
    { name: "counts + LR features (fix)",  short: "counts + LR", color: S2, key: "counts_plus_lr" },
  ];
  if (has80) seriesDefs.push(
    { name: "all detector classes + LR", short: "all classes + LR", color: S3, key: "countsall_plus_lr" });
  legendRow($("#chartMethods"), seriesDefs.map(d => ({ name: d.name, color: d.color })));
  groupedBar($("#chartMethods"), cats,
    seriesDefs.map(d => ({ name: d.short, color: d.color, values: keys.map(m => get(m, d.key)) })),
    { height: 310, yLabel: "Pearson r²" });

  $("#tableMethods").innerHTML =
    `<thead><tr><th>Method</th><th>Variant</th><th>r²</th><th>± sd</th><th>MSE</th>
      <th>Expl. var.</th><th>Spearman</th><th>HR frac.</th></tr></thead><tbody>` +
    T.map(r => `<tr class="${r.method === "rl" && r.variant === "counts_plus_lr" ? "is-hi" : ""}">
       <td>${METHOD_LABEL[r.method] || r.method}</td><td>${r.variant}</td>
       <td>${fmt(r.r2, 3)}</td><td>${fmt(r.r2_sd, 3)}</td><td>${fmt(r.mse, 2)}</td>
       <td>${fmt(r.ev, 3)}</td><td>${fmt(r.spearman, 3)}</td><td>${fmt(r.hr, 2)}</td></tr>`).join("") +
    `</tbody>`;

  const dc = DATA.detector_caveat, vd = DATA.vehicle_dets;
  if (dc) {
    const isAbl = DATA.detector === "coco";
    $("#detectorCard").hidden = false;
    $("#detectorCard").classList.toggle("warn", isAbl);
    $("#detectorCard").querySelector("h2").textContent = isAbl
      ? "Ablation run — class names are textures, not objects"
      : "Detector: YOLOv3, COCO → xView parent classes";
    $("#detectorText").textContent = dc;
    if (vd) $("#vehTable").innerHTML =
      `<thead><tr><th>Class</th><th>Detections</th></tr></thead><tbody>` +
      Object.entries(vd).sort((a, b) => b[1] - a[1])
        .map(([n, c]) => `<tr><td>${n}</td><td>${c}</td></tr>`).join("") + `</tbody>`;

    const dr = DATA.detector_report;
    if (dr && dr.map50 != null) {
      $("#detMetricsWrap").hidden = false;
      $("#detTiles").innerHTML = tiles([
        ["mAP@50", fmt(dr.map50, 3), ""],
        ["mAP@50-95", fmt(dr.map50_95, 3), ""],
        ["Train chips", (dr.train_images || 0).toLocaleString(), ""],
        ["Train time", fmt(dr.hours, 1), "h"],
      ]);
      $("#detTable").innerHTML =
        `<thead><tr><th>xView parent</th><th>Instances</th><th>AP@50</th><th>AP@50-95</th></tr></thead><tbody>` +
        (dr.per_class || []).map(r =>
          `<tr><td>${r.parent}</td><td>${(r.instances || 0).toLocaleString()}</td>` +
          `<td>${fmt(r.ap50, 3)}</td><td>${fmt(r.ap, 3)}</td></tr>`).join("") + `</tbody>`;
    }
  }

  const sw = DATA.lambda_sweep || [];
  if (sw.length) {
    const isLog = TGT === "log_pc_cons";
    const kCO = isLog && "r2log_counts_only" in sw[0] ? "r2log_counts_only" : "r2_counts_only";
    const kLR = isLog && "r2log_counts_plus_lr" in sw[0] ? "r2log_counts_plus_lr" : "r2_counts_plus_lr";
    legendRow($("#chartSweep"), [{ name: "counts only", color: S1 }, { name: "counts + LR", color: S2 }]);
    lineChart($("#chartSweep"), [
      { name: "counts only", color: S1,
        points: sw.map(r => ({ x: r.hr_frac, y: r[kCO], note: `λ = ${r.lam}` })) },
      { name: "counts + LR", color: S2,
        points: sw.map(r => ({ x: r.hr_frac, y: r[kLR], label: `λ${r.lam}`, note: `λ = ${r.lam}` })) },
    ], { xLabel: "HR acquisition fraction", yLabel: "r²", height: 270 });
  }

  const pts = DATA.clusters.filter(c => c.out_of_fold)
                           .map(c => ({ x: c.y_true, y: c.y_pred, cid: c.cid }));
  scatter($("#chartScatter"), pts,
          { xLabel: "True pc_cons", yLabel: "Predicted", height: 270 });

  const curves = DATA.rl_curves || {};
  const seeds = Object.keys(curves);
  if (seeds.length) {
    legendRow($("#chartCurves"), [{ name: "reward", color: S1 }, { name: "HR fraction", color: S2 }]);
    lineChart($("#chartCurves"),
      seeds.flatMap(s => ([
        { name: `reward (seed ${s})`, color: S1, opacity: .5, width: 1.5, noMarkers: true,
          points: curves[s].map(p => ({ x: p.epoch, y: p.reward - Math.min(...curves[s].map(q => q.reward)) })) },
        { name: `HR fraction (seed ${s})`, color: S2, opacity: .5, width: 1.5, noMarkers: true,
          points: curves[s].map(p => ({ x: p.epoch, y: p.hr_frac })) },
      ])),
      { xLabel: "epoch", yLabel: "value", height: 250, xFmt: v => v.toFixed(0) });
  }

  const f3 = DATA.fig3_missed || [];
  if (f3.length) hBar($("#chartMissed"),
    f3.map(r => ({ label: r.class, value: r.missed_by_RL, color: S2 })),
    { padLeft: 110, dp: 1, unit: "missed / cluster " });

  const sh = (DATA.shap || []).slice(0, 14);
  if (sh.length) hBar($("#chartShap"),
    sh.map(r => ({ label: r.feature, value: r.mean_abs_shap, color: S1 })),
    { padLeft: 150, dp: 3, unit: "mean |SHAP| " });
  else $("#chartShap").innerHTML = `<p class="muted">SHAP was not computed in this run.</p>`;

  const gl = DATA.gsd_leakage;
  if (gl) {
    $("#leakCard").hidden = false;
    $("#leakText").innerHTML =
      `The HR scrape used a non-uniform zoom level and imagery is kept at <b>native resolution</b>.
       Every subtile covers the same ground footprint and is resized to a common input size
       before detection, so the effective inference resolution is
       <b>${gl.eff_gsd != null ? fmt(gl.eff_gsd, 3) + " m/px" : "identical"}</b> for every cluster —
       scrape zoom does <em>not</em> change apparent object size. It changes <b>sharpness</b>:
       coarse clusters are upsampled and blurry.
       Correlation between native GSD and true consumption is
       <b>r = ${fmt(gl.pearson_r, 3)}</b> (p = ${fmt(gl.p_value, 3)}), which bounds how much
       signal could ride on that sharpness difference.
       ${Math.abs(gl.pearson_r) > 0.2
          ? "That is large enough to treat the absolute r² with caution."
          : "That is small, so sharpness is unlikely to be doing the work."}`;
  }

  $("#devTable").innerHTML =
    `<thead><tr><th>Paper</th><th>This run</th></tr></thead><tbody>` +
    [["34×34 = 1156 HR subtiles per cluster (10 km × 10 km)",
      `${DATA.grid.side}×${DATA.grid.side} = ${DATA.grid.subtiles} subtiles (1.113 km × 1.113 km)`],
     ["YOLOv3 trained on xView, 10 parent classes",
      DATA.detector === "xview"
        ? `<b>same</b> — COCO weights fine-tuned on xView parents, L = ${DATA.classes.length}: ${DATA.classes.join(", ")}`
        : `<span style="color:var(--warn)">ablation</span> — raw COCO YOLOv3, no xView fine-tuning; L = ${DATA.classes.length}: ${DATA.classes.join(", ")}`],
     ["DigitalGlobe 0.3 m HR imagery", "leafmap HR scrape, native GSD retained (0.30–1.19 m)"],
     ["Sentinel-2 10 m LR imagery", "Sentinel-2, 9.51 m/px"],
     ["N = 320 clusters", `N = ${DATA.n_clusters} clusters`]]
      .map(([a, b]) => `<tr><td>${a}</td><td>${b}</td></tr>`).join("") + `</tbody>`;

  $("#cfgTable").innerHTML =
    `<thead><tr><th>Key</th><th>Value</th></tr></thead><tbody>` +
    Object.entries(DATA.config).map(([k, v]) =>
      `<tr><td>${k}</td><td>${Array.isArray(v) ? v.join(", ") : v}</td></tr>`).join("") + `</tbody>`;
}

let drawnOnce = false;
function redrawAll() {
  if (!DATA) return;
  if (MAP) { try { refreshBasemap(); drawMarkers(); } catch (e) { console.warn("map redraw:", e); } }
  if (!$("#view-results").hidden || !drawnOnce) {
    $$(".legend-row").forEach(e => e.remove());
    renderResults(); drawnOnce = true;
    if (SEL) { gridOverlay($("#dProbGrid"), SEL, "prob"); gridOverlay($("#dMaskGrid"), SEL, "mask"); }
  }
}
let rt; addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(redrawAll, 180); });
const safeMarkers = () => { try { if (MAP) drawMarkers(); } catch (e) { console.warn(e); } };
$("#colorBy").addEventListener("change", safeMarkers);
$("#filterUrban").addEventListener("change", safeMarkers);

/* ───────────────────────────── boot ───────────────────────────────────── */
fetch("data/data.json")
  .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
  .then(d => {
    DATA = d;
    window.__povrlData = true;          // tells the inline failsafe the fetch landed
    $("#loading").hidden = true;
    showView(ACTIVE);
    renderResults(); drawnOnce = true;
    $("#footMeta").innerHTML =
      `${d.n_clusters} clusters &middot; ${d.grid.subtiles} HR subtiles each ` +
      `(${(d.grid.tile_km * 1000).toFixed(0)} m per subtile) &middot; run ${d.generated} &middot; ` +
      `<a href="https://ojs.aaai.org/index.php/AAAI/article/view/16072">Ayush et&nbsp;al., AAAI 2021</a>`;
  })
  .catch(err => {
    $("#loading").hidden = true;
    if (ACTIVE !== "report") $("#nodata").hidden = false;
    const isFile = location.protocol === "file:";
    $("#nodata").innerHTML = isFile
      ? `<h2>Open this over HTTP, not from disk</h2>
         <p>The page is loaded via <code>file://</code>, and browsers block
            <code>fetch()</code> on local files &mdash; so <code>data/data.json</code> cannot be
            read even though it is sitting right there.</p>
         <p>From the folder containing <code>index.html</code>, run:</p>
         <p><code>python -m http.server 8000</code></p>
         <p>then open <a href="http://localhost:8000">http://localhost:8000</a>.
            Deploying to Netlify also works, because that serves over HTTPS.</p>`
      : `<h2>Run data not found</h2>
         <p>Could not load <code>data/data.json</code> (${err && err.message ? err.message : "unknown error"}).</p>
         <p>The site expects <code>data/data.json</code> plus per-cluster thumbnails in
            <code>data/clusters/</code>, alongside <code>index.html</code>.</p>`;
  });
})();
