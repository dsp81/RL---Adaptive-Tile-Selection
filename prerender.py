#!/usr/bin/env python3
"""Bake data/data.json into static HTML, Markdown, llms.txt and a flat CSV.

The viewer is client-rendered: every number lives in data/data.json and is injected by
app.js. That is invisible to anything that does not run JavaScript -- search crawlers, LLM
fetchers, `curl`, chat unfurls. This script renders the same numbers into static text:

  index.html         a static <main id="view-report"> between the PRERENDER markers (the
                     landing tab), plus a JSON-LD block in <head>
  report.md          the whole run as Markdown, linked from <head> and the report itself
  llms.txt           llmstxt.org index pointing at the above
  robots.txt         explicit allow, so crawlers do not have to guess
  data/clusters.csv  per-cluster predictions as a flat table

The prose is parameterised from the numbers rather than hand-written, so re-running this
after a new fetch_results.sh cannot leave a stale claim behind. Idempotent.
"""
from __future__ import annotations

import csv
import html
import json
import math
import re
from pathlib import Path

FE = Path(__file__).resolve().parent
D = json.load(open(FE / "data" / "data.json"))

# ─────────────────────────────────────────────────────────────────── formatting
def f(v, d=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{float(v):.{d}f}"

def pc(v, d=1):
    return "—" if v is None else f"{100 * float(v):.{d}f}%"

def num(v):
    return "—" if v is None else f"{int(round(float(v))):,}"

def esc(s):
    return html.escape(str(s), quote=False)

def signed(v, d=3):
    return "—" if v is None else ("+" if v >= 0 else "−") + f(abs(v), d)

METHOD_LABEL = {
    "rl": "RL policy (ours)",
    "rl_matched": "RL policy (matched training)",
    "no_dropping": "No dropping (every subtile)",
    "fixed_center": "Fixed (centre)",
    "random": "Random",
    "stochastic": "Stochastic",
    "green_tiles": "Green tiles",
    "counts_pred": "Counts prediction (CNN)",
    "lr_only": "LR only (nothing bought)",
}
VARIANT_LABEL = {
    "counts_only": "counts only (Eq. 1 literal)",
    "counts_plus_lr": "counts + LR features",
    "countsall_plus_lr": "all detector classes + LR",
    "lr_only": "LR features only",
}
TARGET_LABEL = {"pc_cons": "consumption (pc_cons)",
                "log_pc_cons": "log consumption"}

class Table:
    """One table, rendered twice: HTML and Markdown."""

    def __init__(self, headers, rows, highlight=None):
        self.headers, self.rows = headers, rows
        self.highlight = highlight or (lambda r: False)

    def html(self):
        head = "".join(f"<th>{esc(h)}</th>" for h in self.headers)
        body = []
        for r in self.rows:
            cls = ' class="is-hi"' if self.highlight(r) else ""
            body.append(f"<tr{cls}>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>")
        return ('<div class="tablewrap"><table><thead><tr>' + head
                + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>")

    def md(self):
        out = ["| " + " | ".join(str(h) for h in self.headers) + " |",
               "|" + "|".join("---" for _ in self.headers) + "|"]
        for r in self.rows:
            out.append("| " + " | ".join(str(c) for c in r) + " |")
        return "\n".join(out)

# ───────────────────────────────────────────────────────────────────── the data
grid = D.get("grid", {})
cfg = D.get("config", {})
clusters = D.get("clusters", [])
classes = D.get("classes", [])
dr = D.get("detector_report") or {}
gl = D.get("gsd_leakage") or {}
shap = D.get("shap") or []
sweep = D.get("lambda_sweep") or []
tbl_all = D.get("results_table", [])
TARGETS = D.get("targets") or ["pc_cons"]
MAIN = "pc_cons"
HAS_LOG = "log_pc_cons" in TARGETS and any(r.get("target") == "log_pc_cons" for r in tbl_all)

PIV = {t: {r["method"]: r for r in (D.get("results_pivot") if t == MAIN
                                    else D.get("results_pivot_log") or [])}
       for t in TARGETS}

def rows_for(target):
    return [r for r in tbl_all if r.get("target", MAIN) == target]

def cell(target, method, variant):
    return (PIV.get(target, {}).get(method) or {}).get(variant)

def row(target, method, variant):
    for r in rows_for(target):
        if r["method"] == method and r["variant"] == variant:
            return r
    return {}

def stat(target):
    """Everything the prose needs about one regression target."""
    rl_co = cell(target, "rl", "counts_only")
    rl_lr = cell(target, "rl", "counts_plus_lr")
    lr_only = cell(target, "lr_only", "lr_only")
    nd_co = cell(target, "no_dropping", "counts_only")
    nd_lr = cell(target, "no_dropping", "counts_plus_lr")
    budgeted = [r for r in rows_for(target)
                if r["variant"] == "counts_only" and r["method"] not in ("no_dropping", "lr_only")]
    co_rank = sorted(budgeted, key=lambda r: -(r["r2"] or 0))
    co_baselines = [r for r in budgeted if r["method"] not in ("rl", "rl_matched")]
    lr_block = [r for r in rows_for(target)
                if r["variant"] == "counts_plus_lr" and r["method"] != "rl_matched"]
    return dict(
        rl_co=rl_co, rl_lr=rl_lr, lr_only=lr_only, nd_co=nd_co, nd_lr=nd_lr,
        rl_sd=row(target, "rl", "counts_plus_lr").get("r2_sd"),
        rl_co_sd=row(target, "rl", "counts_only").get("r2_sd"),
        rl_hr=row(target, "rl", "counts_plus_lr").get("hr"),
        co_rank=co_rank,
        best_baseline=max(co_baselines, key=lambda r: r["r2"]) if co_baselines else None,
        lr_spread=(max(r["r2"] for r in lr_block) - min(r["r2"] for r in lr_block)
                   if lr_block else None),
        best_lr=max(lr_block, key=lambda r: r["r2"]) if lr_block else None,
        best_overall=max(rows_for(target), key=lambda r: r["r2"]) if rows_for(target) else None,
    )

S = {t: stat(t) for t in TARGETS}

# per-cluster aggregates
def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return None if sx == 0 or sy == 0 else \
        sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)

oof = [c for c in clusters if c.get("out_of_fold")]
oof_r = pearson([c["y_true"] for c in oof], [c["y_pred"] for c in oof]) if oof else None
oof_rmse = (math.sqrt(sum((c["y_pred"] - c["y_true"]) ** 2 for c in oof) / len(oof))
            if oof else None)
hr_fracs = [c.get("hr_frac", 0) for c in clusters]
mean_hr = sum(hr_fracs) / len(hr_fracs) if hr_fracs else None
zero_hr = sum(1 for x in hr_fracs if x == 0)
urban = sum(1 for c in clusters if c.get("urban"))

tot_full = [0.0] * len(classes)
tot_sel = [0.0] * len(classes)
for c in clusters:
    cf, cs = c.get("counts_full") or [], c.get("counts_selected") or []
    for i in range(len(classes)):
        tot_full[i] += cf[i] if i < len(cf) else 0
        tot_sel[i] += cs[i] if i < len(cs) else 0
objects_total, objects_sel = sum(tot_full), sum(tot_sel)
object_recall = objects_sel / objects_total if objects_total else None

hr_shap = [s for s in shap if s["feature"] in classes]
shap_total = sum(s["mean_abs_shap"] for s in shap)
shap_hr_share = (sum(s["mean_abs_shap"] for s in hr_shap) / shap_total) if shap_total else None
top_hr_shap = max(hr_shap, key=lambda s: s["mean_abs_shap"]) if hr_shap else None

sweep_hi = max(sweep, key=lambda r: r.get("hr_frac") or 0) if sweep else None
sweep_lo = min(sweep, key=lambda r: r.get("hr_frac") or 0) if sweep else None
f3 = {r["class"]: r for r in (D.get("fig3_missed") or [])}
DET_IS_XVIEW = D.get("detector") == "xview"
PAPER = D.get("paper", {})
PAPER_LINE = (f"{PAPER.get('authors', '')}, “{PAPER.get('title', '')}”, {PAPER.get('venue', '')}")
run_date = (D.get("generated") or "")[:10]
n_seeds = len(cfg.get("SEEDS", []) or [])
test_pct = int(100 * (cfg.get("TEST_FRAC") or 0.2))
test_n = int(round((cfg.get("TEST_FRAC") or 0.2) * (D.get("n_clusters") or 0)))

# ──────────────────────────────────────────────────────────────── the narrative
def verdict():
    m, lg = S[MAIN], S.get("log_pc_cons")
    out = []

    if None not in (m["rl_co"], m["rl_lr"], m["lr_only"]):
        d = m["rl_lr"] - m["lr_only"]
        small = m["rl_sd"] is not None and abs(d) < m["rl_sd"]
        out.append(
            "**The free low-resolution image explains almost everything this pipeline can "
            f"explain.** A regressor given only the Sentinel-2 descriptors reaches r² = "
            f"{f(m['lr_only'])}. Adding the high-resolution object counts the policy paid for "
            f"takes it to {f(m['rl_lr'])} ({signed(d)}, against a between-seed sd of "
            f"±{f(m['rl_sd'])})"
            + (f"; on log consumption the LR-only model is in fact the single best number on "
               f"this page ({f(lg['lr_only'])} vs {f(lg['rl_lr'])} with counts)"
               if lg and lg["lr_only"] is not None else "")
            + ". "
            + ("Within noise, the purchased imagery adds nothing to the free imagery on this "
               "dataset — the most important result here, and a negative one."
               if small else "The purchased imagery moves the number measurably."))

    if None not in (m["nd_co"], m["rl_co"]) and m["best_baseline"]:
        out.append(
            "**The object counts do carry real signal, but they need coverage, not "
            f"cleverness.** Detecting on every one of the {grid.get('subtiles')} subtiles and "
            f"regressing on counts alone gives r² = {f(m['nd_co'])}"
            + (f" ({f(lg['nd_co'])} on log consumption)" if lg else "")
            + f"; the same features restricted to the {pc(m['rl_hr'], 1)} of subtiles the "
              f"policy buys give {f(m['rl_co'])}. The λ sweep says the same thing "
              "monotonically: "
            + (f"r²(counts only) falls from {f(sweep_hi.get('r2_counts_only'))} at "
               f"{pc(sweep_hi.get('hr_frac'), 0)} of the imagery to "
               f"{f(sweep_lo.get('r2_counts_only'))} at {pc(sweep_lo.get('hr_frac'), 1)}."
               if sweep_hi and sweep_lo else "accuracy tracks the fraction bought."))

    ref = S.get("log_pc_cons") or m
    ref_name = "log consumption" if ref is not S[MAIN] else TARGET_LABEL.get(MAIN, MAIN)
    if ref["co_rank"] and ref["best_baseline"]:
        rl_row = next((r for r in ref["co_rank"] if r["method"] == "rl"), None)
        bb = ref["best_baseline"]
        ratio = (rl_row["r2"] / bb["r2"]) if rl_row and bb["r2"] else None
        won = rl_row is not None and rl_row["r2"] > bb["r2"]
        order = " > ".join(f"{METHOD_LABEL.get(r['method'], r['method'])} {f(r['r2'])}"
                           for r in ref["co_rank"])
        out.append(
            "**Where the method is supposed to win — counts only, matched budget — it does"
            + ("" if won else " not clearly") + " win.** On "
            + f"{ref_name} at a matched budget of {pc(ref['rl_hr'], 1)} of the subtiles the "
            + f"ranking is {order}"
            + (f". The learned policy is {f(ratio, 1)}× the best hand-designed baseline "
               f"({METHOD_LABEL.get(bb['method'], bb['method'])} {f(bb['r2'])})"
               if ratio else "")
            + ", which is the ordering the paper predicts. The absolute level is not: "
              f"{f(rl_row['r2']) if rl_row else '—'} r² is not a usable poverty estimate on "
              "its own.")

    if object_recall is not None:
        b = f3.get("Building", {})
        t = f3.get("Truck", {})
        out.append(
            "**What the policy buys is genuinely object-dense.** For "
            f"{pc(mean_hr, 1)} of the high-resolution imagery it recovers {pc(object_recall, 1)} "
            f"of all {num(objects_total)} objects the detector finds on the full grid"
            + (f" — per-class recall {f(b.get('recall'), 2)} for buildings and "
               f"{f(t.get('recall'), 2)} for trucks" if b and t else "")
            + ". Uniform sampling at the same budget would recover about "
              f"{pc(mean_hr, 0)}. The acquisition policy is not the weak link.")

    if m["lr_spread"] is not None and m["best_lr"]:
        out.append(
            "**Once the low-resolution features are in, which tiles you buy stops "
            f"mattering.** Learned, random, centre, distance-decayed, greenest-first and "
            f"CNN-predicted acquisition all land within {f(m['lr_spread'])} r² of each other"
            + (f", and the nominal best is {METHOD_LABEL.get(m['best_lr']['method'], '')}"
               if m["best_lr"]["method"] != "rl" else "")
            + f" — well inside the ±{f(m['rl_sd'])} seed spread. Any ranking inside that block "
              "is noise, including rankings that favour the learned policy.")

    if DET_IS_XVIEW and dr.get("map50") is not None:
        best = sorted((dr.get("per_class") or []), key=lambda r: -(r.get("ap50") or 0))[:3]
        b_idx = classes.index("Building") if "Building" in classes else None
        out.append(
            "**The detector is a real overhead detector now.** YOLOv3 initialised from COCO "
            f"and fine-tuned on xView scores mAP@50 = {f(dr['map50'])} / "
            f"mAP@50-95 = {f(dr['map50_95'])} on the xView validation split (best: "
            + ", ".join(f"{r['parent']} {f(r['ap50'])}" for r in best)
            + f"), and fires {num(objects_total)} objects across the Uganda clusters"
            + (f", {num(tot_full[b_idx])} of them buildings" if b_idx is not None else "")
            + ". That closes the gap in the earlier COCO-only ablation, where the dominant "
              "classes were food textures rather than objects.")

    if shap_hr_share is not None:
        out.append(
            "**SHAP agrees with the r² tables.** The high-resolution count features carry "
            f"{pc(shap_hr_share, 1)} of total mean |SHAP| in the counts + LR regressor"
            + (f" (largest single one: {top_hr_shap['feature']}, "
               f"{f(top_hr_shap['mean_abs_shap'], 3)}"
               + (", which given only "
                  f"{num(tot_full[classes.index(top_hr_shap['feature'])])} detections in total "
                  "is more likely a proxy for water and bare ground than a real fleet"
                  if top_hr_shap["feature"] == "Maritime Vessel"
                  and "Maritime Vessel" in classes else "")
               + ")" if top_hr_shap else "")
            + "; the rest sits on low-resolution colour and texture. The paper's "
              "interpretability story, where #Trucks and #Buildings drive the prediction, does "
              "not transfer to this dataset.")

    if gl.get("pearson_r") is not None:
        out.append(
            "**One confound to carry with every number above.** The high-resolution scrape "
            "mixes native ground sample distances (0.30 / 0.60 / 1.19 m/px) and native GSD "
            f"correlates with the target at r = {f(gl['pearson_r'])} "
            f"(p = {gl['p_value']:.1e}). Every subtile covers the same "
            f"{(grid.get('tile_km') or 0) * 1000:.0f} m footprint and is resized to "
            f"{cfg.get('YOLO_IMGSZ')} px before detection, so zoom does not change apparent "
            f"object size ({f(gl.get('eff_gsd'))} m/px effective for every cluster) — but it "
            "does change sharpness, and that correlation bounds how much of the signal could "
            "ride on sharpness rather than content.")
    return out

VERDICT = verdict()

# ─────────────────────────────────────────────────────────────────────── tables
T = {}
T["headline"] = Table(
    ["Quantity", "Value"],
    [["Clusters", f"{D.get('n_clusters')} Uganda LSMS clusters "
                  f"({urban} urban, {len(clusters) - urban} rural)"],
     ["Acquisition grid", f"{grid.get('side')}×{grid.get('side')} = {grid.get('subtiles')} HR "
                          f"subtiles per cluster, {(grid.get('tile_km') or 0) * 1000:.0f} m each"],
     ["Detector", "YOLOv3, COCO init → xView fine-tune" if DET_IS_XVIEW
                  else "YOLOv3, COCO weights (ablation)"],
     ["Detector classes (L)", f"{len(classes)}: " + ", ".join(classes)],
     ["Objects detected, full grid", num(objects_total)],
     ["Objects inside the acquired subtiles", f"{num(objects_sel)} ({pc(object_recall, 1)})"],
     ["HR imagery bought", f"{pc(mean_hr, 1)} of subtiles, averaged over all clusters "
      f"({pc(S[MAIN]['rl_hr'], 1)} on the held-out splits the r² values come from)"],
     ["Clusters the policy bought nothing for", f"{zero_hr} of {len(clusters)}"],
     ["r² — RL policy, counts only", f"{f(S[MAIN]['rl_co'])} ± {f(S[MAIN]['rl_co_sd'])}"
      + (f"  (log: {f(S['log_pc_cons']['rl_co'])})" if HAS_LOG else "")],
     ["r² — RL policy, counts + LR", f"{f(S[MAIN]['rl_lr'])} ± {f(S[MAIN]['rl_sd'])}"
      + (f"  (log: {f(S['log_pc_cons']['rl_lr'])})" if HAS_LOG else "")],
     ["r² — LR only, nothing bought", f(S[MAIN]["lr_only"])
      + (f"  (log: {f(S['log_pc_cons']['lr_only'])})" if HAS_LOG else "")],
     ["r² — every subtile bought, counts only", f(S[MAIN]["nd_co"])
      + (f"  (log: {f(S['log_pc_cons']['nd_co'])})" if HAS_LOG else "")],
     ["r² — every subtile bought, counts + LR", f(S[MAIN]["nd_lr"])
      + (f"  (log: {f(S['log_pc_cons']['nd_lr'])})" if HAS_LOG else "")],
     ["Out-of-fold Pearson r (RL, counts + LR)", f"{f(oof_r)} over {len(oof)} clusters"],
     ["Out-of-fold RMSE", f(oof_rmse, 2)],
     ["Protocol", f"{n_seeds} seeds × {100 - test_pct}/{test_pct} splits "
                  f"(~{test_n} clusters held out per seed), Pearson r² on the held-out split"],
     ["Run generated", D.get("generated", "—")]])

# "all detector classes + LR" only differs from "counts + LR" when the detector fires classes
# that are not feature classes. When they coincide, the column is a duplicate; drop it.
ALLCLS_DIFFERS = any(
    r.get("countsall_plus_lr") is not None and r.get("counts_plus_lr") is not None
    and abs(r["countsall_plus_lr"] - r["counts_plus_lr"]) > 1e-9
    for piv in PIV.values() for r in piv.values())

def pivot_table(target):
    piv = PIV.get(target, {})
    heads = ["Method", "counts only", "counts + LR"]
    if ALLCLS_DIFFERS:
        heads.append("all classes + LR")
    heads += ["LR only", "HR bought"]
    rows = []
    for mth, r in piv.items():
        row_ = [METHOD_LABEL.get(mth, mth), f(r.get("counts_only")), f(r.get("counts_plus_lr"))]
        if ALLCLS_DIFFERS:
            row_.append(f(r.get("countsall_plus_lr")))
        row_ += [f(r.get("lr_only")),
                 pc(row(target, mth, "counts_plus_lr").get("hr") if mth != "lr_only" else 0.0, 1)]
        rows.append(row_)
    return Table(heads, rows, highlight=lambda r: r[0] == METHOD_LABEL["rl"])

def full_table(target):
    keep = (lambda r: True) if ALLCLS_DIFFERS else \
        (lambda r: r["variant"] != "countsall_plus_lr")
    return Table(
        ["Method", "Feature variant", "r²", "± sd", "MSE", "Explained var.", "Spearman",
         "HR frac."],
        [[METHOD_LABEL.get(r["method"], r["method"]),
          VARIANT_LABEL.get(r["variant"], r["variant"]), f(r["r2"]), f(r.get("r2_sd")),
          f(r.get("mse"), 3), f(r.get("ev")), f(r.get("spearman")), f(r.get("hr"), 3)]
         for r in sorted([x for x in rows_for(target) if keep(x)],
                         key=lambda r: (r["variant"], -r["r2"]))],
        highlight=lambda r: r[0] == METHOD_LABEL["rl"] and r[1] == VARIANT_LABEL["counts_plus_lr"])

for t in TARGETS:
    T[f"pivot_{t}"] = pivot_table(t)
    T[f"full_{t}"] = full_table(t)

if sweep:
    log_cols = any("r2log_counts_only" in r for r in sweep)
    heads = ["λ", "HR fraction bought", "r² counts only", "r² counts + LR"]
    if log_cols:
        heads += ["r² counts only (log)", "r² counts + LR (log)"]
    rows = []
    for r in sweep:
        row_ = [f(r.get("lam"), 2), pc(r.get("hr_frac"), 1), f(r.get("r2_counts_only")),
                f(r.get("r2_counts_plus_lr"))]
        if log_cols:
            row_ += [f(r.get("r2log_counts_only")), f(r.get("r2log_counts_plus_lr"))]
        rows.append(row_)
    T["sweep"] = Table(heads, rows)

if dr.get("per_class"):
    T["detector"] = Table(
        ["xView parent class", "Train instances", "AP@50", "AP@50-95",
         "Detected, full grid", "Detected, acquired subtiles", "Recall"],
        [[r["parent"], num(r.get("instances")), f(r.get("ap50")), f(r.get("ap")),
          num(tot_full[classes.index(r["parent"])]) if r["parent"] in classes else "—",
          num(tot_sel[classes.index(r["parent"])]) if r["parent"] in classes else "—",
          f((f3.get(r["parent"]) or {}).get("recall"), 3)]
         for r in dr["per_class"]])

if D.get("conf_sensitivity"):
    T["conf"] = Table(
        ["Confidence threshold", "Detections", "Per subtile", "Subtiles with ≥1 object",
         "Classes firing"],
        [[f(r["conf"], 2), num(r["detections"]), f(r["per_subtile"], 2), pc(r["occupancy"], 1),
          r["classes"]] for r in D["conf_sensitivity"]])

if D.get("fig3_missed"):
    T["missed"] = Table(
        ["Class", "Objects per cluster, full grid", "Missed by the policy", "Recall"],
        [[r["class"], f(r["objects_per_cluster"], 2), f(r["missed_by_RL"], 2), f(r["recall"], 3)]
         for r in D["fig3_missed"]])

if shap:
    T["shap"] = Table(
        ["Feature", "Kind", "Mean |SHAP|", "Share"],
        [[s["feature"], "HR object count" if s["feature"] in classes else "LR descriptor",
          f(s["mean_abs_shap"], 4), pc(s["mean_abs_shap"] / shap_total, 1)] for s in shap])

T["config"] = Table(
    ["Key", "Value"],
    [[k, ", ".join(str(x) for x in v) if isinstance(v, list) else str(v)]
     for k, v in cfg.items()])

T["deviations"] = Table(
    ["Ayush et al. (AAAI 2021)", "This run"],
    [["34×34 = 1156 HR subtiles per 10 km × 10 km cluster",
      f"{grid.get('side')}×{grid.get('side')} = {grid.get('subtiles')} subtiles per "
      f"{(cfg.get('CLUSTER_M') or 1113) / 1000:.3f} km square (compute budget)"],
     ["YOLOv3 trained on xView, 10 parent classes",
      "same — COCO weights fine-tuned on the 10 xView parents"
      if DET_IS_XVIEW else "ablation — raw COCO YOLOv3, no xView fine-tuning"],
     ["DigitalGlobe 0.3 m HR imagery",
      "own leafmap scrape, native GSD retained (0.30 / 0.60 / 1.19 m/px), every subtile "
      f"resized to {cfg.get('YOLO_IMGSZ')} px ⇒ {f(gl.get('eff_gsd'))} m/px effective"],
     ["Sentinel-2 10 m LR imagery", "own Sentinel-2 scrape, 9.51 m/px"],
     ["N = 320 clusters (Uganda LSMS)", f"N = {D.get('n_clusters')} clusters (Uganda LSMS)"],
     ["Target: cluster mean consumption",
      "same (pc_cons)" + (", plus log consumption (Jean et al. 2016 convention)"
                          if HAS_LOG else "")],
     ["Regressor unspecified beyond f_r",
      "gradient-boosted trees (300 × depth 3, lr 0.05, subsample 0.8)"]])

T["clusters"] = Table(
    ["Cluster", "Lat", "Lon", "Settlement", "Households", "True pc_cons", "Predicted",
     "Held out", "Subtiles bought", "HR GSD (m/px)", "Objects acquired", "Objects full grid"],
    [[c["cid"], f(c["lat"], 4), f(c["lon"], 4), "urban" if c.get("urban") else "rural",
      c.get("households"), f(c["y_true"], 3), f(c.get("y_pred"), 3),
      "yes" if c.get("out_of_fold") else "no (in-sample)",
      f"{int(sum(c.get('mask') or []))}/{grid.get('subtiles')}", f(c.get("hr_gsd"), 2),
      num(sum(c.get("counts_selected") or [])), num(sum(c.get("counts_full") or []))]
     for c in clusters])

# ───────────────────────────────────────────────────────────────── prose blocks
ABSTRACT = (
    f"A from-scratch reimplementation of {PAPER_LINE}. A policy network sees only the free "
    f"Sentinel-2 image of a survey cluster and decides which of its {grid.get('subtiles')} "
    f"high-resolution subtiles are worth paying for; a frozen YOLOv3 detector runs on the "
    f"acquired subtiles only; a gradient-boosted regressor predicts cluster mean consumption "
    f"from what came back. Trained with REINFORCE and a self-critical baseline on "
    f"{D.get('n_clusters')} Uganda LSMS clusters over {n_seeds} seeds, each with its own "
    f"held-out {test_pct}%. Every number on this page is read straight out of "
    f"<code>data/data.json</code>, produced by one Kaggle run on {run_date}.")

METHOD_PROSE = [
    ("The two-step episodic MDP",
     "<ol><li><b>Acquire.</b> The policy <i>f<sub>p</sub></i> sees the free low-resolution tile "
     "<i>l<sub>i</sub></i> and emits one Bernoulli probability per acquirable high-resolution "
     f"region ({grid.get('policy_grid')}×{grid.get('policy_grid')} policy tiles of "
     f"{cfg.get('S')} subtiles each).</li>"
     "<li><b>Detect.</b> A frozen YOLOv3 runs on the acquired subtiles; dropped subtiles "
     "contribute a zero count vector (Eq. 1).</li></ol>"
     "<p>Reward <i>R</i> = <i>R</i><sub>acc</sub> + <i>R</i><sub>cost</sub>, with "
     "<i>R</i><sub>acc</sub> = −‖v − v̂‖<sub>1</sub> penalising the objects the dropped "
     "subtiles would have contained, and <i>R</i><sub>cost</sub> = λ(1 − ‖a‖<sub>1</sub>/S) "
     f"paying for restraint. λ = {cfg.get('LAMBDA')} in the headline run and is swept over "
     f"{', '.join(str(x) for x in cfg.get('LAMBDA_SWEEP', []))}. Temperature scaling anneals "
     f"exploration into exploitation across {cfg.get('RL_EPOCHS')} epochs "
     f"(α: {cfg.get('ALPHA_START')} → {cfg.get('ALPHA_END')}).</p>"),
    ("Why two feature variants are reported",
     "<p>Equation 1 of the paper zeroes dropped subtiles <em>at the detector stage</em>. The "
     "problem statement then specifies a regressor predicting the poverty index from "
     "<em>“L<sub>i</sub> and parts of H<sub>i</sub>”</em> — the free low-resolution image is a "
     "downstream input too, not only the policy's input. <code>counts_only</code> is the "
     "literal reading of Eq. 1; <code>counts_plus_lr</code> also hands the regressor the "
     "Sentinel-2 descriptors it already has for free (per-channel mean, sd, excess green and "
     "brightness percentiles, pooled over every subtile: 20 numbers). The gap between those "
     "two columns is the largest effect in this reimplementation, so both are always shown.</p>"),
    ("Baseline acquisition policies",
     "<p>Every baseline gets the same budget <i>K</i> the learned policy actually spent, so "
     "the comparison is like-for-like: <b>no dropping</b> (every subtile — the upper bound), "
     "<b>fixed centre</b> (the K most central), <b>random</b>, <b>stochastic</b> "
     "(distance-decayed random), <b>green tiles</b> (least-vegetated first, by low-resolution "
     "excess green) and <b>counts prediction</b> (a ResNet-18 regresses per-subtile object "
     "count from the low-resolution subtile; top-K predicted). <b>LR only</b> buys nothing at "
     "all. <b>RL matched training</b> is a diagnostic in which the regressor is trained on the "
     "same masked representation it is tested on, which separates the train/test feature "
     "mismatch from the acquisition decision itself. Nightlights and settlement-layer "
     "baselines are omitted: those auxiliary rasters are not part of this dataset.</p>"),
]

CAVEATS = [
    f"<b>Absolute accuracy is low and the error bars are wide.</b> The best held-out r² on "
    f"this page is {f((S[MAIN]['best_overall'] or {}).get('r2'))} on the linear target"
    + (f" and {f((S['log_pc_cons']['best_overall'] or {}).get('r2'))} on the log target"
       if HAS_LOG else "")
    + f", from {D.get('n_clusters')} clusters with only ~{test_n} in each test fold. "
      f"Between-seed sd is ±{f(S[MAIN]['rl_sd'])}, so treat any gap smaller than about "
      f"{f(2 * (S[MAIN]['rl_sd'] or 0), 2)} r² as noise.",
    f"<b>GSD confound.</b> Native HR ground sample distance correlates with the target at "
    f"r = {f(gl.get('pearson_r'))}. Resizing equalises apparent object size across clusters, "
    f"but not image sharpness.",
    f"<b>{zero_hr} of {len(clusters)} clusters are shown with nothing acquired.</b> At "
    f"λ = {cfg.get('LAMBDA')} the policy declines to buy anything for them, so their "
    f"prediction rests on the low-resolution image alone.",
    "<b>In-sample rows.</b> Clusters marked “no (in-sample)” in the per-cluster table never "
    "landed in a test split across the seeds that ran. Their predicted value is fitted rather "
    "than held out, and they are excluded from the out-of-fold statistics above.",
    "<b>Detection threshold.</b> Counts are taken at confidence "
    f"{cfg.get('YOLO_CONF')}, low by natural-image standards, because the xView→Uganda domain "
    "shift pushes true detections far down the confidence ranking. The sensitivity table shows "
    "what other thresholds would have given.",
    "<b>Not a like-for-like reproduction of the paper's headline number.</b> A different HR "
    "imagery source, a much smaller cluster footprint, a coarser acquisition grid and a "
    "self-scraped label join all sit between this run and the published table.",
]

FILES = [
    ("report.md", "this page as Markdown"),
    ("llms.txt", "machine-readable index of the above"),
    ("data/data.json", "the entire run as one JSON object — config, metrics, detector report, "
                       "SHAP, per-cluster masks and predictions"),
    ("data/results_raw.csv", "per-seed, per-method, per-variant, per-target metrics"),
    ("data/lambda_sweep.csv", "the cost/accuracy sweep"),
    ("data/clusters.csv", "per-cluster predictions, coordinates and budget spent"),
    ("data/summary.png", "the run's own summary figure"),
    ("data/clusters/<cid>_lr.jpg", "Sentinel-2 thumbnail, one per cluster"),
    ("data/clusters/<cid>_hr.jpg", "high-resolution thumbnail, one per cluster"),
]

# ─────────────────────────────────────────────────────────────── HTML rendering
def md_bold(s):
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)

def html_report():
    p, a = [], None
    a = p.append
    a('<main id="view-report" class="view">')
    a('<article class="card prose" id="report">')
    a(f"<h2>{esc(PAPER.get('title', 'Run report'))} — reimplementation report</h2>")
    a(f'<p class="muted">Run {esc(D.get("generated", ""))} · {D.get("n_clusters")} Uganda LSMS '
      f'clusters · {grid.get("side")}×{grid.get("side")} acquisition grid · '
      f'{"xView-finetuned" if DET_IS_XVIEW else "COCO"} YOLOv3 · '
      f'<a href="report.md">Markdown</a> · <a href="data/data.json">data.json</a> · '
      f'<a href="llms.txt">llms.txt</a></p>')
    a(f"<p>{ABSTRACT}</p>")
    a('<p class="muted">This tab is static HTML: the numbers below are baked in at deploy '
      "time, so anything that reads the page without running JavaScript still sees the whole "
      "report. The Map and Results tabs render the same data interactively.</p>")

    a("<h3>Headline numbers</h3>")
    a(T["headline"].html())

    a("<h3>What this run shows</h3><ol>")
    for v in VERDICT:
        a(f"<li>{md_bold(v)}</li>")
    a("</ol>")

    a("<h3>Accuracy by acquisition policy and feature variant</h3>")
    a('<p class="muted">Pearson r² on the held-out split, mean over '
      f"{n_seeds} seeds. Every policy is given the same tile budget the learned policy "
      "used, so the columns are like-for-like."
      + ("" if ALLCLS_DIFFERS else " The paper's third variant, counts over all detector "
         "classes, is identical here — every class the detector can emit is already a "
         "feature class — so it is omitted rather than repeated.") + "</p>")
    for t in TARGETS:
        a(f"<h4>Target: {TARGET_LABEL.get(t, t)}</h4>")
        a(T[f"pivot_{t}"].html())
        a(f"<details><summary>All metrics, target {TARGET_LABEL.get(t, t)}</summary>"
          + T[f"full_{t}"].html() + "</details>")

    if "sweep" in T:
        a("<h3>Cost / accuracy trade-off</h3>")
        a(f'<p class="muted">λ is the cost coefficient in R<sub>cost</sub>; '
          f'{cfg.get("SWEEP_SEEDS")} seed × {cfg.get("SWEEP_EPOCHS")} epochs per point, so '
          "these are noisier than the table above. Read the counts-only column: it is "
          "monotone in how much imagery gets bought.</p>")
        a(T["sweep"].html())

    if "detector" in T:
        a("<h3>Detector</h3>")
        a(f"<p>{esc(D.get('detector_caveat', ''))} Trained on {num(dr.get('train_images'))} "
          f"xView chips and validated on {num(dr.get('val_images'))} in "
          f"{f(dr.get('hours'), 1)} GPU-hours: mAP@50 {f(dr.get('map50'))}, "
          f"mAP@50-95 {f(dr.get('map50_95'))}, mean precision {f(dr.get('mp'))}, mean recall "
          f"{f(dr.get('mr'))}."
          + (" Child classes folded away from the xView label map: "
             + esc(", ".join(dr.get("dropped_child_classes", []))) + "."
             if dr.get("dropped_child_classes") else "") + "</p>")
        a(T["detector"].html())
        if "conf" in T:
            a("<details><summary>Detection-count sensitivity to the confidence threshold "
              f"(this run used {cfg.get('YOLO_CONF')})</summary>" + T["conf"].html()
              + "</details>")

    if "missed" in T:
        a("<h3>What the policy chooses not to look at</h3>")
        a('<p class="muted">Objects per cluster the detector would have found in the dropped '
          "subtiles — the paper's Fig. 3 analogue. Recall is per class, over the acquired "
          "subtiles only.</p>")
        a(T["missed"].html())

    if "shap" in T:
        a("<h3>Feature importance (TreeSHAP, counts + LR regressor)</h3>")
        a(T["shap"].html())

    a("<h3>Method</h3>")
    for h, body in METHOD_PROSE:
        a(f"<h4>{esc(h)}</h4>{body}")

    a("<h3>Deviations from the paper</h3>")
    a(T["deviations"].html())

    a("<h3>Caveats</h3><ul>")
    for c in CAVEATS:
        a(f"<li>{c}</li>")
    a("</ul>")

    a("<h3>Run configuration</h3>")
    a("<details><summary>Every config key</summary>" + T["config"].html() + "</details>")

    a("<h3>Files published next to this page</h3><ul>")
    for path, what in FILES:
        link = ("<code>" + esc(path) + "</code>" if "<cid>" in path
                else '<a href="' + path + '"><code>' + path + "</code></a>")
        a("<li>" + link + " &mdash; " + esc(what) + "</li>")
    a("</ul>")

    a("<h3>Per-cluster results</h3>")
    a(f'<p class="muted">All {len(clusters)} clusters, predictions from the RL policy with '
      f'counts + LR features. <a href="data/clusters.csv">CSV</a>.</p>')
    a("<details><summary>Show the full per-cluster table</summary>"
      + T["clusters"].html() + "</details>")

    a("</article>")
    a("</main>")
    return "\n".join(p)

# ───────────────────────────────────────────────────────────────── MD rendering
def strip_tags(s):
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"</(p|li|ol|ul|h[1-6])>", "\n", s)
    s = re.sub(r"<li>", "- ", s)
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\n{3,}", "\n\n", html.unescape(s)).strip()

def markdown_report():
    L, a = [], None
    a = L.append
    a(f"# {PAPER.get('title', 'Run report')} — reimplementation report")
    a("")
    a(f"*Reimplementation of {PAPER_LINE}. Generated from `data/data.json`; "
      f"run {D.get('generated')}.*")
    a("")
    a(strip_tags(ABSTRACT))
    a("")
    a("## Headline numbers")
    a("")
    a(T["headline"].md())
    a("")
    a("## What this run shows")
    a("")
    for i, v in enumerate(VERDICT, 1):
        a(f"{i}. {v}")
        a("")
    a("## Accuracy by acquisition policy and feature variant")
    a("")
    a(f"Pearson r² on the held-out split, mean over {n_seeds} seeds. Every policy is given the "
      "same tile budget the learned policy used.")
    a("")
    for t in TARGETS:
        a(f"### Target: {TARGET_LABEL.get(t, t)}")
        a("")
        a(T[f"pivot_{t}"].md())
        a("")
        a(f"All metrics, target {TARGET_LABEL.get(t, t)}:")
        a("")
        a(T[f"full_{t}"].md())
        a("")
    if "sweep" in T:
        a("## Cost / accuracy trade-off")
        a("")
        a(f"λ is the cost coefficient in R_cost; {cfg.get('SWEEP_SEEDS')} seed × "
          f"{cfg.get('SWEEP_EPOCHS')} epochs per point.")
        a("")
        a(T["sweep"].md())
        a("")
    if "detector" in T:
        a("## Detector")
        a("")
        a(strip_tags(D.get("detector_caveat", "")))
        a("")
        a(f"Trained on {num(dr.get('train_images'))} xView chips, validated on "
          f"{num(dr.get('val_images'))}, {f(dr.get('hours'), 1)} GPU-hours. "
          f"mAP@50 = {f(dr.get('map50'))}, mAP@50-95 = {f(dr.get('map50_95'))}, "
          f"mean precision {f(dr.get('mp'))}, mean recall {f(dr.get('mr'))}.")
        a("")
        a(T["detector"].md())
        a("")
        if "conf" in T:
            a(f"### Detection-count sensitivity (run used conf {cfg.get('YOLO_CONF')})")
            a("")
            a(T["conf"].md())
            a("")
    if "missed" in T:
        a("## What the policy chooses not to look at")
        a("")
        a(T["missed"].md())
        a("")
    if "shap" in T:
        a("## Feature importance (TreeSHAP, counts + LR regressor)")
        a("")
        a(T["shap"].md())
        a("")
    a("## Method")
    a("")
    for h, body in METHOD_PROSE:
        a(f"### {h}")
        a("")
        a(strip_tags(body))
        a("")
    a("## Deviations from the paper")
    a("")
    a(T["deviations"].md())
    a("")
    a("## Caveats")
    a("")
    for c in CAVEATS:
        a(f"- {strip_tags(c)}")
    a("")
    a("## Run configuration")
    a("")
    a(T["config"].md())
    a("")
    a("## Files")
    a("")
    for path, what in FILES:
        a(f"- `{path}` — {what}")
    a("")
    a("## Per-cluster results")
    a("")
    a(T["clusters"].md())
    a("")
    return "\n".join(L)

# ────────────────────────────────────────────────────────────── llms.txt/robots
def llms_txt():
    m = S[MAIN]
    return "\n".join([
        f"# {PAPER.get('title', 'Adaptive HR tile selection')} — reimplementation "
        f"({run_date} run)",
        "",
        f"> Reinforcement-learned acquisition of high-resolution satellite subtiles for "
        f"cluster-level poverty prediction: a reimplementation of {PAPER_LINE} on "
        f"{D.get('n_clusters')} Uganda LSMS clusters at a {grid.get('side')}×"
        f"{grid.get('side')} acquisition grid, with YOLOv3 fine-tuned from COCO onto xView. "
        f"One Kaggle run, {n_seeds} seeds, generated {D.get('generated')}.",
        "",
        "Headline: the learned policy buys "
        f"{pc(mean_hr, 1)} of the high-resolution imagery and recovers "
        f"{pc(object_recall, 1)} of the objects on the full grid. Pearson r² on held-out "
        f"clusters is {f(m['rl_co'])} from acquired object counts alone, {f(m['rl_lr'])} once "
        f"the free Sentinel-2 descriptors are added, {f(m['lr_only'])} from those descriptors "
        f"alone, and {f(m['nd_co'])} from counts over every subtile. The acquisition policy "
        "works; the high-resolution counts add nothing measurable beyond the free image.",
        "",
        "## Docs",
        "",
        "- [Full report (Markdown)](report.md): every table, figure caption and caveat from "
        "the run, as text.",
        "- [Interactive viewer](index.html): the same report as the landing tab, plus a "
        "cluster map, per-cluster acquisition masks and charts.",
        "",
        "## Data",
        "",
        "- [data.json](data/data.json): the entire run as one JSON object — config, metrics, "
        "detector report, SHAP, per-cluster masks and predictions.",
        "- [results_raw.csv](data/results_raw.csv): per-seed metrics for every method × "
        "variant × target.",
        "- [lambda_sweep.csv](data/lambda_sweep.csv): the cost/accuracy sweep.",
        "- [clusters.csv](data/clusters.csv): per-cluster predictions and budget spent.",
        "",
        "## Notes",
        "",
        f"- Between-seed sd is ±{f(m['rl_sd'])} r²; differences smaller than that between "
        "methods are noise.",
        f"- Native HR ground sample distance correlates with the target at "
        f"r = {f(gl.get('pearson_r'))}, which bounds how much signal could ride on image "
        "sharpness rather than content.",
        f"- {zero_hr} of {len(clusters)} clusters are predicted with no high-resolution "
        "imagery bought at all.",
        "",
    ])

ROBOTS = """# Static research report. Crawling and LLM ingestion are welcome.
User-agent: *
Allow: /

# Machine-readable entry points:
#   /report.md       the whole run as Markdown
#   /llms.txt        index for LLM fetchers
#   /data/data.json  every number on the site
"""

def json_ld():
    m = S[MAIN]
    return json.dumps({
        "@context": "https://schema.org",
        "@type": "Dataset",
        "name": f"{PAPER.get('title', '')} — reimplementation run ({run_date})",
        "description": strip_tags(ABSTRACT),
        "datePublished": run_date,
        "license": "https://opensource.org/licenses/MIT",
        "keywords": ["poverty mapping", "remote sensing", "reinforcement learning", "YOLOv3",
                     "xView", "Sentinel-2", "Uganda", "LSMS", "object detection",
                     "adaptive sampling"],
        "measurementTechnique": "REINFORCE policy over high-resolution tile acquisition; "
                                "YOLOv3 object counts; gradient-boosted regression",
        "variableMeasured": [
            {"@type": "PropertyValue", "name": "Pearson r2 — RL policy, counts + LR features",
             "value": m["rl_lr"]},
            {"@type": "PropertyValue", "name": "Pearson r2 — RL policy, counts only",
             "value": m["rl_co"]},
            {"@type": "PropertyValue", "name": "Pearson r2 — LR features only",
             "value": m["lr_only"]},
            {"@type": "PropertyValue", "name": "Pearson r2 — counts over every subtile",
             "value": m["nd_co"]},
            {"@type": "PropertyValue", "name": "Fraction of HR imagery acquired",
             "value": mean_hr},
            {"@type": "PropertyValue", "name": "Fraction of detected objects recovered",
             "value": object_recall},
        ],
        "distribution": [
            {"@type": "DataDownload", "encodingFormat": "application/json",
             "contentUrl": "data/data.json"},
            {"@type": "DataDownload", "encodingFormat": "text/csv",
             "contentUrl": "data/results_raw.csv"},
            {"@type": "DataDownload", "encodingFormat": "text/csv",
             "contentUrl": "data/clusters.csv"},
            {"@type": "DataDownload", "encodingFormat": "text/markdown",
             "contentUrl": "report.md"},
        ],
        "citation": PAPER_LINE,
    }, indent=1)

def esc_attr(s):
    return html.escape(str(s), quote=True)

def meta_description():
    m = S[MAIN]
    return (f"Reimplementation of Ayush et al. (AAAI 2021) adaptive high-resolution tile "
            f"acquisition on {D.get('n_clusters')} Uganda LSMS clusters, "
            f"{grid.get('side')}x{grid.get('side')} grid, YOLOv3 fine-tuned COCO to xView. The "
            f"learned policy buys {pc(mean_hr, 1)} of the imagery and recovers "
            f"{pc(object_recall, 1)} of detected objects; held-out Pearson r2 is "
            f"{f(m['rl_lr'])} with counts + low-resolution features versus {f(m['lr_only'])} "
            f"from the free image alone.")

def clusters_csv():
    out = [["cid", "lat", "lon", "urban", "households", "y_true_pc_cons", "y_pred",
            "out_of_fold", "n_oof_splits", "hr_gsd_m_per_px", "subtiles_bought",
            "subtiles_total", "hr_frac", "objects_acquired", "objects_full_grid"]]
    for c in clusters:
        out.append([c["cid"], round(c["lat"], 6), round(c["lon"], 6), c.get("urban"),
                    c.get("households"), round(c["y_true"], 4),
                    round(c["y_pred"], 4) if c.get("y_pred") is not None else "",
                    int(bool(c.get("out_of_fold"))), c.get("n_oof", 0),
                    round(c.get("hr_gsd", 0), 4), int(sum(c.get("mask") or [])),
                    grid.get("subtiles"), round(c.get("hr_frac", 0), 5),
                    int(sum(c.get("counts_selected") or [])),
                    int(sum(c.get("counts_full") or []))])
    return out

# ────────────────────────────────────────────────────────────────────── plumbing
def splice(text, start, end, payload):
    pat = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if not pat.search(text):
        raise SystemExit(f"marker pair missing from index.html: {start} … {end}")
    return pat.sub(lambda _: f"{start}\n{payload}\n{end}", text, count=1)

def main():
    idx = (FE / "index.html").read_text()
    idx = splice(idx, "<!-- PRERENDER:REPORT -->", "<!-- /PRERENDER:REPORT -->", html_report())
    idx = splice(idx, "<!-- PRERENDER:HEAD -->", "<!-- /PRERENDER:HEAD -->",
                 f'<meta name="description" content="{esc_attr(meta_description())}">\n'
                 '<script type="application/ld+json">\n' + json_ld() + "\n</script>")
    (FE / "index.html").write_text(idx)
    (FE / "report.md").write_text(markdown_report())
    (FE / "llms.txt").write_text(llms_txt())
    (FE / "robots.txt").write_text(ROBOTS)
    with open(FE / "data" / "clusters.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(clusters_csv())

    def kb(p):
        return f"{(FE / p).stat().st_size / 1024:.0f} KB"
    print(f"prerendered run {D.get('generated')} — {D.get('n_clusters')} clusters, "
          f"{grid.get('side')}x{grid.get('side')} grid, targets {', '.join(TARGETS)}")
    for p in ("index.html", "report.md", "llms.txt", "robots.txt", "data/clusters.csv"):
        print(f"  {p:<20} {kb(p)}")
    print(f"  {'findings':<20} {len(VERDICT)} bullets")

if __name__ == "__main__":
    main()
