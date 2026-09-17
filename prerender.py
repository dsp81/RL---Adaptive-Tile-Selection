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

FIG_LABEL = {
    "rl": "RL policy (ours)", "rl_matched": "RL, matched training",
    "no_dropping": "No dropping (all)", "fixed_center": "Fixed centre",
    "random": "Random", "stochastic": "Stochastic", "green_tiles": "Green tiles",
    "counts_pred": "Counts CNN", "lr_only": "LR only",
}
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
SITE = "https://dsp81.github.io/RL---Adaptive-Tile-Selection/"
LINKS = [
    ("Code and project", "https://github.com/dsp81/RL---Adaptive-Tile-Selection"),
    ("Kaggle notebook — pipeline",
     "https://www.kaggle.com/code/digvijaysinghparihar/povrl-adaptive-hr-tile-selection"),
    ("Kaggle notebook — xView detector",
     "https://www.kaggle.com/code/digvijaysinghparihar/povrl-xview-yolov3"),
    ("Paper (AAAI 2021)", "https://ojs.aaai.org/index.php/AAAI/article/view/16072"),
]
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

# ───────────────────────────────────────────────────── plain-language passages
# The report is read by people who do not work on remote sensing or reinforcement learning.
# These passages carry the same numbers as the tables, with nothing assumed.
def plain_intro():
    m = S[MAIN]
    return f"""<section class="plain" id="plain">
<h4>Start here — what this is, without the jargon</h4>
<p><b>The problem.</b> Finding out how poor a place is normally means sending people door to
door: the Uganda LSMS survey visits a cluster of households and records what they consume.
That is slow and expensive, so there is a long line of research trying to predict it from
satellite images instead. Free imagery — Sentinel-2, about 10&nbsp;m per pixel — is blurry:
you can see fields, water and roughly where a settlement is. Sharp imagery, where individual
buildings and vehicles are visible, is sold by the tile, and covering a whole country with it
costs real money.</p>
<p><b>The idea being tested.</b> Don't buy all of it. Cut each surveyed area into
{grid.get('subtiles')} squares of about {(grid.get('tile_km') or 0) * 1000:.0f}&nbsp;m across,
let a small neural network look <em>only</em> at the free blurry image, and have it decide
which squares are worth paying for. An object detector then counts what it can see —
buildings, vehicles, boats — in the purchased squares only, and a last model turns those
counts into a guess at the area's average consumption. The chooser is trained by trial and
error: rewarded for the objects it manages to capture, charged for every square it buys.</p>
<p><b>How to read the scores.</b> Nearly everything below is reported as
<b>r²</b>: the share of the differences between areas that the model explains. Zero means it
does no better than guessing the same average everywhere; one would mean perfect. The
numbers here sit near {f(m['lr_only'], 2)} — about a third of the variation, which is a real
signal but nowhere near a replacement for a survey. <b>Held out</b> means those areas were
never shown to the model during training, which is the honest way to score it.</p>
<p><b>What came out — two findings that point opposite ways.</b> The choosing works: buying
{pc(mean_hr, 1)} of the sharp imagery captures {pc(object_recall, 1)} of every object the
detector would have found on the full grid, so the network really has learned where the busy
squares are. But the counts don't help the final answer: a model given only the free blurry
image scores r² {f(m['lr_only'])}, and adding everything bought moves it to
{f(m['rl_lr'])} — a change smaller than the run-to-run wobble of ±{f(m['rl_sd'])}. On this
dataset the expensive imagery did not earn its keep.</p>
<p><b>Why that might be.</b> Honestly, three candidates, and this run cannot separate them:
{D.get('n_clusters')} surveyed areas is a small sample; the free image may already encode the
same thing the object counts do, since greenness and built-up texture track wealth directly;
and at roughly one square kilometre per area, counting buildings may simply not add anything
the blur has not already said. What is <em>not</em> the problem is the detector — it works
(see below), and it finds {num(objects_total)} objects across the dataset.</p>
</section>"""

READS = {
    "methods": "Each pair of bars is one way of choosing which squares to buy. The lower bar "
               "in every pair is nearly the same length — once the free image is in the "
               "model, it barely matters which squares you bought, or whether you bought any "
               "at all (the vertical line). The upper bars are what the bought imagery "
               "achieves on its own, and they are short.",
    "sweep": "Left to right, the policy is buying more imagery. The lower line climbs steeply "
             "— object counts get better simply by covering more ground, not by being "
             "cleverer. The upper line is nearly flat, because the free image was already "
             "doing that work.",
    "curves": "Training is working as intended: the score it optimises goes up, while the "
              "share of imagery it buys falls. Nobody set that share by hand — it comes out "
              "of the price the policy is charged per square.",
    "predictions": "On the left, a perfect model would put every point on the diagonal. The "
                   "points sit below it at the right-hand end: the model is cautious, and it "
                   "badly underestimates the best-off areas. On the right, each dot is one "
                   "surveyed area in its real position, shaded by what the survey measured.",
    "recall": "If squares were picked at random, each bar would sit near the share of imagery "
              "bought. They sit far higher, which is the clearest evidence in this run that "
              "the policy is finding where things actually are.",
    "shap": "How much each input moves the final prediction. Almost all of the movement comes "
            "from the free image's colour and texture; the bought object counts barely "
            "register — the same conclusion as the tables, reached a different way.",
    "examples": "Two views of the same place: the free blurry image, and the sharp one with "
                "the policy's choice drawn on it. Outlined squares were bought and examined; "
                "dimmed squares were never purchased.",
}

def reads(key):
    return f'<p class="reads"><b>In plain terms:</b> {READS[key]}</p>'

GLOSSARY = [
    ("Cluster", "A group of surveyed households treated as one place. The survey reports one "
                "consumption figure per cluster; this run has %d of them." % (D.get("n_clusters") or 0)),
    ("LSMS", "The World Bank's Living Standards Measurement Study — the household survey the "
             "target values come from."),
    ("pc_cons", "Per-capita consumption: roughly what an average person in that cluster "
                "consumes, and the number every model here is trying to predict."),
    ("Subtile", "One square of the grid each cluster is cut into — %d of them per cluster, "
                "about %.0f m across." % (grid.get("subtiles") or 0,
                                          (grid.get("tile_km") or 0) * 1000)),
    ("LR / HR", "Low resolution (free Sentinel-2, ~9.5 m per pixel) and high resolution (the "
                "imagery being bought, 0.30–1.19 m per pixel)."),
    ("GSD", "Ground sample distance — how much ground one pixel covers. Smaller is sharper."),
    ("r²", "Share of the variation between clusters that a model explains: 0 is no better "
           "than always guessing the average, 1 would be perfect."),
    ("Spearman", "The same idea as r² but about rank order only — whether the model puts "
                 "clusters in the right order, ignoring how far apart it thinks they are."),
    ("Held out", "Scored on clusters the model never saw while training. Scores measured any "
                 "other way are not trustworthy."),
    ("Seed", "One complete training run with a different random start and a different "
             "train/test split. %d were run; the spread between them is the error bar."
             % len(cfg.get("SEEDS", []) or [])),
    ("λ (lambda)", "The price charged to the policy for each square it buys. Higher λ, less "
                   "imagery bought."),
    ("REINFORCE", "The reinforcement-learning method used: try something, see the reward, "
                  "nudge the network towards whatever scored better."),
    ("mAP@50", "The standard score for an object detector: how well its boxes match real "
               "objects. %s here, on the xView benchmark." % f(dr.get("map50"))),
    ("xView", "A large public dataset of labelled overhead imagery, used to teach the "
              "detector what buildings and vehicles look like from above."),
    ("SHAP", "A method for asking how much each input contributed to a prediction."),
]

def glossary_html():
    return ('<dl class="glossary">'
            + "".join(f"<dt>{esc(t)}</dt><dd>{esc(d)}</dd>" for t, d in GLOSSARY)
            + "</dl>")

# ─────────────────────────────────────────────────────────────── static figures
# Inline SVG, no script: the page has to render identically for a browser, a crawler and a
# `curl | sed`. Marks follow one spec throughout -- <=24px bars with a 4px rounded data-end
# and a 2px surface gap, 2px lines, >=8px markers with a 2px surface ring, hairline solid
# gridlines. Colours are CSS custom properties, so light/dark theming is inherited from
# style.css instead of being baked in (the pair #2a78d6/#eb6834 and its dark-mode steps pass
# the six-check palette validation in both modes). Each mark carries a <title>, which browsers
# render as a native tooltip with no JavaScript at all.

S1, S2 = "var(--series-1)", "var(--series-2)"
SURFACE = "var(--surface-1)"
INK_MUTED = "var(--text-muted)"

def xml(s):
    return html.escape(str(s), quote=True)

def nice_max(v):
    if v <= 0:
        return 1.0
    p = 10 ** math.floor(math.log10(v))
    n = v / p
    for step in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if n <= step:
            return step * p
    return 10 * p

def nice_axis(vmax, ticks=4, vmin=0.0):
    """Round the range out so every tick lands on a round number."""
    step = nice_max((vmax - vmin) / ticks)
    lo = math.floor(vmin / step) * step if vmin < 0 else 0.0
    hi = lo + step * ticks
    while hi < vmax - 1e-12:
        hi += step
    return lo, hi

def tip(text):
    return f"<title>{xml(text)}</title>"

def bar_path(x, y, w, h, r=4, horizontal=False):
    """Bar with a rounded data-end and a square baseline end."""
    r = max(0, min(r, (w if horizontal else h) / 2, (h if horizontal else w) / 2))
    if horizontal:                                  # grows left → right
        return (f"M{x},{y} H{x + w - r} A{r},{r} 0 0 1 {x + w},{y + r} "
                f"V{y + h - r} A{r},{r} 0 0 1 {x + w - r},{y + h} H{x} Z")
    return (f"M{x},{y + h} V{y + r} A{r},{r} 0 0 1 {x + r},{y} "
            f"H{x + w - r} A{r},{r} 0 0 1 {x + w},{y + r} V{y + h} Z")

def svg_open(w, h, label):
    return (f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-label="{xml(label)}" style="width:100%;height:auto;max-width:{w}px">')

def figure(body, caption, legend=None):
    lg = ""
    if legend:
        lg = ('<div class="legend-row">' + "".join(
            f'<span><i class="swatch" style="background:{c}"></i>{esc(n)}</span>'
            for n, c in legend) + "</div>")
    return (f'<figure class="fig">{lg}{body}'
            f'<figcaption class="cap">{caption}</figcaption></figure>')

def y_grid(x0, x1, y_top, y_bot, vmax, ticks=4, fmt=lambda v: f"{v:.2f}", vmin=0.0):
    out = []
    for i in range(ticks + 1):
        v = vmin + (vmax - vmin) * i / ticks
        y = y_bot - (y_bot - y_top) * i / ticks
        out.append(f'<line class="ax" x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="axtext" x="{x0 - 7}" y="{y + 3.5:.1f}" '
                   f'text-anchor="end">{fmt(v)}</text>')
    return "".join(out)

def x_grid(y0, y1, x_left, x_right, vmax, ticks=4, fmt=lambda v: f"{v:.2f}", vmin=0.0):
    out = []
    for i in range(ticks + 1):
        v = vmin + (vmax - vmin) * i / ticks
        x = x_left + (x_right - x_left) * i / ticks
        out.append(f'<line class="ax" y1="{y0}" y2="{y1}" x1="{x:.1f}" x2="{x:.1f}"/>')
        out.append(f'<text class="axtext" x="{x:.1f}" y="{y1 + 15}" '
                   f'text-anchor="middle">{fmt(v)}</text>')
    return "".join(out)

def fig_methods(target):
    """Horizontal grouped bars: r² per acquisition policy, two feature variants."""
    piv = PIV.get(target, {})
    order = ["rl", "rl_matched", "no_dropping", "counts_pred", "green_tiles", "fixed_center",
             "stochastic", "random"]
    rows = [(m, piv[m]) for m in order if m in piv]
    lr_only_v = cell(target, "lr_only", "lr_only")
    W, PADL, PADR, PADT = 760, 150, 58, 26
    rowh, barh, gap = 40, 15, 2
    H = PADT + rowh * len(rows) + 52
    x0, x1 = PADL, W - PADR
    _, vmax = nice_axis(1.06 * max([r.get("counts_plus_lr") or 0 for _, r in rows]
                                   + [lr_only_v or 0, 0.05]))
    o = [svg_open(W, H, f"Pearson r squared by acquisition policy, target {target}")]
    o.append(x_grid(PADT, PADT + rowh * len(rows), x0, x1, vmax))
    for i, (m, r) in enumerate(rows):
        ytop = PADT + i * rowh
        o.append(f'<text class="lbl" x="{x0 - 10}" y="{ytop + rowh / 2 + 4:.0f}" '
                 f'text-anchor="end">{esc(FIG_LABEL.get(m, m))}</text>')
        for k, (key, colour, name) in enumerate(
                [("counts_only", S1, "counts only"), ("counts_plus_lr", S2, "counts + LR")]):
            v = r.get(key)
            if v is None:
                continue
            y = ytop + (rowh - 2 * barh - gap) / 2 + k * (barh + gap)
            w = max(1.5, (x1 - x0) * v / vmax)
            t = f"{METHOD_LABEL.get(m, m)} — {name}: r² {v:.3f}"
            o.append(f'<path d="{bar_path(x0, y, w, barh, 4, True)}" fill="{colour}">'
                     f'{tip(t)}</path>')
            if m == "rl" or (m == "no_dropping" and key == "counts_only"):
                o.append(f'<text class="vlbl" x="{x0 + w + 6:.1f}" y="{y + barh - 3:.0f}">'
                         f'{v:.3f}</text>')
    if lr_only_v is not None:
        lx = x0 + (x1 - x0) * lr_only_v / vmax
        ybot = PADT + rowh * len(rows)
        o.append(f'<line x1="{lx:.1f}" x2="{lx:.1f}" y1="{PADT - 10}" y2="{ybot}" '
                 f'stroke="{INK_MUTED}" stroke-width="1.5" opacity="0.85">'
                 f'<title>LR only, nothing bought: r² {lr_only_v:.3f}</title></line>')
        near_edge = lx > x0 + 0.66 * (x1 - x0)
        o.append(f'<text class="axtext" x="{lx + (-4 if near_edge else 4):.1f}" '
                 f'y="{PADT - 14}" text-anchor="{"end" if near_edge else "start"}">'
                 f'LR only, nothing bought — {lr_only_v:.3f}</text>')
    o.append(f'<text class="lbl" x="{(x0 + x1) / 2:.0f}" y="{H - 6}" text-anchor="middle">'
             f'Pearson r² on the held-out split</text>')
    o.append("</svg>")
    return figure(
        "".join(o),
        f"Target: {TARGET_LABEL.get(target, target)}. Every policy is given the same tile "
        f"budget the learned policy spent. The vertical rule is the regressor that buys "
        f"nothing and sees only the free Sentinel-2 descriptors; no policy clears it.",
        legend=[("counts only (Eq. 1 literal)", S1), ("counts + LR features", S2)])

def fig_sweep():
    if not sweep:
        return ""
    W, H, PADL, PADR, PADT, PADB = 760, 320, 58, 24, 18, 52
    x0, x1, yt, yb = PADL, W - PADR, PADT, H - PADB
    xs = [r["hr_frac"] for r in sweep]
    _, xmax = nice_axis(max(xs))
    _, ymax = nice_axis(1.12 * max(max(r.get("r2_counts_only") or 0,
                                       r.get("r2_counts_plus_lr") or 0) for r in sweep))
    X = lambda v: x0 + (x1 - x0) * v / xmax
    Y = lambda v: yb - (yb - yt) * v / ymax
    o = [svg_open(W, H, "r squared against the fraction of high-resolution imagery bought")]
    o.append(y_grid(x0, x1, yt, yb, ymax))
    o.append(x_grid(yt, yb, x0, x1, xmax, fmt=lambda v: f"{v * 100:.0f}%"))
    pts = sorted(sweep, key=lambda r: r["hr_frac"])
    for key, colour, name, dy in [("r2_counts_only", S1, "counts only", 16),
                                  ("r2_counts_plus_lr", S2, "counts + LR", -11)]:
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(r['hr_frac']):.1f},{Y(r[key]):.1f}"
                     for i, r in enumerate(pts))
        o.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="2" '
                 f'stroke-linejoin="round" stroke-linecap="round"/>')
        for r in pts:
            t = (f"lambda {r['lam']}: buys {r['hr_frac'] * 100:.1f}% of the imagery — "
                 f"{name} r² {r[key]:.3f}")
            o.append(f'<circle cx="{X(r["hr_frac"]):.1f}" cy="{Y(r[key]):.1f}" r="4.5" '
                     f'fill="{colour}" stroke="{SURFACE}" stroke-width="2">'
                     f'{tip(t)}</circle>')
        last = pts[-1]
        o.append(f'<text class="vlbl" x="{X(last["hr_frac"]) - 7:.1f}" '
                 f'y="{Y(last[key]) + dy:.1f}" text-anchor="end">'
                 f'{name} {last[key]:.3f}</text>')
    for r in (pts[0], pts[-1]):
        o.append(f'<text class="axtext" x="{X(r["hr_frac"]):.1f}" y="{yb + 30}" '
                 f'text-anchor="middle">λ {r["lam"]}</text>')
    o.append(f'<text class="lbl" x="{(x0 + x1) / 2:.0f}" y="{H - 6}" text-anchor="middle">'
             f'fraction of high-resolution subtiles bought</text>')
    o.append(f'<text class="lbl" x="14" y="{(yt + yb) / 2:.0f}" text-anchor="middle" '
             f'transform="rotate(-90 14 {(yt + yb) / 2:.0f})">Pearson r²</text>')
    o.append("</svg>")
    return figure(
        "".join(o),
        f"Cost against accuracy, {cfg.get('SWEEP_SEEDS')} seed × {cfg.get('SWEEP_EPOCHS')} "
        "epochs per point. The counts-only curve climbs with coverage; the counts + LR curve "
        "barely moves, because the free image is already carrying it.",
        legend=[("counts only", S1), ("counts + LR", S2)])

def fig_curves():
    curves = D.get("rl_curves") or {}
    if not curves:
        return ""
    seeds = sorted(curves, key=lambda k: int(k))
    panels = [("reward", "reward per episode", S1), ("hr_frac", "fraction bought", S2)]
    W, H, PADL, PADR, PADT, PADB = 368, 258, 46, 16, 16, 60
    out = []
    for key, title, colour in panels:
        x0, x1, yt, yb = PADL, W - PADR, PADT, H - PADB
        epochs = [p["epoch"] for p in curves[seeds[0]]]
        vals = [p[key] for s in seeds for p in curves[s]]
        vmin, vmax = nice_axis(max(vals), vmin=min(vals + [0]))
        X = lambda v: x0 + (x1 - x0) * (v / max(epochs) if max(epochs) else 0.5)
        Y = lambda v: yb - (yb - yt) * (v - vmin) / (vmax - vmin or 1)
        o = [svg_open(W, H, f"policy {title} per epoch, {len(seeds)} seeds")]
        o.append(y_grid(x0, x1, yt, yb, vmax, fmt=lambda v: f"{v:.2f}", vmin=vmin))
        o.append(x_grid(yt, yb, x0, x1, max(epochs), fmt=lambda v: f"{v:.0f}"))
        if vmin < 0:
            o.append(f'<line class="ax" x1="{x0}" x2="{x1}" y1="{Y(0):.1f}" y2="{Y(0):.1f}"/>')
        for s in seeds:                              # every seed, de-emphasised
            d = " ".join(f"{'M' if i == 0 else 'L'}{X(p['epoch']):.1f},{Y(p[key]):.1f}"
                         for i, p in enumerate(curves[s]))
            o.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="1.5" '
                     f'stroke-opacity="0.32" stroke-linejoin="round"><title>seed {s}</title>'
                     f'</path>')
        mean = [sum(curves[s][i][key] for s in seeds) / len(seeds)
                for i in range(len(curves[seeds[0]]))]
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(epochs[i]):.1f},{Y(v):.1f}"
                     for i, v in enumerate(mean))
        o.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="2" '
                 f'stroke-linejoin="round"><title>mean over {len(seeds)} seeds</title></path>')
        o.append(f'<text class="vlbl" x="{X(epochs[-1]):.1f}" y="{Y(mean[-1]) - 9:.1f}" '
                 f'text-anchor="end">{mean[-1]:.2f}</text>')
        o.append(f'<text class="lbl" x="{(x0 + x1) / 2:.0f}" y="{H - 22}" '
                 f'text-anchor="middle">epoch</text>')
        o.append(f'<text class="lbl" x="{(x0 + x1) / 2:.0f}" y="{H - 5}" '
                 f'text-anchor="middle">{title}</text>')
        o.append("</svg>")
        out.append("".join(o))
    return figure(
        f'<div class="fig-pair">{out[0]}{out[1]}</div>',
        f"One thin line per seed ({len(seeds)} of them), the solid line their mean. Two "
        "measures of different scale, so two panels rather than two y-axes on one. Reward "
        "climbs as temperature scaling anneals exploration into exploitation, and the "
        "fraction bought falls out of the cost term rather than being set by hand.")

def fig_scatter():
    pts = [c for c in clusters if c.get("out_of_fold")]
    if not pts:
        return ""
    W, H, PADL, PADR, PADT, PADB = 380, 380, 52, 18, 16, 48
    x0, x1, yt, yb = PADL, W - PADR, PADT, H - PADB
    _, m = nice_axis(max(max(c["y_true"], c["y_pred"]) for c in pts))
    X = lambda v: x0 + (x1 - x0) * v / m
    Y = lambda v: yb - (yb - yt) * v / m
    o = [svg_open(W, H, "predicted against true consumption, held-out clusters")]
    o.append(y_grid(x0, x1, yt, yb, m, fmt=lambda v: f"{v:.0f}"))
    o.append(x_grid(yt, yb, x0, x1, m, fmt=lambda v: f"{v:.0f}"))
    o.append(f'<line x1="{X(0):.1f}" y1="{Y(0):.1f}" x2="{X(m):.1f}" y2="{Y(m):.1f}" '
             f'stroke="{INK_MUTED}" stroke-width="1.25" stroke-opacity="0.6"/>')
    for c in pts:
        t = (f"cluster {c['cid']} — true {c['y_true']:.2f}, "
             f"predicted {c['y_pred']:.2f}")
        o.append(f'<circle cx="{X(c["y_true"]):.1f}" cy="{Y(c["y_pred"]):.1f}" r="3.6" '
                 f'fill="{S1}" fill-opacity="0.55" stroke="{SURFACE}" stroke-width="1.5">'
                 f'{tip(t)}</circle>')
    o.append(f'<text class="lbl" x="{(x0 + x1) / 2:.0f}" y="{H - 8}" text-anchor="middle">'
             f'true pc_cons</text>')
    o.append(f'<text class="lbl" x="14" y="{(yt + yb) / 2:.0f}" text-anchor="middle" '
             f'transform="rotate(-90 14 {(yt + yb) / 2:.0f})">predicted</text>')
    o.append("</svg>")
    return "".join(o)

def fig_map():
    W, H, PAD = 380, 380, 26
    lats = [c["lat"] for c in clusters]
    lons = [c["lon"] for c in clusters]
    la0, la1, lo0, lo1 = min(lats), max(lats), min(lons), max(lons)
    span = max(la1 - la0, lo1 - lo0) * 1.04
    cx, cy = (lo0 + lo1) / 2, (la0 + la1) / 2
    X = lambda lon: PAD + (W - 2 * PAD) * (lon - (cx - span / 2)) / span
    Y = lambda lat: H - PAD - (H - 2 * PAD) * (lat - (cy - span / 2)) / span
    vals = sorted(c["y_true"] for c in clusters)
    q = lambda p: vals[max(0, min(len(vals) - 1, int(p * (len(vals) - 1))))]
    lo_v, hi_v = q(0.05), q(0.95)
    step = lambda v: max(0, min(6, int(round(6 * (v - lo_v) / ((hi_v - lo_v) or 1)))))
    o = [svg_open(W, H, "survey cluster locations, shaded by true consumption")]
    for c in clusters:
        t = (f"cluster {c['cid']} ({'urban' if c['urban'] else 'rural'}) — "
             f"pc_cons {c['y_true']:.2f}, bought {int(sum(c['mask']))}/"
             f"{grid.get('subtiles')} subtiles")
        o.append(f'<circle cx="{X(c["lon"]):.1f}" cy="{Y(c["lat"]):.1f}" r="3.8" '
                 f'fill="var(--seq-{step(c["y_true"]) + 1})" stroke="{SURFACE}" '
                 f'stroke-width="1.2">{tip(t)}</circle>')
    # sequential scale legend: one hue, light → dark, with its ends labelled beside it
    rx0 = PAD + 24
    o.append(f'<rect x="0" y="{H - 44}" width="{W}" height="44" fill="{SURFACE}" '
             f'fill-opacity="0.92"/>')
    o.append(f'<text class="axtext" x="{PAD}" y="{H - 30}">consumption (pc_cons)</text>')
    for i in range(7):
        o.append(f'<rect x="{rx0 + i * 15}" y="{H - 24}" width="13" height="8" rx="2" '
                 f'fill="var(--seq-{i + 1})"/>')
    o.append(f'<text class="axtext" x="{rx0 - 5}" y="{H - 17}" text-anchor="end">'
             f'{lo_v:.1f}</text>')
    o.append(f'<text class="axtext" x="{rx0 + 7 * 15 + 1}" y="{H - 17}">{hi_v:.1f}</text>')
    o.append("</svg>")
    return "".join(o)

def fig_recall():
    rows = [r for r in (D.get("fig3_missed") or []) if r["objects_per_cluster"] > 0]
    if not rows:
        return ""
    rows = sorted(rows, key=lambda r: -r["objects_per_cluster"])
    W, PADL, PADR, PADT = 760, 176, 64, 14
    rowh, barh = 26, 14
    H = PADT + rowh * len(rows) + 50
    x0, x1 = PADL, W - PADR
    o = [svg_open(W, H, "per-class recall of detected objects inside the acquired subtiles")]
    o.append(x_grid(PADT, PADT + rowh * len(rows), x0, x1, 1.0,
                    fmt=lambda v: f"{v * 100:.0f}%"))
    for i, r in enumerate(rows):
        y = PADT + i * rowh + (rowh - barh) / 2
        w = max(1.5, (x1 - x0) * r["recall"])
        o.append(f'<text class="lbl" x="{x0 - 10}" y="{y + barh - 2:.0f}" '
                 f'text-anchor="end">{esc(r["class"])}</text>')
        t = (f"{r['class']}: {r['recall'] * 100:.1f}% of {r['objects_per_cluster']:.2f} "
             f"objects per cluster kept; {r['missed_by_RL']:.2f} missed")
        o.append(f'<path d="{bar_path(x0, y, w, barh, 4, True)}" fill="{S1}">'
                 f'{tip(t)}</path>')
        o.append(f'<text class="vlbl" x="{x0 + w + 6:.1f}" y="{y + barh - 2:.0f}">'
                 f'{r["recall"] * 100:.0f}%</text>')
    o.append(f'<text class="lbl" x="{(x0 + x1) / 2:.0f}" y="{H - 6}" text-anchor="middle">'
             f'share of the class\u2019s objects that fall inside the acquired subtiles</text>')
    o.append("</svg>")
    return figure(
        "".join(o),
        f"Classes ordered by how common they are. Buying {pc(mean_hr, 1)} of the imagery "
        f"keeps {pc(object_recall, 1)} of all detected objects, which is the acquisition "
        "claim of the paper reproducing; uniform sampling at this budget would keep about "
        f"{pc(mean_hr, 0)} of each class.")

def fig_shap(n=14):
    rows = (shap or [])[:n]
    if not rows:
        return ""
    W, PADL, PADR, PADT = 760, 196, 70, 14
    rowh, barh = 26, 14
    H = PADT + rowh * len(rows) + 50
    x0, x1 = PADL, W - PADR
    _, vmax = nice_axis(1.04 * max(r["mean_abs_shap"] for r in rows))
    o = [svg_open(W, H, "mean absolute SHAP per feature")]
    o.append(x_grid(PADT, PADT + rowh * len(rows), x0, x1, vmax))
    for i, r in enumerate(rows):
        is_hr = r["feature"] in classes
        y = PADT + i * rowh + (rowh - barh) / 2
        w = max(1.5, (x1 - x0) * r["mean_abs_shap"] / vmax)
        o.append(f'<text class="lbl" x="{x0 - 10}" y="{y + barh - 2:.0f}" '
                 f'text-anchor="end">{esc(r["feature"])}</text>')
        kind = "HR object count" if is_hr else "LR descriptor"
        t = f"{r['feature']} ({kind}): mean |SHAP| {r['mean_abs_shap']:.4f}"
        o.append(f'<path d="{bar_path(x0, y, w, barh, 4, True)}" fill="{S2 if is_hr else S1}">'
                 f'{tip(t)}</path>')
        o.append(f'<text class="vlbl" x="{x0 + w + 6:.1f}" y="{y + barh - 2:.0f}">'
                 f'{r["mean_abs_shap"]:.3f}</text>')
    o.append(f'<text class="lbl" x="{(x0 + x1) / 2:.0f}" y="{H - 6}" text-anchor="middle">'
             f'mean |SHAP| on the held-out predictions</text>')
    o.append("</svg>")
    return figure(
        "".join(o),
        f"Top {len(rows)} of {len(shap)} features. The bought object counts hold "
        f"{pc(shap_hr_share, 1)} of total importance; everything above them is a free "
        "Sentinel-2 descriptor.",
        legend=[("LR descriptor (free)", S1), ("HR object count (bought)", S2)])

def fig_examples(n=3):
    """A few clusters, with the acquisition mask drawn over the two images."""
    # Sorting by "bought the most" surfaces the clusters where the policy simply bought
    # everything, which shows nothing. What is worth looking at is a partial buy that still
    # caught most of the objects, one of each settlement type, plus one it declined outright.
    def caught(c):
        full = sum(c.get("counts_full") or []) or 1
        return sum(c.get("counts_selected") or []) / full

    cand = [c for c in clusters
            if 4 <= sum(c.get("mask") or []) <= 0.45 * (grid.get("subtiles") or 256)
            and sum(c.get("counts_full") or []) >= 150]
    cand.sort(key=lambda c: -caught(c))
    picks, seen = [], set()
    for c in cand:                                   # one urban, one rural, then the best left
        if c["urban"] not in seen:
            picks.append(c)
            seen.add(c["urban"])
    picks += [c for c in cand if c not in picks][:max(0, n - len(picks))]
    picks = picks[:n]
    zero = [c for c in clusters if not sum(c.get("mask") or [])]
    zero = sorted(zero, key=lambda c: -sum(c.get("counts_full") or []))[:1]
    G = grid.get("side") or 16
    out = []
    for c in picks + zero:
        nsel = int(sum(c["mask"]))
        cells = []
        for i, on in enumerate(c["mask"]):
            r, col = divmod(i, G)
            if on:
                cells.append(f'<rect x="{col + 0.04}" y="{r + 0.04}" width="0.92" '
                             f'height="0.92" rx="0.08" fill="none" stroke="{S2}" '
                             f'stroke-width="0.09"/>')
            else:
                cells.append(f'<rect x="{col}" y="{r}" width="1" height="1" '
                             f'fill="var(--surface-0)" fill-opacity="0.62"/>')
        overlay = (f'<svg class="overlay" viewBox="0 0 {G} {G}" preserveAspectRatio="none">'
                   + "".join(cells) + "</svg>")
        out.append(
            f'<div class="ex">'
            f'<div class="ex-head"><b>Cluster {c["cid"]}</b> '
            f'<span class="muted">{"urban" if c["urban"] else "rural"} · pc_cons '
            f'{c["y_true"]:.2f} · predicted {c["y_pred"]:.2f} · bought {nsel}/{G * G} '
            f'subtiles · {num(sum(c["counts_selected"]))} of '
            f'{num(sum(c["counts_full"]))} objects</span></div>'
            f'<div class="imgrow">'
            f'<figure><figcaption>Sentinel-2 <span class="tag">free</span></figcaption>'
            f'<div class="imgwrap"><img src="data/{c["lr_img"]}" '
            f'alt="Sentinel-2 image of cluster {c["cid"]}"></div></figure>'
            f'<figure><figcaption>High resolution <span class="tag tag-cost">costly</span>'
            f'</figcaption><div class="imgwrap"><img src="data/{c["hr_img"]}" '
            f'alt="High-resolution image of cluster {c["cid"]}, with the acquired subtiles '
            f'outlined">{overlay}</div></figure>'
            f'</div></div>')
    return ('<div class="examples">' + "".join(out) + "</div>"
            + f'<p class="cap">Outlined cells were bought and sent to the detector; dimmed '
              f'cells were never purchased. The last cluster is one the policy declined '
              f'entirely — {zero_hr} of {len(clusters)} end up that way at λ = '
              f'{cfg.get("LAMBDA")}, and are predicted from the free image alone.</p>')

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
    a('<p class="muted">'
      + " · ".join(f'<a href="{u}">{esc(n)}</a>' for n, u in LINKS) + "</p>")
    a(f"<p>{ABSTRACT}</p>")
    a(plain_intro())
    a('<p class="muted">This page runs no JavaScript. Every number, table and figure below '
      "is written into the HTML when the site is built, so it reads the same in a browser, in "
      "a crawler, and in <code>curl</code>. Hovering a bar, line or point shows its exact "
      "value.</p>")

    a("<h3>Headline numbers</h3>")
    a(T["headline"].html())

    a('<h3 id="findings">What this run shows</h3><ol>')
    for v in VERDICT:
        a(f"<li>{md_bold(v)}</li>")
    a("</ol>")

    a('<h3 id="accuracy">Accuracy by acquisition policy and feature variant</h3>')
    a('<p class="muted">Pearson r² on the held-out split, mean over '
      f"{n_seeds} seeds. Every policy is given the same tile budget the learned policy "
      "used, so the columns are like-for-like."
      + ("" if ALLCLS_DIFFERS else " The paper's third variant, counts over all detector "
         "classes, is identical here — every class the detector can emit is already a "
         "feature class — so it is omitted rather than repeated.") + "</p>")
    a(reads("methods"))
    for t in TARGETS:
        a(f"<h4>Target: {TARGET_LABEL.get(t, t)}</h4>")
        a(fig_methods(t))
        a(T[f"pivot_{t}"].html())
        a(f"<details><summary>All metrics, target {TARGET_LABEL.get(t, t)}</summary>"
          + T[f"full_{t}"].html() + "</details>")

    if "sweep" in T:
        a('<h3 id="cost">Cost / accuracy trade-off</h3>')
        a(f'<p class="muted">λ is the cost coefficient in R<sub>cost</sub>; '
          f'{cfg.get("SWEEP_SEEDS")} seed × {cfg.get("SWEEP_EPOCHS")} epochs per point, so '
          "these are noisier than the table above. Read the counts-only column: it is "
          "monotone in how much imagery gets bought.</p>")
        a(reads("sweep"))
        a(fig_sweep())
        a(T["sweep"].html())

    a('<h3 id="training">How the policy trains</h3>')
    a(reads("curves"))
    a(fig_curves())

    a('<h3 id="predictions">Predictions and where they land</h3>')
    a(reads("predictions"))
    a('<div class="fig-pair">' + fig_scatter() + fig_map() + "</div>")
    a(f'<p class="cap">Left: held-out predictions against the survey value for the '
      f'{len(oof)} clusters that were in a test split, pooled over seeds; the diagonal is '
      f'parity, and the pooled correlation is r = {f(oof_r)} (RMSE {f(oof_rmse, 2)}). '
      f'Right: the {len(clusters)} clusters in place, shaded by measured consumption.</p>')

    a('<h3 id="examples">What the policy actually buys</h3>')
    a(reads("examples"))
    a(fig_examples())

    if "detector" in T:
        a('<h3 id="detector">Detector</h3>')
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
        a('<h3 id="missed">What the policy chooses not to look at</h3>')
        a('<p class="muted">Objects per cluster the detector would have found in the dropped '
          "subtiles — the paper's Fig. 3 analogue. Recall is per class, over the acquired "
          "subtiles only.</p>")
        a(reads("recall"))
        a(fig_recall())
        a(T["missed"].html())

    if "shap" in T:
        a('<h3 id="shap">Feature importance (TreeSHAP, counts + LR regressor)</h3>')
        a(reads("shap"))
        a(fig_shap())
        a("<details><summary>All " + str(len(shap)) + " features</summary>"
          + T["shap"].html() + "</details>")

    a('<h3 id="method">Method</h3>')
    for h, body in METHOD_PROSE:
        a(f"<h4>{esc(h)}</h4>{body}")

    a("<h3>Deviations from the paper</h3>")
    a(T["deviations"].html())

    a('<h3 id="caveats">Caveats</h3><ul>')
    for c in CAVEATS:
        a(f"<li>{c}</li>")
    a("</ul>")

    a("<h3>Run configuration</h3>")
    a("<details><summary>Every config key</summary>" + T["config"].html() + "</details>")

    a('<h3 id="glossary">Glossary</h3>')
    a(glossary_html())

    a("<h3>Files published next to this page</h3><ul>")
    for path, what in FILES:
        link = ("<code>" + esc(path) + "</code>" if "<cid>" in path
                else '<a href="' + path + '"><code>' + path + "</code></a>")
        a("<li>" + link + " &mdash; " + esc(what) + "</li>")
    a("</ul>")

    a('<h3 id="clusters">Per-cluster results</h3>')
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

NAV_ITEMS = [("Start here", "#plain"), ("Findings", "#findings"), ("Accuracy", "#accuracy"),
             ("Cost", "#cost"), ("Training", "#training"), ("Predictions", "#predictions"),
             ("Examples", "#examples"), ("Detector", "#detector"), ("Method", "#method"),
             ("Caveats", "#caveats"), ("Glossary", "#glossary"), ("Clusters", "#clusters")]

def nav_html():
    return ('<nav class="tabs" aria-label="Sections">'
            + "".join(f'<a class="tab" href="{h}">{esc(n)}</a>' for n, h in NAV_ITEMS)
            + "</nav>")

def foot_html():
    return (f'<span>{D.get("n_clusters")} clusters · {grid.get("subtiles")} HR subtiles each '
            f'· run {esc(D.get("generated", ""))}</span> <span>'
            + " · ".join(f'<a href="{u}">{esc(n)}</a>' for n, u in LINKS)
            + ' · <a href="report.md">report.md</a> · <a href="llms.txt">llms.txt</a>'
              ' · <a href="data/data.json">data.json</a>'
              ' · <a href="data/clusters.csv">clusters.csv</a></span>')

def markdown_report():
    L, a = [], None
    a = L.append
    a(f"# {PAPER.get('title', 'Run report')} — reimplementation report")
    a("")
    a(f"*Reimplementation of {PAPER_LINE}. Generated from `data/data.json`; "
      f"run {D.get('generated')}.*")
    a("")
    a("- Site: " + SITE)
    a("\n".join(f"- {n}: {u}" for n, u in LINKS))
    a("")
    a(strip_tags(ABSTRACT))
    a("")
    a("## Start here — what this is, without the jargon")
    a("")
    a(strip_tags(plain_intro()).split("\n", 1)[1].strip())
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
    a("*In plain terms: " + READS["methods"] + "*")
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
        a("*In plain terms: " + READS["sweep"] + "*")
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
        a("*In plain terms: " + READS["recall"] + "*")
        a("")
        a(T["missed"].md())
        a("")
    if "shap" in T:
        a("## Feature importance (TreeSHAP, counts + LR regressor)")
        a("")
        a("*In plain terms: " + READS["shap"] + "*")
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
    a("## Glossary")
    a("")
    for term, definition in GLOSSARY:
        a(f"- **{term}** — {definition}")
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
        "## Links",
        "",
        f"- [Site]({SITE}): this report as a static page.",
        "\n".join(f"- [{n}]({u})" for n, u in LINKS),
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
        "url": SITE,
        "codeRepository": LINKS[0][1],
        "sameAs": [u for _, u in LINKS],
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
    idx = splice(idx, "<!-- PRERENDER:NAV -->", "<!-- /PRERENDER:NAV -->", nav_html())
    idx = splice(idx, "<!-- PRERENDER:FOOT -->", "<!-- /PRERENDER:FOOT -->", foot_html())
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
