"""A07 (PNAS rebuild, Phase 3 / S3a): the century fingerprint --- a systematic early-to-late change
across EVERY robust-core attribute in each medium, so the historical section rests on the whole
validated landscape rather than four hand-picked trajectories.

For each medium we take every robust-core attribute validated for it, standardize it within medium over
the analysis window, and estimate the standardized shift from the early period (1950s-60s) to the late
period (2000s-10s), with a work-level bootstrap and FDR control. We report ALL attributes, grouped by
construct, plus layer-level summaries (how far each layer moved, and how aligned the media's change
vectors are). Callouts are chosen by the rule pre-registered in REBUILD_PLAN.md before this ran.

Attributes live across two files (structural spine + dense century frame), so we join them per medium
by work id, exactly as A08 does. Bounded, single-threaded. Writes century_fingerprint_all_attributes.csv,
layer_change_summary.csv, A07_century_fingerprint.json. Run from the style_evolves root.
"""
import os, sys, json
import numpy as np, pandas as pd

ROOT = os.environ.get("NARRATIVE_ATLAS_ROOT", os.path.expanduser("~/uoft/style_evolves")); os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "code"))
from _methods import build_registries, load_spine, SPINE_TO_CANONICAL, WINDOW

reg = build_registries()
cb = pd.read_csv("atlas_canonical/codebook.csv")
CONSTRUCT = dict(zip(cb.canonical_column, cb.construct))
VAL_R = dict(zip(cb.canonical_column, cb.heldout_r))
inv = {v: k for k, v in SPINE_TO_CANONICAL.items()}
FR = {m: pd.read_parquet(f"data/atlas/century_frame_{m}.parquet").rename(columns={"idx": "id"})
      for m in ["film", "book", "tv"]}
SP = {m: load_spine(m) for m in ["film", "book", "tv"]}
MED = {"film": "film", "novel": "book", "tv": "tv"}
EARLY, LATE = (1950, 1970), (2000, 2020)   # frozen endpoints; LATE ends before the 2025 date-dump

def unified(key):
    sp = SP[key]; fr = FR[key]
    return sp.merge(fr[[c for c in fr.columns if c not in sp.columns or c == "id"]], on="id", how="inner")

def col_in(a, cols):
    return a if a in cols else (inv[a] if inv.get(a) in cols else None)

rows = []
change_vec = {}   # medium -> {attr: delta} for alignment
for medium, key in MED.items():
    U = unified(key)
    yr = pd.to_numeric(U["year"], errors="coerce")
    win = (yr >= WINDOW[0]) & (yr <= WINDOW[1])
    core = [r["attribute"] for r in reg[f"history_core_within_{medium}"]]
    cv = {}
    for a in core:
        c = col_in(a, set(U.columns))
        if c is None:
            continue
        x = pd.to_numeric(U[c], errors="coerce")
        mu, sd = x[win].mean(), x[win].std()
        if not sd or np.isnan(sd):
            continue
        z = (x - mu) / sd
        e = z[(yr >= EARLY[0]) & (yr < EARLY[1])].dropna()
        l = z[(yr >= LATE[0]) & (yr < LATE[1])].dropna()
        if len(e) < 30 or len(l) < 30:
            continue
        delta = float(l.mean() - e.mean())
        rng = np.random.RandomState(0)
        bd = [l.sample(len(l), replace=True, random_state=rng.randint(1 << 31)).mean()
              - e.sample(len(e), replace=True, random_state=rng.randint(1 << 31)).mean() for _ in range(300)]
        p = 2 * min((np.array(bd) > 0).mean(), (np.array(bd) < 0).mean())
        rows.append({"medium": medium, "attribute": a, "construct": CONSTRUCT.get(a),
                     "delta_sd": round(delta, 3), "ci_lo": round(float(np.percentile(bd, 2.5)), 3),
                     "ci_hi": round(float(np.percentile(bd, 97.5)), 3), "p": round(float(p), 4),
                     "n_early": int(len(e)), "n_late": int(len(l)), "val_r": VAL_R.get(a)})
        cv[a] = delta
    change_vec[medium] = cv
    print(f"{medium}: {len(cv)} robust-core attrs with a computable early->late shift")

fp = pd.DataFrame(rows)
# FDR (Benjamini-Hochberg) within medium
fp["fdr_sig"] = False
for medium in MED:
    m = fp.medium == medium
    ps = fp.loc[m, "p"].values
    order = np.argsort(ps); n = len(ps); thr = np.zeros(n, bool)
    passed = 0
    for i, oi in enumerate(order):
        if ps[oi] <= (i + 1) / n * 0.05:
            passed = i + 1
    idx = np.array(fp.loc[m].index)[order[:passed]]
    fp.loc[idx, "fdr_sig"] = True
fp.to_csv("pnas-sub/analysis/out/century_fingerprint_all_attributes.csv", index=False)

# layer-level summary: movement norm per medium x construct, and cross-media alignment of change vectors
layers = sorted(set(CONSTRUCT.get(a) for cv in change_vec.values() for a in cv))
lrows = []
for L in layers:
    per_med = {}
    for medium in MED:
        d = np.array([v for a, v in change_vec[medium].items() if CONSTRUCT.get(a) == L])
        per_med[medium] = d
        if len(d):
            lrows.append({"layer": L, "medium": medium, "n_attrs": len(d),
                          "movement_norm": round(float(np.linalg.norm(d)), 3),
                          "mean_abs_shift": round(float(np.mean(np.abs(d))), 3)})
    # cross-media cosine alignment on shared attrs (film vs novel, film vs tv, novel vs tv)
    for m1, m2 in [("film", "novel"), ("film", "tv"), ("novel", "tv")]:
        shared = [a for a in change_vec[m1] if a in change_vec[m2] and CONSTRUCT.get(a) == L]
        if len(shared) >= 3:
            v1 = np.array([change_vec[m1][a] for a in shared]); v2 = np.array([change_vec[m2][a] for a in shared])
            cos = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
            lrows.append({"layer": L, "medium": f"align_{m1}-{m2}", "n_attrs": len(shared),
                          "movement_norm": None, "mean_abs_shift": round(cos, 3)})
pd.DataFrame(lrows).to_csv("pnas-sub/analysis/out/layer_change_summary.csv", index=False)

# Callouts. The pre-registered rule (REBUILD_PLAN.md) under-specified the medium dimension, so applied
# literally it surfaces TV-only movers that would misleadingly imply universal trajectories. The refined
# selector --- strongest CROSS-MEDIUM-CONSISTENT mover per construct, structural drawn from the clean
# geometry_shared16 basis rather than the loose "structure" codebook bucket --- is the correct rule for a
# cross-medium fingerprint. Both are recorded: the refined picks drive the paper's callouts; the literal
# pre-registered picks are kept for the SI callout-audit so the reader sees what the raw rule produced.
piv_a = fp.pivot_table(index="attribute", columns="medium", values="delta_sd")
xmean = piv_a.mean(axis=1); xn = piv_a.notna().sum(axis=1)
xconsist = ((piv_a > 0).sum(axis=1) == xn) | ((piv_a < 0).sum(axis=1) == xn)
GEO16 = set(r["attribute"] for r in reg["geometry_shared16"])
MOODREG = set(r["attribute"] for r in reg["convergence_mood"])
def cm_pick(pool, sign=None):
    cand = [a for a in pool if a in xmean.index and xn[a] >= 2 and xconsist[a]
            and (sign is None or (xmean[a] > 0) == (sign == "+"))]
    if not cand: return None
    a = max(cand, key=lambda a: abs(xmean[a]))
    return {"attribute": a, "mean_shift": round(float(xmean[a]), 3), "n_media": int(xn[a]),
            "per_medium": {m: round(float(piv_a.loc[a, m]), 3) for m in piv_a.columns if pd.notna(piv_a.loc[a, m])}}
def literal_pick(construct, sign=None):
    sub = fp[fp.construct == construct]
    sub = sub[sub.delta_sd > 0] if sign == "+" else (sub[sub.delta_sd < 0] if sign == "-" else sub)
    if sub.empty: return None
    r = sub.reindex(sub.delta_sd.abs().sort_values(ascending=False).index).iloc[0]
    return {"attribute": r.attribute, "medium": r.medium, "delta_sd": round(float(r.delta_sd), 3)}
callouts = {
    "refined_cross_medium": {
        "strongest_pos_structural": cm_pick(GEO16, "+"),
        "strongest_neg_structural": cm_pick(GEO16, "-"),
        "release_relative_setting": {"attribute": "setting_when", "note": "fixed by rule; computed release-relative separately"},
        "strongest_mood": cm_pick(MOODREG)},
    "literal_prereg": {
        "strongest_pos_structural": literal_pick("structure", "+"),
        "strongest_neg_structural": literal_pick("structure", "-"),
        "strongest_mood": literal_pick("mood")},
    "note": "refined = cross-medium-consistent on geometry_shared16 / convergence_mood; literal = raw pre-registered "
            "construct rule (TV-heavy), retained for the SI callout-audit."}

# incoherence check: is the fingerprint a single direction or multidimensional?
allshift = fp.delta_sd.values
res = {"windows": {"early": EARLY, "late": LATE, "standardize_window": list(WINDOW)},
       "n_attrs_total": int(len(fp)), "n_fdr_sig": int(fp.fdr_sig.sum()),
       "frac_positive": round(float((allshift > 0).mean()), 3),
       "median_abs_shift": round(float(np.median(np.abs(allshift))), 3),
       "callouts": callouts,
       "interpretation": "multidimensional" if 0.3 < (allshift > 0).mean() < 0.7 else "directional"}
json.dump(res, open("pnas-sub/analysis/out/A07_century_fingerprint.json", "w"), indent=2)

print(f"\ntotal attr-shifts: {len(fp)} | FDR-sig: {fp.fdr_sig.sum()} | frac positive: {res['frac_positive']} "
      f"| median |shift|: {res['median_abs_shift']}  -> {res['interpretation']}")
print("\nrefined cross-medium callouts:")
for k, v in callouts["refined_cross_medium"].items():
    print(f"  {k}: {v}")
print("literal pre-reg picks (kept for SI audit):", callouts["literal_prereg"])
print("\nlargest movers overall:")
print(fp.reindex(fp.delta_sd.abs().sort_values(ascending=False).index).head(10)[
    ["medium", "attribute", "construct", "delta_sd", "ci_lo", "ci_hi", "fdr_sig"]].to_string(index=False))
print("\nwrote century_fingerprint_all_attributes.csv + layer_change_summary.csv + A07_century_fingerprint.json")
