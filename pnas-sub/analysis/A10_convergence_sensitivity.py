"""A10 (PNAS Phase 2, referee response): show the selective-convergence result as a full
distance trajectory with sensitivity analysis, not a single endpoint contrast.

Mirrors A4_convergence_contrasts.py's transform EXACTLY: per comparable layer (mood, genre, arc)
take the shared attributes across the three media, percentile-rank within each medium
(scale-invariant), and measure the cross-media centroid distance (Euclidean) by decade over cells
with >= 30 works. Pairs are novel-tv=(book,tv) and film-tv=(film,tv).

This script adds five referee-facing views:
 1. SELF-CHECK: reproduce A4's endpoint contrasts (novel-tv mood-minus-genre full ~ 0.31).
 2. FULL TRAJECTORY: centroid distance at every decade 1950..2010 with per-decade cell counts.
 3. BASELINE + PROPORTIONAL: 1950 dist, 2010 dist, absolute change, proportional change
    (change / 1950 dist) -- addresses "mood just had more room to converge".
 4. ENDPOINT SENSITIVITY: mood-minus-genre and mood-minus-arc under alternative windows.
 5. PER-ATTRIBUTE + LEAVE-ONE-OUT: mean per-attribute convergence per layer, and the mood-layer
    leave-one-attribute-out range (min/max mood convergence when each single mood attr is dropped).

No LLM, no new data. Bounded, single-threaded. Run from ~/uoft/style_evolves/.
"""
import os, json
import numpy as np, pandas as pd

ROOT = os.environ.get("NARRATIVE_ATLAS_ROOT", os.path.expanduser("~/uoft/style_evolves")); os.chdir(ROOT)
man = pd.read_csv("data/validation/rescore_manifest.csv"); man = man[man.deploy == True]
LAY = man.set_index("attr_id")["layer"].to_dict()
FR = {m: pd.read_parquet(f"data/atlas/century_frame_{m}.parquet") for m in ["film", "book", "tv"]}
LAYERS = ["mood", "genre", "arc"]
PAIRS = {"novel-tv": ("book", "tv"), "film-tv": ("film", "tv")}
DECS = list(range(1950, 2011, 10))
MIN = 30

# shared attributes per layer (present in all three media), percentile-ranked within each medium
def attrs(df, L): return [c for c in df.columns if LAY.get(c) == L]
SH = {L: [c for c in attrs(FR["film"], L) if c in FR["book"].columns and c in FR["tv"].columns] for L in LAYERS}
PZ = {m: {L: FR[m][SH[L]].rank(pct=True).values for L in LAYERS} for m in FR}
YR = {m: FR[m]["year"].values for m in FR}
CELL = {m: {d: np.where((YR[m] >= d) & (YR[m] < d + 10))[0] for d in DECS} for m in FR}

def cell_dist(a, b, L, d, cols=None):
    """centroid distance for pair (a,b) at decade d; optionally on a column subset `cols`."""
    ia, ib = CELL[a][d], CELL[b][d]
    if len(ia) < MIN or len(ib) < MIN: return np.nan
    Ca, Cb = PZ[a][L], PZ[b][L]
    if cols is not None:
        Ca, Cb = Ca[:, cols], Cb[:, cols]
    ca = np.nanmean(Ca[ia], axis=0); cb = np.nanmean(Cb[ib], axis=0)
    return float(np.linalg.norm(ca - cb))

res = {"layers": LAYERS, "pairs": list(PAIRS), "decades": DECS, "min_cell": MIN,
       "n_attrs": {L: len(SH[L]) for L in LAYERS}, "shared_attrs": SH}

# ---------------------------------------------------------------------------
# 1. SELF-CHECK: reproduce A4 endpoint contrasts (point estimates)
# ---------------------------------------------------------------------------
selfcheck = {}
for pn, (a, b) in PAIRS.items():
    conv_full = {L: cell_dist(a, b, L, 1950) - cell_dist(a, b, L, 2010) for L in LAYERS}
    conv_tr = {L: cell_dist(a, b, L, 1950) - cell_dist(a, b, L, 1990) for L in LAYERS}
    for other in ["genre", "arc"]:
        selfcheck[f"{pn}|mood-minus-{other}|full"] = round(conv_full["mood"] - conv_full[other], 3)
        selfcheck[f"{pn}|mood-minus-{other}|trough"] = round(conv_tr["mood"] - conv_tr[other], 3)
res["selfcheck_endpoint_contrasts"] = selfcheck

# ---------------------------------------------------------------------------
# 2. FULL TRAJECTORY: distance at every decade + per-decade cell counts
# ---------------------------------------------------------------------------
traj = {}
counts = {pn: {str(d): {a: int(len(CELL[a][d])), b: int(len(CELL[b][d]))} for d in DECS}
          for pn, (a, b) in PAIRS.items()}
for pn, (a, b) in PAIRS.items():
    for L in LAYERS:
        traj[f"{pn}|{L}"] = [None if np.isnan(v := cell_dist(a, b, L, d)) else round(v, 4) for d in DECS]
res["trajectory"] = traj
res["cell_counts"] = counts

# ---------------------------------------------------------------------------
# 3. BASELINE + PROPORTIONAL change (1950 -> 2010)
# ---------------------------------------------------------------------------
prop = {}
for pn, (a, b) in PAIRS.items():
    for L in LAYERS:
        d50, d10 = cell_dist(a, b, L, 1950), cell_dist(a, b, L, 2010)
        chg = d50 - d10
        prop[f"{pn}|{L}"] = {"d1950": round(d50, 4), "d2010": round(d10, 4),
                             "abs_change": round(chg, 4),
                             "prop_change": round(chg / d50, 4) if d50 else None}
res["baseline_proportional"] = prop

# ---------------------------------------------------------------------------
# 4. ENDPOINT SENSITIVITY: alternative (start, end) windows
# ---------------------------------------------------------------------------
WINDOWS = [(1950, 2010), (1960, 2010), (1970, 2000), (1950, 2000)]
sens = {}
for (s, e) in WINDOWS:
    row = {}
    for pn, (a, b) in PAIRS.items():
        conv = {L: cell_dist(a, b, L, s) - cell_dist(a, b, L, e) for L in LAYERS}
        # is mood the top-converging layer in this window?
        top = max(conv, key=lambda L: conv[L])
        row[pn] = {"conv": {L: round(conv[L], 4) for L in LAYERS},
                   "mood-minus-genre": round(conv["mood"] - conv["genre"], 4),
                   "mood-minus-arc": round(conv["mood"] - conv["arc"], 4),
                   "top_layer": top, "mood_is_top": top == "mood"}
    sens[f"{s}-{e}"] = row
res["endpoint_sensitivity"] = sens

# ---------------------------------------------------------------------------
# 5. PER-ATTRIBUTE convergence + mood leave-one-out range
# ---------------------------------------------------------------------------
per_attr = {}
for pn, (a, b) in PAIRS.items():
    for L in LAYERS:
        conv = cell_dist(a, b, L, 1950) - cell_dist(a, b, L, 2010)
        per_attr[f"{pn}|{L}"] = {"layer_conv": round(conv, 4), "n_attrs": len(SH[L]),
                                 "mean_per_attr_conv": round(conv / len(SH[L]), 5)}
res["per_attribute"] = per_attr

# leave-one-attribute-out for the MOOD layer: drop each single mood attr, recompute mood convergence
loo = {}
mood_cols = list(range(len(SH["mood"])))
for pn, (a, b) in PAIRS.items():
    vals = []
    for j in mood_cols:
        keep = [c for c in mood_cols if c != j]
        conv_j = cell_dist(a, b, "mood", 1950, cols=keep) - cell_dist(a, b, "mood", 2010, cols=keep)
        vals.append(conv_j)
    genre_conv = cell_dist(a, b, "genre", 1950) - cell_dist(a, b, "genre", 2010)
    arc_conv = cell_dist(a, b, "arc", 1950) - cell_dist(a, b, "arc", 2010)
    loo[pn] = {"mood_full_conv": round(cell_dist(a, b, "mood", 1950) - cell_dist(a, b, "mood", 2010), 4),
               "loo_min": round(float(np.min(vals)), 4), "loo_max": round(float(np.max(vals)), 4),
               "dropped_attr_at_min": SH["mood"][int(np.argmin(vals))],
               "genre_conv": round(genre_conv, 4), "arc_conv": round(arc_conv, 4),
               "loo_min_still_beats_genre": bool(np.min(vals) > genre_conv),
               "loo_min_still_beats_arc": bool(np.min(vals) > arc_conv)}
res["mood_leave_one_out"] = loo

# ---------------------------------------------------------------------------
# write + print summary
# ---------------------------------------------------------------------------
os.makedirs("pnas-sub/analysis/out", exist_ok=True)
json.dump(res, open("pnas-sub/analysis/out/A10_convergence_sensitivity.json", "w"), indent=2)

print("=== 1. SELF-CHECK vs A4 (endpoint contrasts, point estimates) ===")
print(f"  novel-tv mood-minus-genre full = {selfcheck['novel-tv|mood-minus-genre|full']}  (A4 ~ 0.31)")
for k, v in selfcheck.items():
    print(f"    {k:34s} = {v:+.3f}")

print("\n=== 2. FULL DISTANCE TRAJECTORY (centroid distance by decade) ===")
print("  decades: " + " ".join(f"{d:>6d}" for d in DECS))
for pn, (a, b) in PAIRS.items():
    print(f"  -- {pn} (cell counts {a}/{b}):")
    print("     counts: " + " ".join(
        f"{counts[pn][str(d)][a]:>2d}/{counts[pn][str(d)][b]:<3d}" for d in DECS))
    for L in LAYERS:
        row = traj[f"{pn}|{L}"]
        print(f"     {L:6s}: " + " ".join(("  n/a " if x is None else f"{x:6.3f}") for x in row))

print("\n=== 3. BASELINE + PROPORTIONAL change (1950 -> 2010) ===")
print(f"  {'pair|layer':16s} {'d1950':>7s} {'d2010':>7s} {'abs':>7s} {'prop':>7s}")
for k, v in prop.items():
    print(f"  {k:16s} {v['d1950']:7.3f} {v['d2010']:7.3f} {v['abs_change']:+7.3f} "
          f"{(v['prop_change'] if v['prop_change'] is not None else float('nan')):7.3f}")

print("\n=== 4. ENDPOINT SENSITIVITY (does mood converge most under each window?) ===")
print(f"  {'window':10s} {'pair':9s} {'mood':>7s} {'genre':>7s} {'arc':>7s} "
      f"{'m-genre':>8s} {'m-arc':>7s}  top   mood_top")
allmoodtop = True
for w, row in sens.items():
    for pn in PAIRS:
        r = row[pn]; c = r["conv"]
        allmoodtop &= r["mood_is_top"]
        print(f"  {w:10s} {pn:9s} {c['mood']:7.3f} {c['genre']:7.3f} {c['arc']:7.3f} "
              f"{r['mood-minus-genre']:+8.3f} {r['mood-minus-arc']:+7.3f}  {r['top_layer']:5s} {r['mood_is_top']}")

print("\n=== 5. PER-ATTRIBUTE + MOOD LEAVE-ONE-OUT ===")
print(f"  {'pair|layer':16s} {'layer_conv':>10s} {'n_attrs':>7s} {'mean/attr':>10s}")
for k, v in per_attr.items():
    print(f"  {k:16s} {v['layer_conv']:10.3f} {v['n_attrs']:7d} {v['mean_per_attr_conv']:10.4f}")
print("  mood leave-one-attribute-out (drop each single mood attr, recompute mood convergence):")
loo_ok = True
for pn, v in loo.items():
    loo_ok &= v["loo_min_still_beats_genre"] and v["loo_min_still_beats_arc"]
    print(f"    {pn:9s}: mood_full={v['mood_full_conv']:.3f}  LOO range [{v['loo_min']:.3f}, {v['loo_max']:.3f}]  "
          f"(min drops '{v['dropped_attr_at_min']}')")
    print(f"               vs genre {v['genre_conv']:.3f} / arc {v['arc_conv']:.3f}  "
          f"-> LOO-min beats genre={v['loo_min_still_beats_genre']}, beats arc={v['loo_min_still_beats_arc']}")

print("\n=== BOTTOM LINE ===")
print(f"  Mood is the top-converging layer in ALL {len(WINDOWS)} endpoint windows x {len(PAIRS)} pairs: {allmoodtop}")
print(f"  Mood stays top after per-attribute normalization + leave-one-out (min still beats genre & arc): {loo_ok}")
print("\nwrote pnas-sub/analysis/out/A10_convergence_sensitivity.json")
