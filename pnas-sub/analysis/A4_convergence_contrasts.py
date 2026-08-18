"""A4 (PNAS Phase 2): turn "convergence is a mood phenomenon" into a tested contrast.

Reuses fig_f6_crosscutting's transform exactly: per comparable layer (mood, genre, arc), take the
shared attributes across the three media, percentile-rank within each medium (scale-invariant), and
measure the cross-media centroid distance by decade. Then compute the DISTANCE CHANGE (convergence)
from the 1950s to the 1990 trough and to the 2010s for each layer, and the direct paired contrasts
mood-vs-genre and mood-vs-arc, all with a shared nonparametric bootstrap (resample works within
medium x decade once per rep, recompute every layer, so the contrasts are properly paired).
Bounded, single-threaded. Writes results JSON. Run from ~/uoft/style_evolves/.
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

def dist(a, b, L, d, idx):
    ia, ib = idx[a][d], idx[b][d]
    if len(ia) < MIN or len(ib) < MIN: return np.nan
    ca = np.nanmean(PZ[a][L][ia], axis=0); cb = np.nanmean(PZ[b][L][ib], axis=0)
    return np.linalg.norm(ca - cb)

def all_stats(idx):
    """per pair/layer: distance at 1950, 1990, 2010; convergence to trough (1950-1990) and full (1950-2010)."""
    out = {}
    for pn, (a, b) in PAIRS.items():
        for L in LAYERS:
            d50, d90, d10 = dist(a, b, L, 1950, idx), dist(a, b, L, 1990, idx), dist(a, b, L, 2010, idx)
            out[(pn, L)] = {"d1950": d50, "d1990": d90, "d2010": d10,
                            "conv_trough": d50 - d90, "conv_full": d50 - d10}
    return out

pt = all_stats(CELL)
rng = np.random.RandomState(0); REPS = 500
def resample():
    return {m: {d: (rng.choice(CELL[m][d], len(CELL[m][d]), replace=True) if len(CELL[m][d]) else CELL[m][d])
                for d in DECS} for m in FR}
boot = [all_stats(resample()) for _ in range(REPS)]

def ci(vals):
    v = np.array([x for x in vals if not np.isnan(x)])
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] if len(v) else [None, None]

res = {"layers": LAYERS, "pairs": list(PAIRS), "reps": REPS, "by_pair_layer": {}, "contrasts": {}}
print("=== convergence by pair x layer (distance 1950 -> 1990 trough -> 2010; conv = 1950 minus later) ===")
for pn, (a, b) in PAIRS.items():
    for L in LAYERS:
        s = pt[(pn, L)]
        ci_tr = ci([bt[(pn, L)]["conv_trough"] for bt in boot])
        ci_fu = ci([bt[(pn, L)]["conv_full"] for bt in boot])
        res["by_pair_layer"][f"{pn}|{L}"] = {"d1950": round(s["d1950"], 3), "d1990": round(s["d1990"], 3),
            "d2010": round(s["d2010"], 3), "conv_trough": round(s["conv_trough"], 3), "conv_trough_CI": ci_tr,
            "conv_full": round(s["conv_full"], 3), "conv_full_CI": ci_fu}
        print(f"  {pn:9s} {L:6s}: 1950={s['d1950']:.2f} 1990={s['d1990']:.2f} 2010={s['d2010']:.2f} | "
              f"Δto1990={s['conv_trough']:+.2f} CI{ci_tr} | Δto2010={s['conv_full']:+.2f} CI{ci_fu}")

print("\n=== direct contrasts: does mood converge MORE than genre / arc? (paired bootstrap) ===")
for pn in PAIRS:
    for other in ["genre", "arc"]:
        for horizon, key in [("trough", "conv_trough"), ("full", "conv_full")]:
            diff_pt = pt[(pn, "mood")][key] - pt[(pn, other)][key]
            diff_ci = ci([bt[(pn, "mood")][key] - bt[(pn, other)][key] for bt in boot])
            sig = "*" if (diff_ci[0] is not None and (diff_ci[0] > 0 or diff_ci[1] < 0)) else " "
            res["contrasts"][f"{pn}|mood-minus-{other}|{horizon}"] = {"diff": round(diff_pt, 3), "CI": diff_ci, "excludes0": sig == "*"}
            print(f"  {pn:9s} mood - {other:5s} ({horizon:6s}): {diff_pt:+.2f}  CI{diff_ci} {sig}")

os.makedirs("pnas-sub/analysis/out", exist_ok=True)
json.dump(res, open("pnas-sub/analysis/out/A4_convergence_contrasts.json", "w"), indent=2)
print("\nwrote out/A4_convergence_contrasts.json")
