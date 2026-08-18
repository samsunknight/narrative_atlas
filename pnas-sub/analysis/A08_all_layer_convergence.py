"""A08 (PNAS rebuild, Phase 3 / S3b): put STRUCTURE on the same footing as mood, genre and arc in the
cross-media convergence, so "convergence is selective by layer" is tested with the structural layer
present rather than inferred from the separate medium-vs-era analysis.

The structural layer is the sixteen cross-medium attributes (the geometry basis), which live in the
structural spine; mood/genre/arc live in the dense century frame. We join the spine's structural
columns onto the frame by work id, giving one per-medium frame in which all four layers share the same
works, then apply A4's frozen estimator to every layer: percentile-rank within medium, medium x decade
centroids, distance, convergence = distance(1950) - distance(later), with a single shared bootstrap so
the mood-minus-structure / mood-minus-genre / mood-minus-arc contrasts are paired. Robustness: distance
normalized per attribute, Mahalanobis whitening by within-layer covariance, and matched-dimension
subsampling (draw mood attributes down to each comparison layer's dimensionality), so mood's larger
attribute count cannot by itself explain a larger convergence.

Bounded, single-threaded. Writes analysis/out/A08_all_layer_convergence.json. Run from the root.
"""
import os, sys, json
import numpy as np, pandas as pd

ROOT = os.environ.get("NARRATIVE_ATLAS_ROOT", os.path.expanduser("~/uoft/style_evolves")); os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "code"))
from _methods import load_spine, VAL16_KEYS, build_registries

FR = {m: pd.read_parquet(f"data/atlas/century_frame_{m}.parquet").rename(columns={"idx": "id"})
      for m in ["film", "book", "tv"]}
SP = {m: load_spine(m) for m in ["film", "book", "tv"]}
reg = build_registries()

# Layers sourced from their declared registries (not prefix-matched), so the basis is canonical:
# structure = geometry_shared16 (the 16 cross-medium spine columns); mood/genre/arc = convergence_*.
STRUCT = [c for c in SP["book"].columns if any(k in c.lower() for k in VAL16_KEYS)
          and c in SP["tv"].columns and c in SP["film"].columns]
def reg_layer(name):
    attrs = [r["attribute"] for r in reg[f"convergence_{name}"]]
    return [a for a in attrs if all(a in FR[m].columns for m in FR)]
LCOLS = {"structure": STRUCT, "mood": reg_layer("mood"), "genre": reg_layer("genre"), "arc": reg_layer("arc")}
print("layer sizes:", {k: len(v) for k, v in LCOLS.items()})

# unified per-medium frame: spine (id, year, structure) inner-joined to frame (mood/genre/arc)
FRAME_ATTRS = LCOLS["mood"] + LCOLS["genre"] + LCOLS["arc"]
UNI = {}
for m in ["film", "book", "tv"]:
    sp = SP[m][["id", "year"] + STRUCT]
    fr = FR[m][["id"] + [c for c in FRAME_ATTRS if c in FR[m].columns]]
    UNI[m] = sp.merge(fr, on="id", how="inner")
    print(f"  {m}: unified n={len(UNI[m])} (spine {len(SP[m])}, frame {len(FR[m])})")

LAYERS = ["structure", "mood", "genre", "arc"]
PAIRS = {"novel-tv": ("book", "tv"), "film-tv": ("film", "tv"), "film-novel": ("film", "book")}
DECS = list(range(1950, 2011, 10)); MIN = 30

# precompute per medium: percentile ranks per layer, decade-cell indices, year
def prep(frames):
    PZ = {m: {L: frames[m][LCOLS[L]].rank(pct=True).values for L in LAYERS} for m in frames}
    YR = {m: frames[m]["year"].values for m in frames}
    CELL = {m: {d: np.where((YR[m] >= d) & (YR[m] < d + 10))[0] for d in DECS} for m in frames}
    return PZ, CELL

def cov_whitener(frames, L):
    """Inverse-sqrt of the pooled within-medium correlation matrix of a layer, for Mahalanobis distance."""
    X = np.vstack([pd.DataFrame(frames[m][LCOLS[L]].rank(pct=True).values).values for m in frames])
    C = np.corrcoef(np.nan_to_num(X, nan=np.nanmean(X)), rowvar=False)
    C = np.nan_to_num(C, nan=0.0); np.fill_diagonal(C, 1.0)
    w, V = np.linalg.eigh(C); w = np.clip(w, 1e-3, None)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T

def dist(PZ, CELL, a, b, L, d, W=None, per_attr=False):
    ia, ib = CELL[a][L][d] if False else CELL[a][d], CELL[b][d]
    if len(ia) < MIN or len(ib) < MIN: return np.nan
    ca = np.nanmean(PZ[a][L][ia], axis=0); cb = np.nanmean(PZ[b][L][ib], axis=0)
    diff = ca - cb
    if W is not None: diff = W @ diff
    dd = float(np.linalg.norm(diff))
    if per_attr: dd /= np.sqrt(len(diff))
    return dd

def conv(PZ, CELL, a, b, L, W=None, per_attr=False):
    d50 = dist(PZ, CELL, a, b, L, 1950, W, per_attr)
    d90 = dist(PZ, CELL, a, b, L, 1990, W, per_attr)
    d10 = dist(PZ, CELL, a, b, L, 2010, W, per_attr)
    return {"d1950": d50, "d1990": d90, "d2010": d10,
            "conv_trough": d50 - d90 if not (np.isnan(d50) or np.isnan(d90)) else np.nan,
            "conv_full": d50 - d10 if not (np.isnan(d50) or np.isnan(d10)) else np.nan}

def ci(vals):
    v = np.array([x for x in vals if x is not None and not np.isnan(x)])
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] if len(v) >= 20 else [None, None]

PZ0, CELL0 = prep(UNI)
WHIT = {L: cov_whitener(UNI, L) for L in LAYERS}
REPS = 300
rng = np.random.RandomState(0)
def resample_cells():
    return {m: {d: (rng.choice(CELL0[m][d], len(CELL0[m][d]), replace=True) if len(CELL0[m][d]) else CELL0[m][d])
               for d in DECS} for m in UNI}
boots = [resample_cells() for _ in range(REPS)]

def run_variant(W=None, per_attr=False, mood_cols=None):
    """point + bootstrap convergence for every pair x layer under a distance variant; mood_cols overrides
    the mood attribute set (for matched-dimension subsampling)."""
    global LCOLS, PZ0
    saved = LCOLS["mood"];
    if mood_cols is not None:
        LCOLS["mood"] = mood_cols
        PZv = {m: {L: (UNI[m][LCOLS[L]].rank(pct=True).values if L == "mood" else PZ0[m][L]) for L in LAYERS} for m in UNI}
    else:
        PZv = PZ0
    pt = {(pn, L): conv(PZv, CELL0, a, b, L, W, per_attr) for pn, (a, b) in PAIRS.items() for L in LAYERS}
    bt = [{(pn, L): conv(PZv, bc, a, b, L, W, per_attr) for pn, (a, b) in PAIRS.items() for L in LAYERS} for bc in boots]
    LCOLS["mood"] = saved
    return pt, bt

def summarize(pt, bt, key="conv_full"):
    out = {"by_pair_layer": {}, "contrasts": {}}
    for pn in PAIRS:
        for L in LAYERS:
            out["by_pair_layer"][f"{pn}|{L}"] = {key: None if np.isnan(pt[(pn, L)][key]) else round(pt[(pn, L)][key], 3),
                                                 "CI": ci([b[(pn, L)][key] for b in bt])}
        for other in ["structure", "genre", "arc"]:
            dp = pt[(pn, "mood")][key] - pt[(pn, other)][key]
            dci = ci([b[(pn, "mood")][key] - b[(pn, other)][key] for b in bt])
            excl = dci[0] is not None and (dci[0] > 0 or dci[1] < 0)
            out["contrasts"][f"{pn}|mood-minus-{other}"] = {"diff": None if np.isnan(dp) else round(dp, 3),
                                                            "CI": dci, "excludes0": bool(excl)}
    return out

res = {"layer_sizes": {k: len(v) for k, v in LCOLS.items()}, "min_cell": MIN, "reps": REPS,
       "unified_n": {m: int(len(UNI[m])) for m in UNI}, "variants": {}}

# --- main (Euclidean, full horizon) ---
pt, bt = run_variant()
res["variants"]["euclidean_full"] = summarize(pt, bt, "conv_full")
res["variants"]["euclidean_trough"] = summarize(pt, bt, "conv_trough")
# --- per-attribute normalized + Mahalanobis whitened ---
ptn, btn = run_variant(per_attr=True); res["variants"]["per_attr_full"] = summarize(ptn, btn, "conv_full")
ptw, btw = run_variant(W={L: WHIT[L] for L in LAYERS}) if False else (None, None)
# whitening is per-layer; run each layer's whitener
def run_whitened():
    pt = {}; bt = [dict() for _ in range(REPS)]
    for pn, (a, b) in PAIRS.items():
        for L in LAYERS:
            pt[(pn, L)] = conv(PZ0, CELL0, a, b, L, WHIT[L])
            for i, bc in enumerate(boots): bt[i][(pn, L)] = conv(PZ0, bc, a, b, L, WHIT[L])
    return pt, bt
ptw, btw = run_whitened(); res["variants"]["whitened_full"] = summarize(ptw, btw, "conv_full")
# --- matched-dimension: mood down to structure(16) and arc(9) sizes ---
mood_all = LCOLS["mood"]
for tgt, k in [("match_structure", len(LCOLS["structure"])), ("match_arc", len(LCOLS["arc"]))]:
    rr = np.random.RandomState(1); DRAWS = 25
    diffs = {pn: {o: [] for o in ["structure", "genre", "arc"]} for pn in PAIRS}
    for _ in range(DRAWS):
        cols = list(rr.choice(mood_all, size=min(k, len(mood_all)), replace=False))
        PZv = {m: {L: (UNI[m][cols].rank(pct=True).values if L == "mood" else PZ0[m][L]) for L in LAYERS} for m in UNI}
        for pn, (a, b) in PAIRS.items():
            cm = conv(PZv, CELL0, a, b, "mood")["conv_full"]
            for o in ["structure", "genre", "arc"]:
                diffs[pn][o].append(cm - conv(PZv, CELL0, a, b, o)["conv_full"])
    res["variants"][f"matched_{tgt}"] = {"draws": DRAWS, "mood_k": k, "contrasts": {
        f"{pn}|mood-minus-{o}": {"mean_diff": round(float(np.nanmean(diffs[pn][o])), 3),
                                 "frac_positive": round(float(np.nanmean(np.array(diffs[pn][o]) > 0)), 3)}
        for pn in PAIRS for o in ["structure", "genre", "arc"]}}

# --- console + wording verdict ---
print("\n=== all-layer convergence (Euclidean, full horizon; conv = dist1950 - dist2010) ===")
ef = res["variants"]["euclidean_full"]
for pn in PAIRS:
    row = "  %-11s " % pn + " ".join(f"{L[:4]}={ef['by_pair_layer'][f'{pn}|{L}']['conv_full']}" for L in LAYERS)
    print(row)
print("\n=== does mood converge MORE than each layer? (Euclidean full) ===")
for pn in PAIRS:
    for o in ["structure", "genre", "arc"]:
        c = ef["contrasts"][f"{pn}|mood-minus-{o}"]
        print(f"  {pn:11s} mood-{o:9s}: {c['diff']} CI{c['CI']} {'*' if c['excludes0'] else ' '}")
print("\n=== matched-dimension (mood subsampled): frac of draws with mood>comparison ===")
for tgt in ["matched_match_structure", "matched_match_arc"]:
    v = res["variants"][tgt]
    print(f"  [{tgt} k={v['mood_k']}]")
    for pn in ["novel-tv", "film-tv"]:
        for o in ["structure", "genre", "arc"]:
            c = v["contrasts"][f"{pn}|mood-minus-{o}"]
            print(f"    {pn:9s} mood-{o:9s}: mean {c['mean_diff']} frac+ {c['frac_positive']}")

# structure convergence + persistence verdict (novel-tv structural distance trajectory)
struct_traj = {pn: {int(d): round(dist(PZ0, CELL0, PAIRS[pn][0], PAIRS[pn][1], "structure", d), 3) for d in DECS}
               for pn in ["novel-tv", "film-tv", "film-novel"]}
res["structure_distance_trajectory"] = struct_traj
print("\n=== structural distance trajectory by decade (does structure persist?) ===")
for pn, t in struct_traj.items():
    print(f"  {pn:11s}: " + " ".join(f"{d}:{v}" for d, v in t.items()))

os.makedirs("pnas-sub/analysis/out", exist_ok=True)
json.dump(res, open("pnas-sub/analysis/out/A08_all_layer_convergence.json", "w"), indent=2)
print("\nwrote out/A08_all_layer_convergence.json")
