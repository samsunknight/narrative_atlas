"""A18 (referee response): anchored ABSOLUTE cross-medium convergence for the sixteen structural
attributes, to answer the objection that the headline convergence estimator ranks within medium and
so removes absolute level differences by construction.

A4/A08 percentile-rank each attribute WITHIN medium before taking medium x decade centroids. That
measures alignment of within-medium relative position over time, not whether the media grow closer in
absolute organization. The thought experiment the referees raise: if television sits ten units above
the novel on an attribute but both move from their historical low to their historical high together,
the within-medium rank distance collapses while the absolute ten-unit gap never closes.

The sixteen structural attributes are the one layer measured on identical cross-medium scales and
supported by the invariance battery, so they are the layer where an anchored absolute distance is
licensed. Here we standardize each of the sixteen on the POOLED across-medium distribution (one mean
and sd per attribute over all works of all media), which preserves any cross-medium level difference
rather than centering it away, then take medium x decade centroids, the Euclidean cross-medium
distance per decade, and the 1950->2010 and 1950->1990 change, with a cell-resampling bootstrap.
We report the anchored trajectory beside the within-medium-ranked one so the two can be read together.

Bounded, single-threaded. Writes pnas-sub/analysis/out/A18_anchored_convergence.json. Run from the root.
"""
import os, sys, json
import numpy as np, pandas as pd

ROOT = os.environ.get("NARRATIVE_ATLAS_ROOT", os.path.expanduser("~/uoft/style_evolves")); os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "code"))
from _methods import load_spine, VAL16_KEYS

SP = {m: load_spine(m) for m in ["film", "book", "tv"]}
STRUCT = [c for c in SP["book"].columns if any(k in c.lower() for k in VAL16_KEYS)
          and c in SP["tv"].columns and c in SP["film"].columns]
assert len(STRUCT) == 16, f"expected 16 structural attributes, got {len(STRUCT)}"
print("structural attributes:", len(STRUCT))

# Pooled standardization: one mean/sd per attribute over ALL works of ALL media (preserves cross-medium
# level differences; only puts the sixteen attributes on a common scale for the Euclidean distance).
POOL = pd.concat([SP[m][STRUCT] for m in SP], ignore_index=True)
MU, SD = POOL.mean(), POOL.replace([np.inf, -np.inf], np.nan).std().replace(0, 1.0)

DECS = list(range(1950, 2011, 10)); MIN = 30
PAIRS = {"novel-tv": ("book", "tv"), "film-tv": ("film", "tv"), "film-novel": ("film", "book")}

# per medium: pooled-z structural matrix (absolute) and within-medium percentile-rank matrix (for the
# side-by-side comparison), plus decade cell indices.
ABS = {m: ((SP[m][STRUCT] - MU) / SD).values for m in SP}
RNK = {m: SP[m][STRUCT].rank(pct=True).values for m in SP}
YR = {m: SP[m]["year"].values for m in SP}
CELL = {m: {d: np.where((YR[m] >= d) & (YR[m] < d + 10))[0] for d in DECS} for m in SP}

def dist(M, a, b, d, idx_a=None, idx_b=None):
    ia = CELL[a][d] if idx_a is None else idx_a
    ib = CELL[b][d] if idx_b is None else idx_b
    if len(ia) < MIN or len(ib) < MIN: return np.nan
    ca = np.nanmean(M[a][ia], axis=0); cb = np.nanmean(M[b][ib], axis=0)
    return float(np.linalg.norm(ca - cb))

def traj(M):
    return {p: {d: dist(M, a, b, d) for d in DECS} for p, (a, b) in PAIRS.items()}

def boot_change(M, a, b, d0, d1, B=1000):
    """Bootstrap the distance change d(d0)-d(d1) by resampling works within each medium-decade cell."""
    base0, base1 = dist(M, a, b, d0), dist(M, a, b, d1)
    if np.isnan(base0) or np.isnan(base1): return None
    rng = np.random.default_rng(20260823)
    ch = []
    for _ in range(B):
        def rs(m, d):
            idx = CELL[m][d]; return rng.choice(idx, size=len(idx), replace=True)
        s0 = dist(M, a, b, d0, rs(a, d0), rs(b, d0))
        s1 = dist(M, a, b, d1, rs(a, d1), rs(b, d1))
        ch.append(s0 - s1)
    ch = np.array(ch)
    return {"d0": round(base0, 4), "d1": round(base1, 4),
            "change": round(base0 - base1, 4),
            "ci": [round(float(np.percentile(ch, 2.5)), 4), round(float(np.percentile(ch, 97.5)), 4)],
            "frac_of_1950": round((base0 - base1) / base0, 4) if base0 else None}

def centroid(m, d):
    idx = CELL[m][d]
    return np.nanmean(ABS[m][idx], axis=0) if len(idx) >= MIN else None

def asymmetry(a, b, d0=1950, d1=2010):
    """Decompose the a-b narrowing into each medium's own displacement projected onto the baseline
    inter-medium axis. 'largely b moving toward a' holds if b's toward-a projection dominates a's."""
    ca0, cb0, ca1, cb1 = centroid(a, d0), centroid(b, d0), centroid(a, d1), centroid(b, d1)
    if any(x is None for x in (ca0, cb0, ca1, cb1)): return None
    axis = ca0 - cb0; L = np.linalg.norm(axis)
    if L == 0: return None
    u = axis / L                      # unit vector pointing from b toward a at baseline
    da = ca1 - ca0; db = cb1 - cb0    # each medium's displacement over the century
    proj_b_toward_a = float(db @ u)   # >0 means b moved toward where a was
    proj_a_toward_b = float(-(da @ u))# >0 means a moved toward where b was
    total = proj_b_toward_a + proj_a_toward_b
    return {"baseline_gap": round(L, 4),
            f"{b}_toward_{a}": round(proj_b_toward_a, 4),
            f"{a}_toward_{b}": round(proj_a_toward_b, 4),
            f"share_{b}": round(proj_b_toward_a / total, 4) if total else None,
            "disp_norm_" + a: round(float(np.linalg.norm(da)), 4),
            "disp_norm_" + b: round(float(np.linalg.norm(db)), 4)}

out = {"n_structural": len(STRUCT), "attributes": STRUCT,
       "absolute_trajectory": {p: {str(d): (None if np.isnan(v) else round(v, 4))
                                   for d, v in dd.items()} for p, dd in traj(ABS).items()},
       "ranked_trajectory": {p: {str(d): (None if np.isnan(v) else round(v, 4))
                                 for d, v in dd.items()} for p, dd in traj(RNK).items()},
       "absolute_change": {}, "ranked_change": {}, "asymmetry": {}}

for p, (a, b) in PAIRS.items():
    out["absolute_change"][p] = {"1950_2010": boot_change(ABS, a, b, 1950, 2010),
                                 "1950_1990": boot_change(ABS, a, b, 1950, 1990)}
    out["ranked_change"][p] = {"1950_2010": boot_change(RNK, a, b, 1950, 2010),
                               "1950_1990": boot_change(RNK, a, b, 1950, 1990)}
# asymmetry for the two TV pairs: is the narrowing driven by TV moving toward film/novel?
out["asymmetry"]["novel-tv"] = asymmetry("book", "tv")
out["asymmetry"]["film-tv"] = asymmetry("film", "tv")

print("\n=== ABSOLUTE (pooled-standardized) structural distance by decade ===")
for p, dd in out["absolute_trajectory"].items():
    print(f"  {p:11}", {d: dd[str(d)] for d in DECS})
print("\n=== absolute 1950->2010 change (structural) ===")
for p in PAIRS:
    c = out["absolute_change"][p]["1950_2010"]
    print(f"  {p:11} {c}")
print("\n=== ranked (within-medium) 1950->2010 change, for comparison ===")
for p in PAIRS:
    c = out["ranked_change"][p]["1950_2010"]
    print(f"  {p:11} {c}")
print("\n=== asymmetry: is the narrowing TV moving toward film/novel? ===")
for p in ("novel-tv", "film-tv"):
    print(f"  {p:11} {out['asymmetry'][p]}")

os.makedirs("pnas-sub/analysis/out", exist_ok=True)
json.dump(out, open("pnas-sub/analysis/out/A18_anchored_convergence.json", "w"), indent=2)
print("\nwrote pnas-sub/analysis/out/A18_anchored_convergence.json")
