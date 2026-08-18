"""A2 (PNAS Phase 2): uncertainty + null for the headline medium-vs-era variance ratio.

Replicates reproduce.py's between/within ratio on the 16-attribute cross-medium basis EXACTLY
(self-check: point estimate must round to 1.66), then adds:
  (i)  a nonparametric bootstrap 95% CI (resample works within each medium x decade cell, fixed
       pooled z-transform), and
  (ii) a permutation null that shuffles medium labels WITHIN each decade (preserving per-decade
       medium composition), giving a p-value for "medium organizes form beyond era."
Also reports the film-vs-novel-only ratio. Bounded, single-threaded. Writes results JSON.
Run from ~/uoft/style_evolves/.
"""
import os, sys, json
import numpy as np, pandas as pd

ROOT = os.environ.get("NARRATIVE_ATLAS_ROOT", os.path.expanduser("~/uoft/style_evolves"))
for cd in [os.path.join(os.path.dirname(ROOT), "code"), os.path.join(ROOT, "code")]:
    if os.path.isdir(cd) and cd not in sys.path:
        sys.path.insert(0, cd)
def P(*a): return os.path.join(ROOT, *a)

F = pd.read_csv(P("data/corpus/film_structural_1890_2025.csv")).rename(columns={"film_idx": "id"})
B = pd.read_csv(P("data/corpus/book_structural_1890_2025.csv")).rename(columns={"book_idx": "id"})
T = pd.read_csv(P("data/corpus/tv_structural_1890_2025.csv")).rename(columns={"tv_idx": "id"})
from _methods import VAL16_KEYS as VALCM_KEYS
VALCM = [c for c in B.columns if any(k in c.lower() for k in VALCM_KEYS) and c in T.columns and c in F.columns]
assert len(VALCM) == 16, f"expected 16-attr basis, got {len(VALCM)}"
DECS = list(range(1910, 2021, 10))          # ALL in-window decades: (year//10)*10 over win 1915-2020
MIN = 30

# EXACT reproduce.py replication: pooled z-transform over the FULL frames (no win, no dropna, NaN kept),
# then win(1915-2020), decade = (year//10)*10, skipna centroid means, cells with >=30 rows.
mu = pd.concat([B[VALCM], T[VALCM], F[VALCM]]).mean()
sd = pd.concat([B[VALCM], T[VALCM], F[VALCM]]).std().replace(0, 1)

def prep(df, medium):
    d = df[(df.year >= 1915) & (df.year <= 2020)].copy()
    d["dec"] = (d["year"] // 10) * 10
    d["m"] = medium
    return d[["m", "dec"] + VALCM]

D = pd.concat([prep(B, "bk"), prep(F, "fm"), prep(T, "tv")], ignore_index=True)
Z = ((D[VALCM] - mu) / sd).values          # NaN kept; skipna at centroid time
mvec = D["m"].values
dvec = D["dec"].values

def ratio_from(zmat, m_lab, d_lab, media):
    """between/within variance ratio over medium x decade centroids (>=MIN rows/cell; skipna means)."""
    C = {m: {} for m in media}
    for m in media:
        for d in DECS:
            sel = (m_lab == m) & (d_lab == d)
            if sel.sum() >= MIN:
                C[m][d] = np.nanmean(zmat[sel], axis=0)
    C = {m: v for m, v in C.items() if v}
    if len(C) < 2:
        return np.nan
    medmean = {m: np.mean(list(v.values()), axis=0) for m, v in C.items()}
    grand = np.mean([c for m in C for c in C[m].values()], axis=0)
    between = np.mean([np.mean((medmean[m] - grand) ** 2) for m in C])
    within = np.mean([np.mean([np.mean((c - medmean[m]) ** 2) for c in C[m].values()]) for m in C])
    return between / within if within > 0 else np.nan

# ---- point estimate (self-check vs reproduce.py = 1.66) ----
obs_all = ratio_from(Z, mvec, dvec, ["bk", "fm", "tv"])
obs_fn = ratio_from(Z, mvec, dvec, ["bk", "fm"])
print(f"point estimate ratio (3 media, 16-basis): {obs_all:.3f}  [reproduce.py pins 1.66]")
print(f"point estimate ratio (film vs novel only): {obs_fn:.3f}")

rng = np.random.RandomState(0)
REPS = 1000

# ---- (i) bootstrap CI: resample works within each medium x decade cell ----
def boot_once(media):
    idx = np.arange(len(Z))
    take = np.empty(0, dtype=int)
    for m in media:
        for d in DECS:
            cell = idx[(mvec == m) & (dvec == d)]
            if len(cell) >= MIN:
                take = np.concatenate([take, rng.choice(cell, len(cell), replace=True)])
    return ratio_from(Z[take], mvec[take], dvec[take], media)

boot_all = np.array([boot_once(["bk", "fm", "tv"]) for _ in range(REPS)])
boot_fn = np.array([boot_once(["bk", "fm"]) for _ in range(REPS)])
ci_all = np.nanpercentile(boot_all, [2.5, 97.5])
ci_fn = np.nanpercentile(boot_fn, [2.5, 97.5])
print(f"bootstrap 95% CI (3 media): [{ci_all[0]:.2f}, {ci_all[1]:.2f}]  median {np.nanmedian(boot_all):.2f}")
print(f"bootstrap 95% CI (film-novel): [{ci_fn[0]:.2f}, {ci_fn[1]:.2f}]  median {np.nanmedian(boot_fn):.2f}")

# ---- (ii) permutation null: shuffle medium labels WITHIN decade (preserve composition) ----
def perm_once(media):
    mp = mvec.copy()
    for d in DECS:
        sel = np.where(dvec == d)[0]
        if len(sel):
            mp[sel] = rng.permutation(mvec[sel])
    return ratio_from(Z, mp, dvec, media)

null_all = np.array([perm_once(["bk", "fm", "tv"]) for _ in range(REPS)])
p_all = (np.nansum(null_all >= obs_all) + 1) / (np.sum(~np.isnan(null_all)) + 1)
print(f"permutation null (3 media): mean {np.nanmean(null_all):.3f}, 95th pct "
      f"{np.nanpercentile(null_all,95):.3f}; p(null>=obs) = {p_all:.4f}")

out = {"basis_n": len(VALCM),
       "point_3media": round(float(obs_all), 3), "point_film_novel": round(float(obs_fn), 3),
       "boot_ci_3media": [round(float(ci_all[0]), 2), round(float(ci_all[1]), 2)],
       "boot_ci_film_novel": [round(float(ci_fn[0]), 2), round(float(ci_fn[1]), 2)],
       "perm_p_3media": float(p_all), "perm_null_mean": round(float(np.nanmean(null_all)), 3),
       "reps": REPS}
os.makedirs(os.path.join(ROOT, "pnas-sub/analysis/out"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "pnas-sub/analysis/out/A2_variance_ratio.json"), "w"), indent=2)
print("\nwrote out/A2_variance_ratio.json")
