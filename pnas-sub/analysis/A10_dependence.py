"""A10 (PNAS rebuild, Phase 4 / S4b): dependence among works.

Works are not independent --- they cluster in adaptation source families and title franchises --- so an
interval that treats 149k works as independent overstates precision. We re-estimate the two load-bearing
results under cluster resampling: (1) the ten adaptation deltas under a block bootstrap by SOURCE work
(so a much-adapted novel counts once), and (2) the medium-vs-era ratio under a one-per-franchise
restriction (title-stem clusters). Per the plan, we lead with effect sizes and specification stability,
not tiny p-values. Bounded, single-threaded. Writes analysis/out/A10_dependence.json. Run from the root.
"""
import os, sys, re, json
import numpy as np, pandas as pd

ROOT = os.environ.get("NARRATIVE_ATLAS_ROOT", os.path.expanduser("~/uoft/style_evolves")); os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "code"))
from _methods import VAL16_KEYS, load_spine, WINDOW
def norm(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())
def stem(s):
    """A crude franchise/series stem: title lowercased, cut at the first subtitle/number/sequel marker."""
    t = re.split(r"[:\-–]| part | chapter | episode | vol| \d| ii| iii| iv| \bthe \b", " " + str(s).lower() + " ", maxsplit=1)[0]
    return re.sub(r"[^a-z0-9]", "", t)[:24]

SP = {m: load_spine(m) for m in ["film", "book", "tv"]}
F, B = SP["film"], SP["book"]
OUT = "pnas-sub/analysis/out"

# ---- (1) adaptation deltas, block bootstrap by SOURCE ----
SM = {}
for c in [c for c in F.columns if c not in ("id", "title", "year")]:
    for k, n in [("science_fictional", "sci-fi"), ("fantastical", "fantastical"), ("realistic_was_the_world", "realistic"),
                 ("world_building", "world-building"), ("how_many_protagonists", "#protagonists"), ("competent", "competence"),
                 ("proactive", "proactiveness"), ("relatable", "relatability"), ("plot_driven", "plot-driven"),
                 ("character_driven", "character-driven")]:
        if k in c: SM[n] = c
PAIRS = pd.read_csv("data/matched/adaptation_pairs.csv").dropna(subset=["filmLabel", "bookLabel"])
Fn = F.assign(nt=F.title.map(norm)).drop_duplicates("nt").set_index("nt")
Bn = B.assign(nt=B.title.map(norm)).drop_duplicates("nt").set_index("nt")
psd = {n: pd.concat([F[a], B[a]]).astype(float).std() for n, a in SM.items()}
rows = []
for _, r in PAIRS.iterrows():
    ft, bt = norm(r.filmLabel), norm(r.bookLabel)
    if ft in Fn.index and bt in Bn.index:
        rows.append({"src": bt, **{n: Fn.loc[ft][a] - Bn.loc[bt][a] for n, a in SM.items()}})
AD = pd.DataFrame(rows)
sources = AD.src.unique()
HEAD = ["realistic", "plot-driven", "character-driven", "sci-fi", "fantastical", "#protagonists"]
rng = np.random.RandomState(0)
def cluster_boot(col):
    out = []
    for _ in range(1000):
        pick = rng.choice(sources, len(sources), replace=True)
        sub = pd.concat([AD[AD.src == s] for s in pick])
        out.append(sub[col].astype(float).mean() / psd[col])
    return [round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)]
adapt = {}
for n in HEAD:
    pt = float(AD[n].astype(float).mean() / psd[n]); ci = cluster_boot(n)
    adapt[n] = {"delta_SD": round(pt, 3), "source_block_CI": ci, "excludes0": bool(ci[0] > 0 or ci[1] < 0)}
print(f"adaptation source-block bootstrap ({len(sources)} source clusters, {len(AD)} pairs):")
for n in HEAD:
    print(f"  {n:15s} {adapt[n]['delta_SD']:+.3f}  block-CI{adapt[n]['source_block_CI']} {'*' if adapt[n]['excludes0'] else ' '}")

# ---- (2) medium-vs-era ratio, one-per-franchise ----
VALCM = [c for c in B.columns if any(k in c.lower() for k in VAL16_KEYS) and c in SP["tv"].columns and c in F.columns]
def prep(df):
    d = df.copy(); y = pd.to_numeric(d.year, errors="coerce")
    d = d[(y >= WINDOW[0]) & (y <= WINDOW[1])].copy(); d["dec"] = (pd.to_numeric(d.year) // 10 * 10)
    d["stem"] = d.title.map(stem); return d
D = {m: prep(SP[m]) for m in SP}
def ratio(frames):
    mu = {m: {dc: frames[m][frames[m].dec == dc][VALCM].mean() for dc in sorted(frames[m].dec.dropna().unique())
              if (frames[m].dec == dc).sum() >= 30} for m in frames}
    cent = {m: pd.concat(mu[m].values(), axis=1).mean(axis=1) for m in frames}
    grand = pd.concat(cent.values(), axis=1).mean(axis=1)
    between = np.mean([np.sum((cent[m] - grand) ** 2) for m in frames])
    within = np.mean([np.mean([np.sum((mu[m][dc] - cent[m]) ** 2) for dc in mu[m]]) for m in frames])
    return float(between / within) if within else np.nan
r_full = ratio(D)
one_per = {m: D[m].sort_values("year").drop_duplicates("stem") for m in D}
r_1pf = ratio(one_per)
nclust = {m: {"n": int(len(D[m])), "n_franchise_stems": int(D[m].stem.nunique())} for m in D}
print(f"\nmedium-vs-era ratio: full-window {r_full:.3f} | one-per-franchise {r_1pf:.3f}")
print("franchise stems:", {m: nclust[m]["n_franchise_stems"] for m in nclust}, "of", {m: nclust[m]["n"] for m in nclust})

res = {"adaptation_source_block": adapt, "n_source_clusters": int(len(sources)),
       "medium_era_ratio": {"full_window": round(r_full, 3), "one_per_franchise": round(r_1pf, 3), "clusters": nclust},
       "note": "Lead with effect sizes + spec stability, not p from nominally-independent works. "
               "Franchise stem is a title heuristic (no franchise QIDs pulled)."}
json.dump(res, open(os.path.join(OUT, "A10_dependence.json"), "w"), indent=2)
print("wrote out/A10_dependence.json")
