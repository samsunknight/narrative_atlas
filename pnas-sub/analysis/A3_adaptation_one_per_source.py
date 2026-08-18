"""A3 (PNAS Phase 2): adaptation robustness to multiply-adapted sources.

The within-pair adaptation signature already has source-clustered (CR1) SEs, Holm adjustment, and a
medium-label-swap check. This adds the reviewer-facing one-per-source check: some novels have several
film adaptations, so re-run the ten-attribute film-minus-novel deltas (a) on the full 437 pairs, (b)
keeping only the EARLIEST adaptation per source (316 sources), and (c) over 500 draws each keeping ONE
RANDOM adaptation per source, to show the sign pattern and the significant shifts are not driven by a
few multiply-adapted franchises. Replicates regen_adaptation_deltas.py's matching + SD standardization
(self-check: full-sample deltas must match the shipped adaptation_deltas.csv). Run from ~/uoft/style_evolves/.
"""
import os, re, json
import numpy as np, pandas as pd

ROOT = os.environ.get("NARRATIVE_ATLAS_ROOT", os.path.expanduser("~/uoft/style_evolves"))
def P(*a): return os.path.join(ROOT, *a)
def norm(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())

F = pd.read_csv(P("data/corpus/film_structural_1890_2025.csv")).rename(columns={"film_idx": "id"})
B = pd.read_csv(P("data/corpus/book_structural_1890_2025.csv")).rename(columns={"book_idx": "id"})
ATTRS = [c for c in F.columns if c not in ("id", "title", "year")]
SM = {}
for c in ATTRS:
    for k, n in [("science_fictional","sci-fi"),("fantastical","fantastical"),("realistic_was_the_world","realistic"),
                 ("world_building","world-building"),("how_many_protagonists","#protagonists"),("competent","competence"),
                 ("proactive","proactiveness"),("relatable","relatability"),("plot_driven","plot-driven"),("character_driven","character-driven")]:
        if k in c: SM[n] = c
PAIRS = pd.read_csv(P("data/matched/adaptation_pairs.csv")).dropna(subset=["filmLabel", "bookLabel"])
Fn = F.assign(nt=F.title.map(norm)).drop_duplicates("nt").set_index("nt")
Bn = B.assign(nt=B.title.map(norm)).drop_duplicates("nt").set_index("nt")
pooled_sd = {n: pd.concat([F[a], B[a]]).astype(float).std() for n, a in SM.items()}
xs = {n: (F[a].mean() - B[a].mean()) for n, a in SM.items()}     # cross-sectional difference (sign reference)

rows = []
for _, r in PAIRS.iterrows():
    ft, bt = norm(r.filmLabel), norm(r.bookLabel)
    if ft in Fn.index and bt in Bn.index:
        d = {n: (Fn.loc[ft][a] - Bn.loc[bt][a]) for n, a in SM.items()}
        d["_src"] = bt; d["_fy"] = Fn.loc[ft]["year"]
        rows.append(d)
AD = pd.DataFrame(rows)
print(f"matched pairs: {len(AD)} from {AD._src.nunique()} sources")

def deltas(sub):
    return {n: (sub[n].astype(float).mean() / pooled_sd[n]) for n in SM}
def sign_agree(dd):
    return sum(np.sign(dd[n]) == np.sign(xs[n]) for n in SM)

full = deltas(AD)
earliest = deltas(AD.sort_values("_fy").drop_duplicates("_src", keep="first"))
print(f"\nfull-sample sign agreement: {sign_agree(full)}/10   earliest-per-source: {sign_agree(earliest)}/10")

# random-one-per-source bootstrap
rng = np.random.RandomState(0); REPS = 500
bd = {n: [] for n in SM}; bsign = []
for _ in range(REPS):
    one = AD.groupby("_src", group_keys=False).apply(lambda g: g.iloc[[rng.randint(len(g))]])
    dd = deltas(one); bsign.append(sign_agree(dd))
    for n in SM: bd[n].append(dd[n])

print("\nattr            full   earliest   random-one 95% CI      sign-stable")
res = {"n_pairs": int(len(AD)), "n_sources": int(AD._src.nunique()), "reps": REPS, "attrs": {}}
for n in SM:
    lo, hi = np.percentile(bd[n], [2.5, 97.5])
    stable = (lo > 0) == (hi > 0)          # CI excludes 0 -> sign is stable across random-one draws
    print(f"{n:15s} {full[n]:+.2f}   {earliest[n]:+.2f}     [{lo:+.2f},{hi:+.2f}]   {'yes' if stable else 'CROSSES 0'}")
    res["attrs"][n] = {"full_SD": round(full[n], 3), "earliest_SD": round(earliest[n], 3),
                       "random_one_CI": [round(float(lo), 3), round(float(hi), 3)], "sign_stable": bool(stable)}
res["sign_agree_full"] = int(sign_agree(full)); res["sign_agree_earliest"] = int(sign_agree(earliest))
res["sign_agree_random_min"] = int(min(bsign)); res["sign_agree_random_median"] = int(np.median(bsign))
print(f"\nrandom-one sign agreement: median {int(np.median(bsign))}/10, min {min(bsign)}/10")
os.makedirs(P("pnas-sub/analysis/out"), exist_ok=True)
json.dump(res, open(P("pnas-sub/analysis/out/A3_adaptation_one_per_source.json"), "w"), indent=2)
print("wrote out/A3_adaptation_one_per_source.json")
