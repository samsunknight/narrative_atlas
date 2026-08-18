"""A16 (PNAS rebuild, Phase 4 / S4a): the high-confidence direct-adaptation subset.

The source-linked adaptation contrast (A3/A8) pools every Wikidata "based on" film-novel link, which
includes films built from multiple sources or loosely "based on characters". A cleaner subset keeps
only films with a SINGLE book source in the matched set --- a proxy for a direct one-novel adaptation,
excluding composite and franchise-style links (the adaptation_pairs table carries no relation-type
qualifier, so single-source is the available high-confidence filter; noted as a limitation). We
recompute the ten film-minus-novel deltas on the full set, the single-source high-confidence set, and
the short-lag set, and check that the headline shifts hold. Reuses A8's matching EXACTLY.

Run from the style_evolves root. Writes analysis/out/A16_adaptation_hiconf.json.
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
    for k, n in [("science_fictional", "sci-fi"), ("fantastical", "fantastical"), ("realistic_was_the_world", "realistic"),
                 ("world_building", "world-building"), ("how_many_protagonists", "#protagonists"), ("competent", "competence"),
                 ("proactive", "proactiveness"), ("relatable", "relatability"), ("plot_driven", "plot-driven"),
                 ("character_driven", "character-driven")]:
        if k in c: SM[n] = c
PAIRS = pd.read_csv(P("data/matched/adaptation_pairs.csv")).dropna(subset=["filmLabel", "bookLabel"])
Fn = F.assign(nt=F.title.map(norm)).drop_duplicates("nt").set_index("nt")
Bn = B.assign(nt=B.title.map(norm)).drop_duplicates("nt").set_index("nt")
pooled_sd = {n: pd.concat([F[a], B[a]]).astype(float).std() for n, a in SM.items()}

rows = []
for _, r in PAIRS.iterrows():
    ft, bt = norm(r.filmLabel), norm(r.bookLabel)
    if ft in Fn.index and bt in Bn.index:
        d = {n: (Fn.loc[ft][a] - Bn.loc[bt][a]) for n, a in SM.items()}
        d["_film"], d["_src"] = ft, bt
        d["_lag"] = float(Fn.loc[ft]["year"]) - float(Bn.loc[bt]["year"])
        rows.append(d)
AD = pd.DataFrame(rows)
# high-confidence: a film matched to exactly ONE book source (single-source, non-composite)
film_count = AD["_film"].value_counts()
AD["single_source"] = AD["_film"].map(film_count) == 1
print(f"matched pairs: {len(AD)} | single-source (hi-conf): {int(AD.single_source.sum())} "
      f"| multi-source dropped: {int((~AD.single_source).sum())}")

def deltas(sub):
    return {n: round(float(sub[n].astype(float).mean() / pooled_sd[n]), 3) for n in SM}
def boot_ci(sub, n_attr, reps=1000):
    vals = sub[n_attr].astype(float).values / pooled_sd[n_attr]
    rng = np.random.RandomState(0)
    bs = [vals[rng.randint(0, len(vals), len(vals))].mean() for _ in range(reps)]
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]

subsets = {"full": AD, "hi_confidence_single_source": AD[AD.single_source],
           "short_lag_le10": AD[AD["_lag"].abs() <= 10]}
res = {"n_pairs": len(AD), "n_single_source": int(AD.single_source.sum()),
       "note": "single-source = film matched to exactly one book source; proxy for direct adaptation "
               "(no relation-type qualifier in adaptation_pairs). Deltas in pooled SD.", "subsets": {}}
HEADLINE = ["realistic", "plot-driven", "character-driven", "sci-fi", "fantastical", "#protagonists"]
for name, sub in subsets.items():
    dd = deltas(sub)
    res["subsets"][name] = {"n": int(len(sub)), "deltas": dd,
                            "headline_ci": {n: boot_ci(sub, n) for n in HEADLINE}}
    print(f"\n=== {name} (n={len(sub)}) ===")
    for n in HEADLINE:
        ci = res["subsets"][name]["headline_ci"][n]
        star = "*" if (ci[0] > 0 or ci[1] < 0) else " "
        print(f"  {n:15s} {dd[n]:+.3f}  CI[{ci[0]:+.2f},{ci[1]:+.2f}] {star}")

# decision: do the headline shifts hold (same sign, exclude 0) in the hi-confidence subset?
hc = res["subsets"]["hi_confidence_single_source"]
fu = res["subsets"]["full"]
hold = {n: (np.sign(hc["deltas"][n]) == np.sign(fu["deltas"][n]) and
            (hc["headline_ci"][n][0] > 0 or hc["headline_ci"][n][1] < 0)) for n in HEADLINE}
res["headline_holds_in_hiconf"] = hold
print("\nheadline shifts hold in hi-confidence subset:", hold)
json.dump(res, open(P("pnas-sub/analysis/out/A16_adaptation_hiconf.json"), "w"), indent=2)
print("wrote out/A16_adaptation_hiconf.json")
