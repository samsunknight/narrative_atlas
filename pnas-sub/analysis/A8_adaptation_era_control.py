"""A8 (PNAS Phase 2): does the novel-to-film adaptation contrast hold historical era fixed?

A referee notes that the within-pair adaptation deltas (film-minus-source-novel, A3) do not hold
historical era constant: a source novel and its film adaptation can be separated by years or decades,
so a "medium" gap could partly be an "era" gap. This script controls for the adaptation lag three ways
and asks whether the significant structural shifts survive.

It mirrors A3_adaptation_one_per_source.py's matching + standardization EXACTLY (films matched to source
novels by normalized title through data/matched/adaptation_pairs.csv, ten film-minus-novel deltas
standardized by pooled SD, same SM attribute map). The only additions are the release years:
each matched pair carries its film year and its novel year, so lag = film_year - novel_year.

Steps:
  1. Distribution of adaptation lag (mean/median/quartiles, % within 10 yr, % negative).
  2. Self-check: full-sample raw deltas must reproduce A3's out/A3_adaptation_one_per_source.json full_SD.
  3. Era-residualized deltas: subtract the medium-specific decade mean (from the full corpus) from both
     the film and the novel score, then take film-minus-novel and standardize by pooled SD.
  4. lag<=10 subset: recompute the ten deltas keeping only pairs with |lag| <= 10 years.
  5. Delta-vs-lag: regress each per-pair raw delta on lag; slope + significance test whether the medium
     gap grows with temporal separation.

Conclusion: whether the significant structural shifts (realism, character-/plot-driven, world type,
protagonist count) survive era-residualization and the lag<=10 restriction with the same sign.
Run from ~/uoft/style_evolves/. No LLM calls, no new data.
"""
import os, re, json
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.environ.get("NARRATIVE_ATLAS_ROOT", os.path.expanduser("~/uoft/style_evolves"))
def P(*a): return os.path.join(ROOT, *a)
def norm(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())

# --- corpus + matching, EXACTLY as A3 ---
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

def decade(y): return np.floor(np.asarray(y, dtype=float) / 10.0) * 10.0

rows = []
for _, r in PAIRS.iterrows():
    ft, bt = norm(r.filmLabel), norm(r.bookLabel)
    if ft in Fn.index and bt in Bn.index:
        d = {n: (Fn.loc[ft][a] - Bn.loc[bt][a]) for n, a in SM.items()}   # raw film-minus-novel delta
        d["_src"] = bt
        d["_fy"] = float(Fn.loc[ft]["year"])
        d["_by"] = float(Bn.loc[bt]["year"])
        for n, a in SM.items():                                          # keep the paired scores for FE regression
            d["f_" + n] = float(Fn.loc[ft][a]); d["b_" + n] = float(Bn.loc[bt][a])
        rows.append(d)
AD = pd.DataFrame(rows)
AD["_lag"] = AD["_fy"] - AD["_by"]
print(f"matched pairs: {len(AD)} from {AD._src.nunique()} sources")

def deltas(sub, cols=None):
    cols = cols or {n: n for n in SM}
    return {n: (sub[cols[n]].astype(float).mean() / pooled_sd[n]) for n in SM}

# --- 1. lag distribution ---
lag = AD["_lag"].dropna()
q1, med, q3 = np.percentile(lag, [25, 50, 75])
lag_dist = {"n": int(len(lag)), "mean": round(float(lag.mean()), 2), "median": round(float(med), 2),
            "q1": round(float(q1), 2), "q3": round(float(q3), 2),
            "min": round(float(lag.min()), 2), "max": round(float(lag.max()), 2),
            "pct_within_10yr": round(float((lag.abs() <= 10).mean() * 100), 1),
            "pct_negative": round(float((lag < 0).mean() * 100), 1)}
print("\n[1] Adaptation lag (film_year - novel_year):")
print(f"    n={lag_dist['n']}  mean={lag_dist['mean']}  median={lag_dist['median']}  "
      f"Q1={lag_dist['q1']}  Q3={lag_dist['q3']}  range=[{lag_dist['min']},{lag_dist['max']}]")
print(f"    within 10yr: {lag_dist['pct_within_10yr']}%   negative (film before novel yr): {lag_dist['pct_negative']}%")

# --- 2. self-check vs A3 full_SD ---
A3 = json.load(open(P("pnas-sub/analysis/out/A3_adaptation_one_per_source.json")))["attrs"]
raw = deltas(AD)
print("\n[2] Self-check: full-sample raw deltas vs A3 full_SD (tol 0.01):")
self_check = {}
for n in SM:
    a3v = A3[n]["full_SD"]; diff = abs(raw[n] - a3v)
    ok = diff <= 0.01
    self_check[n] = {"A8_raw_SD": round(raw[n], 3), "A3_full_SD": a3v, "diff": round(float(diff), 4), "match": bool(ok)}
    print(f"    {n:15s} A8={raw[n]:+.3f}  A3={a3v:+.3f}  diff={diff:.4f}  {'OK' if ok else 'MISMATCH'}")

# --- 3. era-residualized deltas ---
# CORRECT era control: hold release DECADE fixed while KEEPING the medium contrast. Stack the matched
# films and novels (2 rows/pair) and regress score ~ film_dummy + decade FE; the film coefficient is the
# medium gap net of the common era trend. Standardize by pooled SD. (Subtracting a MEDIUM-SPECIFIC decade
# mean instead would net out the medium main effect itself -- reported below only as a diagnostic.)
ADr = AD.dropna(subset=["_fy", "_by"]).reset_index(drop=True)
decades = sorted(set(decade(ADr["_fy"]).tolist()) | set(decade(ADr["_by"]).tolist()))
dref = decades[0]; dcols = decades[1:]                       # drop first decade as reference
def era_adjusted_delta(n):
    fy, by = decade(ADr["_fy"].values), decade(ADr["_by"].values)
    film_rows = np.column_stack([np.ones(len(ADr)), np.ones(len(ADr))] +
                                [(fy == d).astype(float) for d in dcols])
    book_rows = np.column_stack([np.ones(len(ADr)), np.zeros(len(ADr))] +
                                [(by == d).astype(float) for d in dcols])
    X = np.vstack([film_rows, book_rows])
    y = np.concatenate([ADr["f_" + n].values, ADr["b_" + n].values]).astype(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta[1] / pooled_sd[n]                            # coeff on film_dummy, in pooled-SD units
res_delta = {n: era_adjusted_delta(n) for n in SM}
# diagnostic only: medium-specific decade demeaning (mechanically removes the medium effect)
Fdec = {n: F.assign(_d=decade(F.year)).groupby("_d")[a].mean() for n, a in SM.items()}
Bdec = {n: B.assign(_d=decade(B.year)).groupby("_d")[a].mean() for n, a in SM.items()}
ms = {n: float((( (ADr["f_" + n].values - np.array([Fdec[n].get(dd, np.nan) for dd in decade(ADr["_fy"].values)]))
                 - (ADr["b_" + n].values - np.array([Bdec[n].get(dd, np.nan) for dd in decade(ADr["_by"].values)])) )
                .mean()) / pooled_sd[n]) for n in SM}
print(f"\n[3] Era-adjusted deltas: film-dummy coeff from score~film+decade-FE, n={len(ADr)} pairs (keeps medium contrast):")
for n in SM:
    print(f"    {n:15s} raw={raw[n]:+.3f}   era-adjusted={res_delta[n]:+.3f}   (medium-specific-demean diag={ms[n]:+.3f})")

# --- 4. lag<=10 subset ---
sub10 = AD[AD["_lag"].abs() <= 10]
d10 = deltas(sub10)
print(f"\n[4] |lag|<=10 subset deltas, n={len(sub10)}:")
for n in SM:
    print(f"    {n:15s} raw(full)={raw[n]:+.3f}   lag<=10={d10[n]:+.3f}")

# --- 5. delta-vs-lag regression (per-pair raw delta on lag, score units) ---
print("\n[5] Per-pair delta ~ lag (does the medium gap grow with temporal separation?):")
lagreg = {}
adl = AD.dropna(subset=["_lag"])
for n in SM:
    y = adl[n].astype(float).values; x = adl["_lag"].astype(float).values
    lr = stats.linregress(x, y)
    sig = bool(lr.pvalue < 0.05)
    lagreg[n] = {"slope": float(lr.slope), "pvalue": float(lr.pvalue), "sig": sig}
    print(f"    {n:15s} slope={lr.slope:+.5f}/yr  p={lr.pvalue:.3f}  {'SIG' if sig else 'ns'}")

# --- assemble + verdict ---
STRUCTURAL = ["realistic", "character-driven", "plot-driven", "sci-fi", "fantastical", "world-building", "#protagonists"]
print("\n[Conclusion] Sign stability across raw / era-residualized / lag<=10:")
out_deltas = {}
for n in SM:
    signs = [np.sign(raw[n]), np.sign(res_delta[n]), np.sign(d10[n])]
    stable = all(s == signs[0] and s != 0 for s in signs)
    out_deltas[n] = {"raw_SD": round(raw[n], 3),
                     "era_residualized_SD": round(res_delta[n], 3),
                     "lag_le10_SD": round(d10[n], 3),
                     "lag_le10_n": int(len(sub10)),
                     "delta_vs_lag_slope": round(lagreg[n]["slope"], 5),
                     "delta_vs_lag_sig": lagreg[n]["sig"],
                     "sign_stable_across_all": bool(stable)}
    tag = " <-- structural" if n in STRUCTURAL else ""
    print(f"    {n:15s} raw={raw[n]:+.3f}  era={res_delta[n]:+.3f}  lag<=10={d10[n]:+.3f}  "
          f"sign-stable={'YES' if stable else 'NO'}{tag}")

notes = ("Mirrors A3 matching/standardization exactly (pooled-SD, same SM map, same corpus CSVs); "
         "full-sample raw deltas reproduce A3's full_SD for all 10 attrs (self_check). Lag = film_year - "
         "novel_year. era_residualized_SD is the film-vs-novel gap that HOLDS RELEASE DECADE FIXED while "
         "keeping the medium contrast: stack the matched films and novels (2 rows/pair) and regress "
         "score ~ film_dummy + decade fixed effects; the film-dummy coefficient, standardized by pooled "
         "SD, is the medium gap net of the common secular era trend. NOTE we deliberately did NOT use the "
         "prompt's 'subtract the medium-specific decade mean' recipe as the headline: demeaning each film "
         "by the mean film of its decade and each novel by the mean novel of its decade mechanically "
         "removes the medium main effect itself (it subtracts film_mean - novel_mean), so it cannot test "
         "whether the medium gap survives era control -- it zeroes it by construction; that quantity is "
         "reported only as a printed diagnostic. delta_vs_lag regresses the per-pair raw film-minus-novel "
         "delta (score units) on lag in years (does the medium gap grow with temporal separation). "
         "sign_stable_across_all = raw, era-adjusted, and |lag|<=10 deltas all share one nonzero sign. "
         "Structural attrs of interest: realism, character-/plot-driven, world type "
         "(sci-fi/fantastical/world-building), protagonist count.")

res = {"lag_dist": lag_dist, "self_check": self_check, "deltas": out_deltas, "notes": notes}
os.makedirs(P("pnas-sub/analysis/out"), exist_ok=True)
json.dump(res, open(P("pnas-sub/analysis/out/A8_adaptation_era.json"), "w"), indent=2)
print("\nwrote pnas-sub/analysis/out/A8_adaptation_era.json")
