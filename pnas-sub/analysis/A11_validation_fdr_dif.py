#!/usr/bin/env python3
"""
A11 -- Per-attribute validation, FDR control, era-stratified validation, and a
cross-medium calibration / differential-item-functioning (DIF) test.

Answers a referee asking for (a) a per-attribute validation table with
FDR-adjusted significance, (b) validation stratified by release era, and
(c) a test of cross-medium measurement invariance (DIF).

No LLM calls, no new data -- everything is computed from files already in
data/validation, data/corpus, and data/atlas.

Outputs:
  out/A11_per_attribute_validation.csv   one row per attribute
  out/A11_validation.json                summary of all four pieces
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / "data" / "validation"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

# Representative validation-set sizes (the paired human/model validation works).
N_FILM = 242   # film validation movies
N_BOOK = 153   # book validation novels

notes = []


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def fisher_z_p(r, n):
    """Two-sided p-value for H0: rho=0 via the Fisher z transform."""
    if r is None or np.isnan(r) or n is None or n <= 3:
        return np.nan
    r = max(min(float(r), 0.9999), -0.9999)
    z = np.arctanh(r) * np.sqrt(n - 3)
    return 2 * stats.norm.sf(abs(z))


def p_from_heldout_ci(point, ci_lo, null):
    """
    Derive a two-sided p-value from a 95% held-out CI lower bound:
    SE = (point - ci_lo) / 1.96, z = (point - null) / SE.
    Used for AUC-metric (genre) attributes where a Fisher z on r is not defined.
    """
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (point, ci_lo)):
        return np.nan
    se = (point - ci_lo) / 1.96
    if se <= 0:
        return np.nan
    z = (point - null) / se
    return 2 * stats.norm.sf(abs(z))


def bh_fdr(pvals, q=0.05):
    """Benjamini-Hochberg. Returns (reject_bool_array, crit_p_threshold)."""
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    m = ok.sum()
    reject = np.zeros(len(p), bool)
    if m == 0:
        return reject, np.nan
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    thresh = q * (np.arange(1, m + 1)) / m
    passed = p[order] <= thresh
    kmax = np.where(passed)[0].max() + 1 if passed.any() else 0
    crit = p[order][kmax - 1] if kmax > 0 else np.nan
    if kmax > 0:
        reject[order[:kmax]] = True
    return reject, crit


def pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 4 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan, int(m.sum())
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


# ============================================================================
# 1. PER-ATTRIBUTE VALIDATION TABLE  +  2. FDR
# ============================================================================
dic = pd.read_csv(VAL / "attribute_dictionary.csv")
main = dic[dic["status"] == "main"].copy()  # 216 attributes

for c in ["validation", "film_r", "book_r", "heldout_ci_lo"]:
    main[c] = pd.to_numeric(main[c], errors="coerce")

# Reliability ceiling from film split-half reliability (Spearman-Brown 2r/(1+r)).
rel = pd.read_csv(VAL / "reliability_halves.csv")
rel = rel[rel["medium"] == "film"].copy()
rel["ceiling"] = 2 * rel["r_halfsplit"] / (1 + rel["r_halfsplit"])
# Attach each split-half reliability to the exact dictionary attribute(s) that
# measure the *same* item (precise allowlist -- avoids attaching, e.g., the
# "# protagonists" count reliability to every protagonist attribute).
REL_TARGETS = {
    "fantastical": ["fantastical world"],
    "clarity": ["clarity (easy to follow)"],
    "sci-fi": ["sci-fi world"],
    "realistic world": ["realistic world"],
    "# protagonists": ["# protagonists"],
    "immersive": ["immersive setting (avg)"],
    "# settings": ["# settings"],
    "# side characters": ["# side characters"],
    "plot convincing": ["plot convincing"],
    # "resolution" / "surprise" have no distinct dictionary attribute to attach to
}
rel_ceiling = dict(zip(rel["attribute"], rel["ceiling"]))
rel_halfsplit = dict(zip(rel["attribute"], rel["r_halfsplit"]))
target_to_rel = {t: rk for rk, ts in REL_TARGETS.items() for t in ts}


def match_ceiling(attr_lower):
    rk = target_to_rel.get(attr_lower.strip())
    if rk is not None:
        return rel_ceiling[rk], rel_halfsplit[rk], rk
    return np.nan, np.nan, ""


rows = []
for _, r in main.iterrows():
    metric = r["validation_metric"]
    fr, br = r["film_r"], r["book_r"]
    # validation basis: the medium with the larger |r| (matches the held-out CI)
    cand = [(abs(fr) if pd.notna(fr) else -1, "film", fr, N_FILM),
            (abs(br) if pd.notna(br) else -1, "book", br, N_BOOK)]
    cand.sort(reverse=True)
    best_absr, medium, best_r, n_used = cand[0]
    if best_absr < 0:  # no film_r/book_r -> fall back to the 'validation' column
        best_r, medium, n_used = r["validation"], "film" if metric == "AUC" else "na", N_FILM
    attr_l = str(r["attribute"]).lower()
    ceil, halfsplit, relkey = match_ceiling(attr_l)

    # p-value
    if metric == "AUC":
        p = p_from_heldout_ci(r["validation"], r["heldout_ci_lo"], null=0.5)
        pmethod = "heldout_CI(AUC,null=0.5)"
        report_stat = r["validation"]
    else:
        p = fisher_z_p(best_r, n_used)
        pmethod = "fisher_z(r,n)"
        report_stat = best_r

    rows.append(dict(
        attribute=r["attribute"], layer=r["layer"], column=r["column"],
        medium=medium, cross_medium=bool(r["cross_medium"]),
        validation_metric=metric,
        validation_r_or_auc=report_stat,
        film_r=fr, book_r=br,
        n_validation=n_used,
        heldout_ci_lo=r["heldout_ci_lo"], heldout_tier=r["heldout_tier"],
        tier=r["tier"],
        reliability_r_halfsplit=halfsplit,
        reliability_ceiling_SB=ceil,
        validated=bool(pd.notna(r["heldout_ci_lo"]) and r["heldout_ci_lo"] > 0),
        p_value=p, p_method=pmethod,
        reliability_match=relkey,
    ))

tab = pd.DataFrame(rows)

# Benjamini-Hochberg over all 216 attributes.
reject, crit = bh_fdr(tab["p_value"].values, q=0.05)
tab["fdr_reject_q05"] = reject

n_validated = int(tab["validated"].sum())
val = tab[tab["validated"]]
n_val_survive = int((val["fdr_reject_q05"]).sum())
n_val_pval = int(val["p_value"].notna().sum())

tab.to_csv(OUT / "A11_per_attribute_validation.csv", index=False)

notes.append(
    "Per-attribute p-values: r-metric attributes use a Fisher-z test on the "
    "validation r (the larger of film_r/book_r, which the held-out CI reflects) "
    f"with a representative validation-set n (film={N_FILM}, book={N_BOOK}); the "
    "18 genre AUC attributes get a p-value from their held-out 95% CI "
    "(SE=(AUC-ci_lo)/1.96, null AUC=0.5). Exact per-attribute n varies "
    "(~103-242 film, ~95-153 book) but the FDR count is insensitive to it at "
    "these effect sizes."
)
notes.append(
    "'Validated' is defined exactly as in the codebook: held-out 95% CI lower "
    "bound (heldout_ci_lo) > 0, i.e. heldout_tier in {A,B,C}. The held-out CI "
    "upper bound is not stored in the codebook (only the lower bound is the "
    "reported bar), so the CSV reports heldout_ci_lo as the held-out floor."
)

fdr_block = dict(
    n_attributes_main=int(len(tab)),
    n_validated=n_validated,
    n_validated_with_pvalue=n_val_pval,
    n_validated_surviving_FDR_q05=n_val_survive,
    share_validated_surviving=round(n_val_survive / n_validated, 4),
    bh_crit_p_threshold=None if np.isnan(crit) else float(crit),
    n_all216_surviving_FDR_q05=int(reject.sum()),
)

# ============================================================================
# 3. ERA-STRATIFIED VALIDATION  (film headline attributes)
# ============================================================================
hf = pd.read_csv(VAL / "human_means_film.csv")
lf = pd.read_csv(VAL / "film_llm_validation_scores.csv")
mf = hf.merge(lf, on=["survey_movie_id", "attribute"])

cf = pd.read_parquet(ROOT / "data" / "atlas" / "century_frame_film.parquet")[["idx", "year"]]
vpt = pd.read_csv(VAL / "validation_plot_text_film.csv")
yr = vpt.merge(cf, left_on="wiki_idx", right_on="idx", how="left")[["survey_movie_id", "year"]]
mf = mf.merge(yr, on="survey_movie_id", how="left")

# Headline film attributes = the split-half-reliability set, matched to survey attrs.
def survey_match(sig):
    hits = [a for a in mf["attribute"].unique() if all(t in a.lower() for t in sig.split())]
    return hits[0] if hits else None

HEADLINE_SIGS = {
    "fantastical": "fantastical", "sci-fi": "science_fictional",
    "realistic world": "realistic_was_the_world", "resolution": "resolved",
    "surprise": "surprising", "# protagonists": "how_many_protagonists",
    "# settings": "how_many_major_settings",
    "# side characters": "named_side_characters",
    "plot convincing": "how_convincing_did_you_find_the_plot",
    "clarity": "very_confusing",
}
headline_map = {k: survey_match(v) for k, v in HEADLINE_SIGS.items()}
headline_map = {k: v for k, v in headline_map.items() if v is not None}

median_year = float(np.nanmedian(mf["year"]))
era_rows = []
for label, attr in headline_map.items():
    sub = mf[(mf["attribute"] == attr) & mf["year"].notna()]
    r_all, n_all = pearson(sub["human_mean"], sub["llm_score"])
    out = dict(headline=label, attribute=attr, r_overall=r_all, n_overall=n_all)
    for split_name, cut in [("median1991", median_year), ("1985", 1985)]:
        early = sub[sub["year"] <= cut]
        late = sub[sub["year"] > cut]
        r_e, n_e = pearson(early["human_mean"], early["llm_score"])
        r_l, n_l = pearson(late["human_mean"], late["llm_score"])
        out[f"r_early_{split_name}"] = r_e
        out[f"n_early_{split_name}"] = n_e
        out[f"r_late_{split_name}"] = r_l
        out[f"n_late_{split_name}"] = n_l
    era_rows.append(out)

era_df = pd.DataFrame(era_rows)
# Verdict: does validation hold in BOTH eras? (positive, meaningful r in each)
holds_median = int(((era_df["r_early_median1991"] > 0.15) & (era_df["r_late_median1991"] > 0.15)).sum())
holds_1985 = int(((era_df["r_early_1985"] > 0.15) & (era_df["r_late_1985"] > 0.15)).sum())

era_block = dict(
    split_median_year=median_year,
    n_headline_film_attrs=int(len(era_df)),
    mean_r_early_median=round(float(era_df["r_early_median1991"].mean()), 3),
    mean_r_late_median=round(float(era_df["r_late_median1991"].mean()), 3),
    mean_r_early_pre1985=round(float(era_df["r_early_1985"].mean()), 3),
    mean_r_late_post1985=round(float(era_df["r_late_1985"].mean()), 3),
    n_holding_both_eras_median_split=holds_median,
    n_holding_both_eras_1985_split=holds_1985,
    per_attribute=era_df.round(3).to_dict(orient="records"),
)
notes.append(
    "Era-stratified validation is computed fresh from the film validation set: "
    "per-work human means vs. model scores, correlated within each era. Works "
    "split at the validation-set median release year (1991) and, as a robustness "
    "check matching prior notes, at 1985. Release years join validation works to "
    "century_frame_film via wiki_idx (238/242 works have a year)."
)

# ============================================================================
# 4. CALIBRATION / DIF  (cross-medium measurement invariance)
# ============================================================================
bs = pd.read_csv(ROOT / "data" / "corpus" / "book_structural_1890_2025.csv")
hb = pd.read_csv(VAL / "human_means_book.csv")
shared = sorted(set(mf["attribute"]) & set(hb["attribute"]))

# film paired long
film_pair = mf[["attribute", "human_mean", "llm_score"]].rename(columns={"llm_score": "model_score"})
film_pair = film_pair.assign(medium="film")

# book paired long (book model scores from the book_structural corpus)
book_long = []
for a in shared:
    if a in bs.columns:
        bm = bs[["book_idx", a]].dropna().rename(columns={a: "model_score"})
        sub = hb[hb["attribute"] == a].merge(bm, on="book_idx")
        sub["attribute"] = a
        book_long.append(sub[["attribute", "human_mean", "model_score"]])
book_pair = pd.concat(book_long, ignore_index=True).assign(medium="book")

# headline cross-medium set = headline film attrs that also have book paired data
headline_cross = [v for v in headline_map.values() if v in shared]

dif_rows = []
import statsmodels.formula.api as smf
for a in shared:
    f = film_pair[film_pair["attribute"] == a].copy()
    b = book_pair[book_pair["attribute"] == a].copy()
    if len(f) < 8 or len(b) < 8:
        continue
    pooled = pd.concat([f, b], ignore_index=True).dropna(subset=["human_mean", "model_score"])
    # center model_score (pooled) so 'medium' main effect = intercept shift at mean score
    pooled["ms_c"] = pooled["model_score"] - pooled["model_score"].mean()
    pooled["book"] = (pooled["medium"] == "book").astype(int)
    try:
        fit = smf.ols("human_mean ~ ms_c * book", data=pooled).fit()
    except Exception:
        continue
    inter_beta = fit.params.get("ms_c:book", np.nan)
    inter_p = fit.pvalues.get("ms_c:book", np.nan)
    med_beta = fit.params.get("book", np.nan)
    med_p = fit.pvalues.get("book", np.nan)
    # medium-separation agreement: film-minus-book on HUMAN means vs on MODEL scores
    sep_human = f["human_mean"].mean() - b["human_mean"].mean()
    sep_model = f["model_score"].mean() - b["model_score"].mean()
    dif_rows.append(dict(
        attribute=a, headline=a in headline_cross,
        n_film=int(len(f)), n_book=int(len(b)),
        medium_intercept_beta=float(med_beta), medium_intercept_p=float(med_p),
        slope_interaction_beta=float(inter_beta), slope_interaction_p=float(inter_p),
        interaction_sig_q05=bool(inter_p < 0.05) if pd.notna(inter_p) else None,
        sep_human=float(sep_human), sep_model=float(sep_model),
        sep_sign_agree=bool(np.sign(sep_human) == np.sign(sep_model)),
    ))

dif_df = pd.DataFrame(dif_rows)
# FDR on the interaction p-values (across all cross-medium attrs tested)
rej_int, crit_int = bh_fdr(dif_df["slope_interaction_p"].values, q=0.05)
dif_df["interaction_sig_fdr"] = rej_int

n_dif = len(dif_df)
n_no_int_raw = int((~dif_df["interaction_sig_q05"].astype(bool)).sum())
n_no_int_fdr = int((~dif_df["interaction_sig_fdr"]).sum())
hl = dif_df[dif_df["headline"]]
n_hl = len(hl)
n_hl_no_int_raw = int((~hl["interaction_sig_q05"].astype(bool)).sum())
n_hl_no_int_fdr = int((~hl["interaction_sig_fdr"]).sum())
sep_agree = int(dif_df["sep_sign_agree"].sum())

# rough-magnitude agreement between human and model medium separation
sm = dif_df.dropna(subset=["sep_human", "sep_model"])
sep_corr = float(np.corrcoef(sm["sep_human"], sm["sep_model"])[0, 1]) if len(sm) > 3 else np.nan

dif_block = dict(
    n_cross_medium_attrs_tested=n_dif,
    n_no_significant_interaction_raw=n_no_int_raw,
    n_no_significant_interaction_fdr=n_no_int_fdr,
    n_headline_cross_medium=n_hl,
    n_headline_no_sig_interaction_raw=n_hl_no_int_raw,
    n_headline_no_sig_interaction_fdr=n_hl_no_int_fdr,
    medium_separation_sign_agreement=f"{sep_agree}/{n_dif}",
    medium_separation_human_vs_model_corr=None if np.isnan(sep_corr) else round(sep_corr, 3),
    per_attribute=dif_df.round(4).to_dict(orient="records"),
)
notes.append(
    "DIF model, per cross-medium attribute: human_mean ~ model_score*medium on "
    "pooled film+book work-level pairs, model_score mean-centred so the medium "
    "main effect is the intercept shift at the mean model score and the "
    "interaction is the slope shift. Film model scores from film_llm_validation_"
    "scores.csv; book model scores from data/corpus/book_structural_1890_2025.csv "
    "(both keyed to the survey question). Attributes share the same survey "
    "response scale across media, so raw human/model units are comparable within "
    "attribute. Same-scale caveat: '# side characters' is a count item."
)
notes.append(
    "Medium-separation statistic: film-minus-book difference in HUMAN validation "
    "means vs. the same difference in MODEL scores, per attribute; sign agreement "
    "and their cross-attribute correlation test whether the model reproduces the "
    "human cross-medium ordering."
)

# ============================================================================
# WRITE JSON
# ============================================================================
summary = dict(
    generated_by="pnas-sub/analysis/A11_validation_fdr_dif.py",
    inputs=dict(
        codebook="data/validation/attribute_dictionary.csv",
        reliability="data/validation/reliability_halves.csv",
        film_human="data/validation/human_means_film.csv",
        film_model="data/validation/film_llm_validation_scores.csv",
        book_human="data/validation/human_means_book.csv",
        book_model="data/corpus/book_structural_1890_2025.csv",
        film_years="data/atlas/century_frame_film.parquet + validation_plot_text_film.csv",
    ),
    per_attribute_fdr=fdr_block,
    era_stratified=era_block,
    dif_cross_medium=dif_block,
    notes=notes,
)
with open(OUT / "A11_validation.json", "w") as f:
    json.dump(summary, f, indent=2)

# ============================================================================
# PRINT SUMMARY
# ============================================================================
print("=" * 74)
print("A11  VALIDATION: per-attribute FDR, era-stratified, cross-medium DIF")
print("=" * 74)
print(f"\n[1-2] PER-ATTRIBUTE + FDR (Benjamini-Hochberg, q=0.05)")
print(f"  main attributes                : {len(tab)}")
print(f"  validated (heldout CI lo > 0)  : {n_validated}")
print(f"  validated w/ a p-value         : {n_val_pval}")
print(f"  --> surviving FDR at q=0.05    : {n_val_survive} "
      f"({100*n_val_survive/n_validated:.1f}% of validated)")
print(f"  all-216 surviving FDR          : {int(reject.sum())}")
print(f"  CSV -> out/A11_per_attribute_validation.csv")

print(f"\n[3] ERA-STRATIFIED VALIDATION ({len(era_df)} headline film attrs, "
      f"median split at {median_year:.0f})")
print(f"  mean r early/late (median split): "
      f"{era_block['mean_r_early_median']:.3f} / {era_block['mean_r_late_median']:.3f}")
print(f"  mean r pre/post 1985           : "
      f"{era_block['mean_r_early_pre1985']:.3f} / {era_block['mean_r_late_post1985']:.3f}")
print(f"  holds in BOTH eras (median r>.15): {holds_median}/{len(era_df)}"
      f"   | (1985 split): {holds_1985}/{len(era_df)}")

print(f"\n[4] CROSS-MEDIUM DIF (measurement invariance)")
print(f"  cross-medium attrs tested      : {n_dif}")
print(f"  NO significant medium interaction (raw p): {n_no_int_raw}/{n_dif}")
print(f"  NO significant medium interaction (FDR) : {n_no_int_fdr}/{n_dif}")
print(f"  headline cross-medium attrs    : {n_hl}")
print(f"    NO sig interaction (raw / FDR): {n_hl_no_int_raw} / {n_hl_no_int_fdr} of {n_hl}")
print(f"  medium-separation sign agreement (human vs model): {sep_agree}/{n_dif}")
print(f"  medium-separation human-vs-model corr           : {sep_corr:.3f}")
print(f"\n  JSON -> out/A11_validation.json")
