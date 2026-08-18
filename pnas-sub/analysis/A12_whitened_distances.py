"""A12 (PNAS Phase 2): covariance-aware (whitened / Mahalanobis) recomputation of the
convergence contrasts (A4) and the medium-vs-era variance ratio (A2), answering referee major
comments 8 & 12 (Euclidean distance is sensitive to attribute redundancy; the model couples
attributes ~1.4x more tightly than humans, so redundancy could inflate mood's apparent convergence
and the medium-over-era geometry).

Mirrors A4_convergence_contrasts.py and A2_variance_ratio_inference.py EXACTLY for the Euclidean
baselines, then re-derives every distance under a Mahalanobis metric d(a,b) = sqrt((a-b)' S^-1 (a-b)):
  TASK 1  self-check: A4 novel-tv mood-minus-genre (full) ~= 0.31 and A2 ratio ~= 1.66.
  TASK 2  Mahalanobis convergence: S = pooled within-cell covariance of that layer's percentile-
          ranked attributes (all 3 media, all decades); recompute convergence + mood-minus-{genre,arc}.
  TASK 3  Mahalanobis geometry: whiten the medium-vs-era ratio by S^-1, S = pooled covariance of the
          16 standardized structural attributes.
  TASK 4  human-covariance variant where feasible (human inter-attribute covariance on validation
          works), else report only the model-covariance whitening honestly.
Bounded, single-threaded. No LLM, no new data. Writes out/A12_whitened.json. Run from ~/uoft/style_evolves/.
"""
import os, sys, json
import numpy as np, pandas as pd

ROOT = os.environ.get("NARRATIVE_ATLAS_ROOT", os.path.expanduser("~/uoft/style_evolves")); os.chdir(ROOT)  # canonical root
RIDGE = 1e-3  # regularizer added to the diagonal of every covariance before inversion


def whiten_inv(S):
    """S^-1 with a ridge on the diagonal for stability (pseudo-inverse fallback if still singular)."""
    S = np.asarray(S, dtype=float)
    Sr = S + RIDGE * np.eye(S.shape[0])
    try:
        return np.linalg.inv(Sr)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(Sr)


def maha2(v, Sinv):
    """squared Mahalanobis length of vector v under Sinv (skips NaN dims defensively)."""
    v = np.asarray(v, dtype=float)
    m = ~np.isnan(v)
    if not m.all():
        v = v[m]; Sinv = Sinv[np.ix_(m, m)]
    return float(v @ Sinv @ v)


# ======================================================================================
# PART A — CONVERGENCE (mirror A4 exactly, then add the whitened metric)
# ======================================================================================
man = pd.read_csv("data/validation/rescore_manifest.csv"); man = man[man.deploy == True]
LAY = man.set_index("attr_id")["layer"].to_dict()
FR = {m: pd.read_parquet(f"data/atlas/century_frame_{m}.parquet") for m in ["film", "book", "tv"]}
LAYERS = ["mood", "genre", "arc"]
PAIRS = {"novel-tv": ("book", "tv"), "film-tv": ("film", "tv")}
DECS = list(range(1950, 2011, 10))
MIN = 30

def attrs(df, L): return [c for c in df.columns if LAY.get(c) == L]
SH = {L: [c for c in attrs(FR["film"], L) if c in FR["book"].columns and c in FR["tv"].columns] for L in LAYERS}
PZ = {m: {L: FR[m][SH[L]].rank(pct=True).values for L in LAYERS} for m in FR}
YR = {m: FR[m]["year"].values for m in FR}
CELL = {m: {d: np.where((YR[m] >= d) & (YR[m] < d + 10))[0] for d in DECS} for m in FR}

# ---- pooled within-cell covariance per layer: pool all 3 media, all decades ----
# residual = row minus its (medium x decade) cell centroid; pool residuals; pairwise-complete cov.
SINV_MODEL = {}
for L in LAYERS:
    resid = []
    for m in FR:
        for d in DECS:
            idx = CELL[m][d]
            if len(idx) >= 2:
                block = PZ[m][L][idx]
                resid.append(block - np.nanmean(block, axis=0))
    R = np.vstack(resid)
    S = pd.DataFrame(R).cov().values  # pairwise-complete-obs covariance of percentile ranks
    SINV_MODEL[L] = whiten_inv(S)

def dist_euc(a, b, L, d, idx):
    ia, ib = idx[a][d], idx[b][d]
    if len(ia) < MIN or len(ib) < MIN: return np.nan
    ca = np.nanmean(PZ[a][L][ia], axis=0); cb = np.nanmean(PZ[b][L][ib], axis=0)
    return np.linalg.norm(ca - cb)

def dist_maha(a, b, L, d, idx):
    ia, ib = idx[a][d], idx[b][d]
    if len(ia) < MIN or len(ib) < MIN: return np.nan
    ca = np.nanmean(PZ[a][L][ia], axis=0); cb = np.nanmean(PZ[b][L][ib], axis=0)
    return np.sqrt(maha2(ca - cb, SINV_MODEL[L]))

def all_stats(idx, distfn):
    out = {}
    for pn, (a, b) in PAIRS.items():
        for L in LAYERS:
            d50, d90, d10 = distfn(a, b, L, 1950, idx), distfn(a, b, L, 1990, idx), distfn(a, b, L, 2010, idx)
            out[(pn, L)] = {"d1950": d50, "d1990": d90, "d2010": d10,
                            "conv_trough": d50 - d90, "conv_full": d50 - d10}
    return out

pt_euc = all_stats(CELL, dist_euc)
pt_maha = all_stats(CELL, dist_maha)

rng = np.random.RandomState(0); REPS = 500
def resample():
    return {m: {d: (rng.choice(CELL[m][d], len(CELL[m][d]), replace=True) if len(CELL[m][d]) else CELL[m][d])
                for d in DECS} for m in FR}
boot_samples = [resample() for _ in range(REPS)]
boot_euc = [all_stats(s, dist_euc) for s in boot_samples]
boot_maha = [all_stats(s, dist_maha) for s in boot_samples]

def ci(vals):
    v = np.array([x for x in vals if not np.isnan(x)])
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] if len(v) else [None, None]

def conv_block(pt, boot):
    bp = {}
    for pn, (a, b) in PAIRS.items():
        for L in LAYERS:
            s = pt[(pn, L)]
            bp[f"{pn}|{L}"] = {
                "d1950": round(s["d1950"], 3), "d1990": round(s["d1990"], 3), "d2010": round(s["d2010"], 3),
                "conv_trough": round(s["conv_trough"], 3), "conv_trough_CI": ci([bt[(pn, L)]["conv_trough"] for bt in boot]),
                "conv_full": round(s["conv_full"], 3), "conv_full_CI": ci([bt[(pn, L)]["conv_full"] for bt in boot])}
    ct = {}
    for pn in PAIRS:
        for other in ["genre", "arc"]:
            for horizon, key in [("trough", "conv_trough"), ("full", "conv_full")]:
                diff_pt = pt[(pn, "mood")][key] - pt[(pn, other)][key]
                diff_ci = ci([bt[(pn, "mood")][key] - bt[(pn, other)][key] for bt in boot])
                sig = diff_ci[0] is not None and (diff_ci[0] > 0 or diff_ci[1] < 0)
                ct[f"{pn}|mood-minus-{other}|{horizon}"] = {"diff": round(diff_pt, 3), "CI": diff_ci, "excludes0": bool(sig)}
    return bp, ct

euc_by, euc_contrasts = conv_block(pt_euc, boot_euc)
maha_by, maha_contrasts = conv_block(pt_maha, boot_maha)

# ======================================================================================
# PART B — GEOMETRY (mirror A2 exactly, then whiten the ratio)
# ======================================================================================
AROOT = os.environ.get("NARRATIVE_ATLAS_ROOT", os.path.expanduser("~/uoft/style_evolves"))
for cd in [os.path.join(os.path.dirname(AROOT), "code"), os.path.join(AROOT, "code")]:
    if os.path.isdir(cd) and cd not in sys.path: sys.path.insert(0, cd)
def P(*a): return os.path.join(AROOT, *a)

Bc = pd.read_csv(P("data/corpus/book_structural_1890_2025.csv")).rename(columns={"book_idx": "id"})
Fc = pd.read_csv(P("data/corpus/film_structural_1890_2025.csv")).rename(columns={"film_idx": "id"})
Tc = pd.read_csv(P("data/corpus/tv_structural_1890_2025.csv")).rename(columns={"tv_idx": "id"})
from _methods import VAL16_KEYS as VALCM_KEYS
VALCM = [c for c in Bc.columns if any(k in c.lower() for k in VALCM_KEYS) and c in Tc.columns and c in Fc.columns]
assert len(VALCM) == 16, f"expected 16-attr basis, got {len(VALCM)}"
GDECS = list(range(1910, 2021, 10))
GMIN = 30

mu = pd.concat([Bc[VALCM], Tc[VALCM], Fc[VALCM]]).mean()
sd = pd.concat([Bc[VALCM], Tc[VALCM], Fc[VALCM]]).std().replace(0, 1)

def prep(df, medium):
    d = df[(df.year >= 1915) & (df.year <= 2020)].copy()
    d["dec"] = (d["year"] // 10) * 10; d["m"] = medium
    return d[["m", "dec"] + VALCM]

Dg = pd.concat([prep(Bc, "bk"), prep(Fc, "fm"), prep(Tc, "tv")], ignore_index=True)
Zg = ((Dg[VALCM] - mu) / sd)
mvec = Dg["m"].values; dvec = Dg["dec"].values

# S = pooled covariance of the 16 standardized structural attributes (pairwise complete obs)
S_STRUCT = Zg.cov().values
SINV_STRUCT = whiten_inv(S_STRUCT)
Zgv = Zg.values

def ratio_euclid(zmat, m_lab, d_lab, media):
    C = {m: {} for m in media}
    for m in media:
        for d in GDECS:
            sel = (m_lab == m) & (d_lab == d)
            if sel.sum() >= GMIN: C[m][d] = np.nanmean(zmat[sel], axis=0)
    C = {m: v for m, v in C.items() if v}
    if len(C) < 2: return np.nan
    medmean = {m: np.mean(list(v.values()), axis=0) for m, v in C.items()}
    grand = np.mean([c for m in C for c in C[m].values()], axis=0)
    between = np.mean([np.mean((medmean[m] - grand) ** 2) for m in C])
    within = np.mean([np.mean([np.mean((c - medmean[m]) ** 2) for c in C[m].values()]) for m in C])
    return between / within if within > 0 else np.nan

def ratio_maha(zmat, m_lab, d_lab, media, Sinv):
    C = {m: {} for m in media}
    for m in media:
        for d in GDECS:
            sel = (m_lab == m) & (d_lab == d)
            if sel.sum() >= GMIN: C[m][d] = np.nanmean(zmat[sel], axis=0)
    C = {m: v for m, v in C.items() if v}
    if len(C) < 2: return np.nan
    medmean = {m: np.mean(list(v.values()), axis=0) for m, v in C.items()}
    grand = np.mean([c for m in C for c in C[m].values()], axis=0)
    between = np.mean([maha2(medmean[m] - grand, Sinv) for m in C])
    within = np.mean([np.mean([maha2(c - medmean[m], Sinv) for c in C[m].values()]) for m in C])
    return between / within if within > 0 else np.nan

ratio_euc_3 = ratio_euclid(Zgv, mvec, dvec, ["bk", "fm", "tv"])
ratio_euc_fn = ratio_euclid(Zgv, mvec, dvec, ["bk", "fm"])
ratio_mah_3 = ratio_maha(Zgv, mvec, dvec, ["bk", "fm", "tv"], SINV_STRUCT)
ratio_mah_fn = ratio_maha(Zgv, mvec, dvec, ["bk", "fm"], SINV_STRUCT)

# bootstrap CI for the whitened ratio (resample within medium x decade cells; fixed S^-1)
rng2 = np.random.RandomState(0); GREPS = 1000
def gboot(media):
    idx = np.arange(len(Zgv)); take = np.empty(0, dtype=int)
    for m in media:
        for d in GDECS:
            cell = idx[(mvec == m) & (dvec == d)]
            if len(cell) >= GMIN: take = np.concatenate([take, rng2.choice(cell, len(cell), replace=True)])
    return ratio_maha(Zgv[take], mvec[take], dvec[take], media, SINV_STRUCT)
gboot_3 = np.array([gboot(["bk", "fm", "tv"]) for _ in range(GREPS)])
ci_mah_3 = np.nanpercentile(gboot_3, [2.5, 97.5])

# ======================================================================================
# PART C — HUMAN-COVARIANCE VARIANT (where feasible)
# ======================================================================================
# Human inter-attribute covariance is estimable only where validation works were rated on the SAME
# attribute ids used here. Mood layer: 0 human overlap -> not feasible. Structural VAL16: partial.
hf = pd.read_csv("data/validation/human_means_film.csv")
hb = pd.read_csv("data/validation/human_means_book.csv")
hf_wide = hf.pivot_table(index="survey_movie_id", columns="attribute", values="human_mean")
hb_wide = hb.pivot_table(index="book_idx", columns="attribute", values="human_mean")

human_note = {}
# mood
mood_overlap = [a for a in SH["mood"] if a in set(hf.attribute.unique()) | set(hb.attribute.unique())]
human_note["mood_feasible"] = False
human_note["mood_human_overlap"] = len(mood_overlap)

# structural: attrs present in BOTH the human film table and human book table AND in VAL16
struct_h_film = [a for a in VALCM if a in hf_wide.columns]
struct_h_both = [a for a in VALCM if a in hf_wide.columns and a in hb_wide.columns]
human_note["struct_overlap_film"] = len(struct_h_film)
human_note["struct_overlap_film_and_book"] = len(struct_h_both)

human_ratio = {"feasible": False, "reason": None}
if len(struct_h_both) >= 3:
    sub = struct_h_both
    # human covariance on the shared sub-basis, standardized to unit variance (correlation-like) so it
    # is comparable to the standardized model attributes; pool film+book validation works.
    Hf = hf_wide[sub]; Hb = hb_wide[sub]
    Hpool = pd.concat([(Hf - Hf.mean()) / Hf.std(ddof=0).replace(0, 1),
                       (Hb - Hb.mean()) / Hb.std(ddof=0).replace(0, 1)], ignore_index=True)
    S_human = Hpool.cov().values
    SINV_HUMAN = whiten_inv(S_human)
    # model covariance on the SAME sub-basis (from standardized corpus attrs), for apples-to-apples
    sub_idx = [VALCM.index(a) for a in sub]
    Zsub = Zg[sub]
    S_model_sub = Zsub.cov().values
    SINV_MODEL_SUB = whiten_inv(S_model_sub)
    Zsubv = Zsub.values
    # ratios on the reduced sub-basis under three metrics
    r_euc_sub = ratio_euclid(Zsubv, mvec, dvec, ["bk", "fm", "tv"])
    r_mah_model_sub = ratio_maha(Zsubv, mvec, dvec, ["bk", "fm", "tv"], SINV_MODEL_SUB)
    r_mah_human_sub = ratio_maha(Zsubv, mvec, dvec, ["bk", "fm", "tv"], SINV_HUMAN)
    # coupling: mean abs off-diagonal correlation, human vs model, on the shared sub-basis
    def mean_abs_offdiag(cov):
        d = np.sqrt(np.diag(cov)); C = cov / np.outer(d, d)
        n = C.shape[0]; return float((np.abs(C).sum() - n) / (n * (n - 1)))
    human_ratio = {"feasible": True, "sub_basis_n": len(sub), "sub_basis": sub,
                   "n_works_film": int(Hf.dropna(how="all").shape[0]), "n_works_book": int(Hb.dropna(how="all").shape[0]),
                   "ratio_euclid_subbasis": round(float(r_euc_sub), 3),
                   "ratio_maha_MODEL_cov_subbasis": round(float(r_mah_model_sub), 3),
                   "ratio_maha_HUMAN_cov_subbasis": round(float(r_mah_human_sub), 3),
                   "mean_abs_offdiag_corr_model_sub": round(mean_abs_offdiag(S_model_sub), 3),
                   "mean_abs_offdiag_corr_human_sub": round(mean_abs_offdiag(S_human), 3)}
else:
    human_ratio = {"feasible": False, "reason": f"only {len(struct_h_both)} VAL16 attrs shared across human film+book tables"}

# ======================================================================================
# ASSEMBLE + PRINT
# ======================================================================================
res = {
    "meta": {"ridge": RIDGE, "conv_reps": REPS, "geom_reps": GREPS,
             "note": "Euclidean baselines mirror A4/A2 exactly; Mahalanobis uses pooled within-cell "
                     "(convergence) / pooled (geometry) MODEL covariance; human-cov variant is partial."},
    "task1_selfcheck": {
        "A4_euclid_noveltv_mood_minus_genre_full": euc_contrasts["novel-tv|mood-minus-genre|full"]["diff"],
        "A2_euclid_ratio_3media": round(float(ratio_euc_3), 3),
        "A2_euclid_ratio_film_novel": round(float(ratio_euc_fn), 3)},
    "task2_convergence": {
        "euclid_by_pair_layer": euc_by, "euclid_contrasts": euc_contrasts,
        "maha_by_pair_layer": maha_by, "maha_contrasts": maha_contrasts},
    "task3_geometry": {
        "euclid_ratio_3media": round(float(ratio_euc_3), 3), "euclid_ratio_film_novel": round(float(ratio_euc_fn), 3),
        "maha_ratio_3media": round(float(ratio_mah_3), 3), "maha_ratio_film_novel": round(float(ratio_mah_fn), 3),
        "maha_ratio_3media_CI": [round(float(ci_mah_3[0]), 3), round(float(ci_mah_3[1]), 3)]},
    "task4_human_covariance": {"note": human_note, "structural_ratio_variant": human_ratio}}

# verdicts
def mood_wins(contrasts):
    keys = [k for k in contrasts if k.startswith(("novel-tv|", "film-tv|"))]
    pos = sum(1 for k in keys if contrasts[k]["diff"] > 0)
    excl = sum(1 for k in keys if contrasts[k]["excludes0"] and contrasts[k]["diff"] > 0)
    return pos, excl, len(keys)
me_p, me_x, me_n = mood_wins(euc_contrasts)
mm_p, mm_x, mm_n = mood_wins(maha_contrasts)
res["verdict"] = {
    "mood_converges_most_euclid": f"{me_p}/{me_n} contrasts positive, {me_x} exclude 0",
    "mood_converges_most_maha": f"{mm_p}/{mm_n} contrasts positive, {mm_x} exclude 0",
    "mood_result_survives_whitening": bool(mm_p >= me_p - 1 and mm_p >= me_n / 2),
    "medium_over_era_euclid": round(float(ratio_euc_3), 3),
    "medium_over_era_maha": round(float(ratio_mah_3), 3),
    "medium_over_era_survives_whitening": bool(ratio_mah_3 > 1.0)}

os.makedirs("pnas-sub/analysis/out", exist_ok=True)
json.dump(res, open("pnas-sub/analysis/out/A12_whitened.json", "w"), indent=2)

# ---- console summary ----
print("=" * 78)
print("TASK 1  SELF-CHECK")
print(f"  A4 Euclid novel-tv mood-minus-genre (full): {res['task1_selfcheck']['A4_euclid_noveltv_mood_minus_genre_full']:+.3f}  [target ~= 0.31]")
print(f"  A2 Euclid ratio 3-media: {ratio_euc_3:.3f}  [target ~= 1.66];  film-novel: {ratio_euc_fn:.3f}")
print("\nTASK 2  MAHALANOBIS CONVERGENCE  (conv = distance 1950 minus later; + = converged)")
for pn in PAIRS:
    for L in LAYERS:
        e = euc_by[f"{pn}|{L}"]; m = maha_by[f"{pn}|{L}"]
        print(f"  {pn:9s} {L:6s} full: euclid {e['conv_full']:+.3f} CI{e['conv_full_CI']} | maha {m['conv_full']:+.3f} CI{m['conv_full_CI']}")
print("  -- contrasts (mood minus other, full horizon): --")
for pn in PAIRS:
    for other in ["genre", "arc"]:
        k = f"{pn}|mood-minus-{other}|full"
        e = euc_contrasts[k]; m = maha_contrasts[k]
        print(f"  {pn:9s} mood-{other:5s}: euclid {e['diff']:+.3f} CI{e['CI']} {'*' if e['excludes0'] else ' '} | "
              f"maha {m['diff']:+.3f} CI{m['CI']} {'*' if m['excludes0'] else ' '}")
print(f"\n  verdict mood: euclid {res['verdict']['mood_converges_most_euclid']} | maha {res['verdict']['mood_converges_most_maha']}")
print(f"  -> mood-converges-most survives whitening: {res['verdict']['mood_result_survives_whitening']}")
print("\nTASK 3  MAHALANOBIS GEOMETRY (medium-vs-era ratio)")
print(f"  euclid 3-media {ratio_euc_3:.3f} -> maha 3-media {ratio_mah_3:.3f}  CI[{ci_mah_3[0]:.2f},{ci_mah_3[1]:.2f}]")
print(f"  euclid film-novel {ratio_euc_fn:.3f} -> maha film-novel {ratio_mah_fn:.3f}")
print(f"  -> medium-over-era (ratio>1) survives whitening: {res['verdict']['medium_over_era_survives_whitening']}")
print("\nTASK 4  HUMAN-COVARIANCE VARIANT")
print(f"  mood human overlap: {human_note['mood_human_overlap']}/{len(SH['mood'])} attrs -> NOT feasible to whiten mood by human cov")
print(f"  structural VAL16 overlap: film {human_note['struct_overlap_film']}/16, film+book {human_note['struct_overlap_film_and_book']}/16")
if human_ratio.get("feasible"):
    hr = human_ratio
    print(f"  sub-basis n={hr['sub_basis_n']} (works: {hr['n_works_film']} film, {hr['n_works_book']} book)")
    print(f"  ratio on sub-basis: euclid {hr['ratio_euclid_subbasis']:.3f} | MODEL-cov whiten {hr['ratio_maha_MODEL_cov_subbasis']:.3f} | HUMAN-cov whiten {hr['ratio_maha_HUMAN_cov_subbasis']:.3f}")
    print(f"  mean|off-diag corr| sub-basis: model {hr['mean_abs_offdiag_corr_model_sub']:.3f} vs human {hr['mean_abs_offdiag_corr_human_sub']:.3f}")
else:
    print(f"  structural human ratio NOT feasible: {human_ratio.get('reason')}")
print("=" * 78)
print("wrote pnas-sub/analysis/out/A12_whitened.json")
