"""A02 (PNAS): human-vs-model inter-attribute covariance agreement on title-matched films.

Reproduces the main-text claim (02_results1_signatures.tex L13-14; si.tex L103): across the
attributes that humans and the model both score on the SAME title-matched films, the two
inter-attribute correlation matrices "agree closely (off-diagonal r=0.82) and share their
principal axis, though the model couples attributes about 1.4 times more tightly than raters."

DESIGN (paired, matched-film — NOT the full-corpus pooling A12 uses).
  * Human side : per-work human MEANS, data/validation/human_means_film.csv (survey_movie_id x attribute).
  * Model side : the DEPLOYED corpus score for the SAME film, data/atlas/century_frame_film.parquet,
    linked survey_movie_id -> corpus row via data/validation/survey_to_corpus_map.csv. This is the
    canonical, corrected linkage used by code/build_film_validation.py and replication/reproduce.py
    (it supersedes the earlier mis-linked plot-LLM pass in film_llm_validation_scores.csv).
  * Basis (FULL overlap): every human-means attribute that is also a corpus column (scalar structural
    + evaluative/mood scalars). Both matrices are built across the SAME matched films.

THREE NUMBERS, per basis:
  1. offdiag_agreement_r : Pearson r between the upper-triangle off-diagonals of the human vs model
     inter-attribute CORRELATION matrices.
  2. principal_axis_cosine : |cos| between the leading eigenvector of the two correlation matrices.
  3. coupling_ratio : mean(|off-diagonal|) of the MODEL correlation matrix / mean(|off-diagonal|) of
     the HUMAN correlation matrix.
Reported on the FULL overlap and on the 8-attr STRUCTURAL subset (A12's struct_h_both list) as a
cross-check against A12's model-vs-human coupling 0.231/0.188 = 1.23x.

SELF-CHECK: n matched films should be ~240 and the 8 struct attrs are exactly A12's sub_basis; the
full-overlap coupling_ratio is the number the main text's "1.4 times" must match. Correlation uses
pairwise-complete observations (human means carry ~11% NaN from skipped protagonist items; model 0%).
Bounded, single-threaded, no LLM. Writes out/A02_covariance_agreement.json. Run from ~/uoft/style_evolves/.
"""
import os, json
import numpy as np, pandas as pd

ROOT = os.environ.get("NARRATIVE_ATLAS_ROOT", os.path.expanduser("~/uoft/style_evolves")); os.chdir(ROOT)
V = os.path.join(ROOT, "data/validation")

# 8-attr structural subset: exactly A12_whitened.json task4.structural_ratio_variant.sub_basis
STRUCT8 = [
    "1_how_many_protagonists_were_there_a_protagonist_is_the_person_or_enti",
    "8b_how_competent_was_this_protagonist_by_competent_we_mean_effective_a",
    "13_how_relatable_did_you_find_this_protagonist",
    "3_how_realistic_was_the_world_of_the_movie_by_realistic_we_mean_plausi",
    "5_how_fantastical_was_the_world_of_the_movies_by_fantastical_we_mean_a",
    "6_how_science_fictional_was_the_world_of_the_movie_by_science_fictiona",
    "7_overall_how_relevant_was_world_building_to_the_movie_world_building",
    "8c_how_proactive_was_this_protagonist_by_proactive_we_mean_motivated_a",
]


def mean_abs_offdiag(C):
    """Mean absolute off-diagonal of a square (correlation) matrix, ignoring NaN cells."""
    C = np.asarray(C, float)
    n = C.shape[0]
    off = C[~np.eye(n, dtype=bool)]
    return float(np.nanmean(np.abs(off)))


def offdiag_agreement(Ch, Cm):
    """Pearson r between the upper-triangle off-diagonal entries of two correlation matrices."""
    n = Ch.shape[0]
    iu = np.triu_indices(n, k=1)
    x = np.asarray(Ch)[iu]; y = np.asarray(Cm)[iu]
    m = ~(np.isnan(x) | np.isnan(y))
    return float(np.corrcoef(x[m], y[m])[0, 1])


def principal_axis_cosine(Ch, Cm):
    """|cosine| between the leading eigenvectors of two symmetric correlation matrices.

    NaNs (from pairwise-complete correlation on attrs never co-rated) are zero-filled on the
    off-diagonal so the eigendecomposition is defined; diagonal forced to 1.
    """
    def lead_vec(C):
        C = np.array(C, float)
        C[np.isnan(C)] = 0.0
        np.fill_diagonal(C, 1.0)
        w, v = np.linalg.eigh(C)
        return v[:, np.argmax(w)]
    a, b = lead_vec(Ch), lead_vec(Cm)
    return float(abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))


# ---- load ----
CORPUS = pd.read_parquet(os.path.join(ROOT, "data/atlas/century_frame_film.parquet")).reset_index(drop=True)
HM = pd.read_csv(os.path.join(V, "human_means_film.csv")); HM["survey_movie_id"] = HM.survey_movie_id.astype(str)
CMAP = pd.read_csv(os.path.join(V, "survey_to_corpus_map.csv")); CMAP["survey_movie_id"] = CMAP.survey_movie_id.astype(str)
SID2IDX = dict(zip(CMAP[CMAP.corpus_idx >= 0].survey_movie_id, CMAP[CMAP.corpus_idx >= 0].corpus_idx.astype(int)))

# FULL overlap basis: every human-means attribute that is also a corpus column
FULL = [a for a in HM.attribute.unique() if a in CORPUS.columns]

# human wide: films x attributes (per-work means)
HW = HM.pivot_table(index="survey_movie_id", columns="attribute", values="human_mean")[FULL]
# keep only films with a corpus link
sids = [s for s in HW.index if s in SID2IDX]
HW = HW.loc[sids]
# model wide: the deployed corpus score for the same films, same attribute columns
MW = pd.DataFrame({a: [CORPUS.iloc[SID2IDX[s]][a] for s in sids] for a in FULL}, index=sids)

assert list(HW.columns) == list(MW.columns) == FULL


def run_basis(cols):
    hw, mw = HW[cols], MW[cols]
    Ch = hw.corr().values          # human inter-attribute correlation (pairwise complete)
    Cm = mw.corr().values          # model inter-attribute correlation (pairwise complete)
    mh, mm = mean_abs_offdiag(Ch), mean_abs_offdiag(Cm)
    return {
        "n_attrs": len(cols),
        "offdiag_agreement_r": round(offdiag_agreement(Ch, Cm), 4),
        "principal_axis_cosine": round(principal_axis_cosine(Ch, Cm), 4),
        "mean_abs_offdiag_human": round(mh, 4),
        "mean_abs_offdiag_model": round(mm, 4),
        "coupling_ratio": round(mm / mh, 4),
    }


full = run_basis(FULL)
struct = run_basis(STRUCT8)

res = {
    "n_films": len(sids),
    "n_attrs_full": full["n_attrs"],
    "offdiag_agreement_r_full": full["offdiag_agreement_r"],
    "principal_axis_cosine_full": full["principal_axis_cosine"],
    "coupling_ratio_full": full["coupling_ratio"],
    "mean_abs_offdiag_model_full": full["mean_abs_offdiag_model"],
    "mean_abs_offdiag_human_full": full["mean_abs_offdiag_human"],
    "n_attrs_struct": struct["n_attrs"],
    "offdiag_agreement_r_struct": struct["offdiag_agreement_r"],
    "principal_axis_cosine_struct": struct["principal_axis_cosine"],
    "coupling_ratio_struct": struct["coupling_ratio"],
    "mean_abs_offdiag_model_struct": struct["mean_abs_offdiag_model"],
    "mean_abs_offdiag_human_struct": struct["mean_abs_offdiag_human"],
    "meta": {
        "model_side": "deployed corpus century_frame_film.parquet via survey_to_corpus_map.csv (canonical linkage)",
        "human_side": "human_means_film.csv per-work means",
        "basis_full": "all human-means attributes that are corpus columns (scalar structural + evaluative/mood scalars)",
        "basis_struct": "A12 struct_h_both 8-attr subset",
        "metric": "inter-attribute Pearson correlation (pairwise complete), coupling = mean|off-diag| model / human",
        "A12_reference": {"model_meanabs_offdiag": 0.231, "human_meanabs_offdiag": 0.188, "ratio": 1.23,
                          "note": "A12 uses FULL-corpus model cov and pooled film+book human cov, not matched films"},
    },
}

os.makedirs(os.path.join(ROOT, "pnas-sub/analysis/out"), exist_ok=True)
json.dump(res, open(os.path.join(ROOT, "pnas-sub/analysis/out/A02_covariance_agreement.json"), "w"), indent=2)

print("=" * 72)
print(f"matched films: {res['n_films']}")
print(f"FULL overlap basis: {res['n_attrs_full']} attrs")
print(f"  off-diagonal agreement r : {res['offdiag_agreement_r_full']:.3f}   [target ~0.82]")
print(f"  principal-axis |cosine|  : {res['principal_axis_cosine_full']:.3f}")
print(f"  mean|off-diag| model     : {res['mean_abs_offdiag_model_full']:.3f}")
print(f"  mean|off-diag| human     : {res['mean_abs_offdiag_human_full']:.3f}")
print(f"  coupling ratio (m/h)     : {res['coupling_ratio_full']:.3f}   [target ~1.4]")
print(f"8-attr STRUCTURAL subset (A12 cross-check):")
print(f"  off-diagonal agreement r : {res['offdiag_agreement_r_struct']:.3f}")
print(f"  mean|off-diag| model     : {res['mean_abs_offdiag_model_struct']:.3f}   [A12: 0.231]")
print(f"  mean|off-diag| human     : {res['mean_abs_offdiag_human_struct']:.3f}   [A12: 0.188]")
print(f"  coupling ratio (m/h)     : {res['coupling_ratio_struct']:.3f}   [A12: 1.23]")
print("=" * 72)
print("wrote pnas-sub/analysis/out/A02_covariance_agreement.json")
