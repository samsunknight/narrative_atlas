#!/usr/bin/env python3
"""F3 JOINT-STRUCTURE validation (main text): does the atlas's *multivariate*
structure agree with human perception, not just each marginal attribute?

Reconstructs, from shipped package data only, the three headline numbers the
paper reports for the clean title-matched film validation set:

  (a) off-diagonal Pearson r between the human and machine 44x44 inter-attribute
      correlation matrices  -> paper 0.77
  (b) PC1 Tucker congruence (|cosine| of the matched leading eigenvector of each
      correlation matrix)    -> paper 0.96
  (c) mean|corr| human vs machine and the ratio (coupling "halo" inflation)
      -> paper human 0.245 / machine 0.345 = ~1.4x

METHOD
------
* Human ground truth: per-work human means over the validation films, long form
  in data/validation/human_means_film.csv (survey_movie_id, attribute, human_mean).
* Machine scores: the SAME films' deployed-corpus scores from
  data/atlas/century_frame_film.parquet.
* Films are joined by TITLE (normalize, strip a trailing "(YYYY)") using the survey
  title in data/validation/survey_to_corpus_map.csv against the atlas `title` column.
  This is the clean title-join the paper reports; it yields 115 films.
* Attributes: the 44 survey structural attributes that are scored as columns in the
  deployed atlas (the 5 evaluative/reception survey items with no atlas column drop out).
* Each 44x44 correlation matrix is computed pairwise (pandas .corr()); the off-diagonal
  agreement is the Pearson r of the two matrices' upper-triangle entries; coupling is the
  mean absolute off-diagonal correlation; PC1 = leading eigenvector of the (nan->0)
  correlation matrix, congruence = |cosine| of the two leading vectors.

Run from the package root:  ../.venv/bin/python3 code/joint_structure.py
Prints the three numbers; importable as compute() for reproduce.py.
"""
import os, re
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def P(*a): return os.path.join(ROOT, *a)
def norm(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())
def title_base(t): return norm(re.sub(r"\(\d{4}\)\s*$", "", str(t)))  # drop trailing "(YYYY)"


def compute():
    CORPUS = pd.read_parquet(P("data/atlas/century_frame_film.parquet")).reset_index(drop=True)
    HM = pd.read_csv(P("data/validation/human_means_film.csv"))
    HM["survey_movie_id"] = HM.survey_movie_id.astype(str)
    CMAP = pd.read_csv(P("data/validation/survey_to_corpus_map.csv"))
    CMAP["survey_movie_id"] = CMAP.survey_movie_id.astype(str)

    # the 44 survey structural attributes that exist as deployed-atlas columns
    attrs = [a for a in sorted(HM.attribute.unique()) if a in CORPUS.columns]

    # title-join survey films to the atlas
    CMAP["nt"] = CMAP.survey_title.map(title_base)
    atl = CORPUS.assign(nt=CORPUS.title.map(norm)).drop_duplicates("nt").set_index("nt")
    matched = CMAP[CMAP.nt.isin(atl.index)].copy()
    sid2row = {r.survey_movie_id: atl.loc[r.nt] for _, r in matched.iterrows()}

    # human films x 44 attrs
    Hdf = pd.DataFrame({a: HM[HM.attribute == a].set_index("survey_movie_id").human_mean
                        for a in attrs})
    # machine films x 44 attrs (same films, by title)
    Mdf = pd.DataFrame({sid: sid2row[sid][attrs] for sid in Hdf.index if sid in sid2row}).T
    Mdf.columns = attrs

    common = Hdf.index.intersection(Mdf.index)
    Hdf = Hdf.loc[common].apply(pd.to_numeric, errors="coerce")
    Mdf = Mdf.loc[common].apply(pd.to_numeric, errors="coerce")

    Hc, Mc = Hdf.corr(), Mdf.corr()
    iu = np.triu_indices(len(attrs), 1)
    ho, mo = Hc.values[iu], Mc.values[iu]
    m = ~(np.isnan(ho) | np.isnan(mo))

    offdiag_r = float(np.corrcoef(ho[m], mo[m])[0, 1])
    mean_h = float(np.nanmean(np.abs(ho)))
    mean_m = float(np.nanmean(np.abs(mo)))
    ratio = mean_m / mean_h

    def pc1(cm):
        A = cm.values.copy(); A[np.isnan(A)] = 0.0
        w, V = np.linalg.eigh(A); return V[:, np.argmax(w)]
    h1, m1 = pc1(Hc), pc1(Mc)
    tucker = float(abs(np.dot(h1, m1) / (np.linalg.norm(h1) * np.linalg.norm(m1))))

    return {"n_films": int(len(common)), "n_attr": len(attrs),
            "offdiag_r": offdiag_r, "tucker_pc1": tucker,
            "coupling_human": mean_h, "coupling_machine": mean_m, "coupling_ratio": ratio}


if __name__ == "__main__":
    r = compute()
    print(f"F3 joint-structure (clean title-join): {r['n_films']} films x {r['n_attr']} attrs")
    print(f"  (a) off-diagonal corr-matrix agreement Pearson r = {r['offdiag_r']:.3f}   (paper 0.77)")
    print(f"  (b) PC1 Tucker congruence                        = {r['tucker_pc1']:.3f}   (paper 0.96)")
    print(f"  (c) mean|corr| human = {r['coupling_human']:.3f}  machine = {r['coupling_machine']:.3f}  "
          f"ratio = {r['coupling_ratio']:.3f}   (paper ~1.4x)")
