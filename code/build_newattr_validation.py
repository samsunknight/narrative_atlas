"""Recompute the frozen 'new-attribute' validation table (data/validation/newattr_final.csv)
from the shipped per-work AGGREGATE human means + the DEPLOYED atlas parquets. NO LLM scoring.

Human means come from build_newattr_human_means.py's output (per-work, PII-free):
    data/validation/newattr_human_means_film.csv   (work_key = survey_movie_id)
    data/validation/newattr_human_means_book.csv    (work_key = normalized title)
Atlas scores come from data/atlas/century_frame_{film,book}.parquet.

ALIGNMENT (this is the alignment the working validators use; a value-join on the frame index
would collapse r toward 0):
  film : survey_movie_id -> corpus_idx via survey_to_corpus_map.csv, then CORPUS.iloc[idx].
  book : normalized title -> the atlas row with that title (titles that are unique atlas-side).

AGGREGATION for each newattr_final row (see build_newattr_human_means.py header for the human
side):
  setting_when  : the era-defining option correlation (distant-future era -- the cleanest era
                  signal, and the headline the frozen table reports; film 0.71 / book 0.78).
  setting_where : mean of the three place-option correlations.
  conflict_layer(6-type) : mean of the six per-type correlations (newattr_final 'note').
  shapes / scalars : the single-column correlation for that attribute.
Genre is not a newattr_final row (it is validated by AUC elsewhere), so no AUC path is needed here.

Output: data/validation/newattr_final_recomputed.csv, and a printed frozen-vs-recomputed table.
Run with --write to additionally regenerate data/validation/newattr_final.csv in place.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def P(*a): return os.path.join(ROOT, *a)

F = pd.read_parquet(P("data/atlas/century_frame_film.parquet")).reset_index(drop=True)
B = pd.read_parquet(P("data/atlas/century_frame_book.parquet")).reset_index(drop=True)
HMF = pd.read_csv(P("data/validation/newattr_human_means_film.csv")); HMF["work_key"] = HMF.work_key.astype(str)
HMB = pd.read_csv(P("data/validation/newattr_human_means_book.csv")); HMB["work_key"] = HMB.work_key.astype(str)
HMF["option"] = HMF.option.fillna(""); HMB["option"] = HMB.option.fillna("")

CMAP = pd.read_csv(P("data/validation/survey_to_corpus_map.csv")); CMAP["survey_movie_id"] = CMAP.survey_movie_id.astype(str)
SID2IDX = dict(zip(CMAP[CMAP.corpus_idx >= 0].survey_movie_id, CMAP[CMAP.corpus_idx >= 0].corpus_idx.astype(int)))

import re
def nt(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())
B["_nt"] = B.title.map(nt)
_vc = B._nt.value_counts(); _good = set(_vc[_vc == 1].index)
def book_col_map(col): return dict(zip(B[B._nt.isin(_good)]._nt, B[B._nt.isin(_good)][col]))


def pearson(xs, ys, minn=10):
    if len(xs) >= minn and np.std(xs) > 0 and np.std(ys) > 0:
        return float(np.corrcoef(xs, ys)[0, 1]), len(xs)
    return float("nan"), len(xs)


def film_r(attribute, option, col):
    sub = HMF[(HMF.attribute == attribute) & (HMF.option == option)]
    hb = dict(zip(sub.work_key, sub.human_mean))
    xs, ys = [], []
    for k, hv in hb.items():
        idx = SID2IDX.get(k)
        if idx is None or col not in F.columns: continue
        cv = F.iloc[idx][col]
        if pd.notna(cv) and pd.notna(hv): xs.append(float(cv)); ys.append(float(hv))
    return pearson(xs, ys)


def book_r(attribute, option, col):
    if col not in B.columns: return float("nan"), 0
    tc = book_col_map(col)
    sub = HMB[(HMB.attribute == attribute) & (HMB.option == option)]
    xs, ys = [], []
    for k, hv in zip(sub.work_key, sub.human_mean):
        if k in tc and pd.notna(tc[k]) and pd.notna(hv): xs.append(float(tc[k])); ys.append(float(hv))
    return pearson(xs, ys)


# option slug lists
WHEN = ["around_present_day", "in_the_near_past", "in_the_distant_past", "in_the_near_future", "in_the_distant_future"]
WHERE = ["somewhere_on_earth_that_really_exists", "somewhere_on_earth_that_is_imaginary", "somewhere_off_earth_that_is_imaginary"]
CONF = ["self", "other", "society", "nature", "technology", "fate"]
ERA = "in_the_distant_future"   # era-defining option (the setting-era headline)


def recompute_row(attribute):
    """Return (film_r, book_r) recomputed for a newattr_final attribute name."""
    if attribute == "setting_when":
        fr = film_r("setting_when", ERA, f"setting_when__{ERA}")[0]
        br = book_r("setting_when", ERA, f"book_setting_when__{ERA}")[0]
        return fr, br
    if attribute == "setting_where":
        fr = np.nanmean([film_r("setting_where", s, f"setting_where__{s}")[0] for s in WHERE])
        br = np.nanmean([book_r("setting_where", s, f"book_setting_where__{s}")[0] for s in WHERE])
        return float(fr), float(br)
    if attribute == "conflict_layer(6-type)":
        fr = np.nanmean([film_r("conflict", s, f"conflict_{s}")[0] for s in CONF])
        br = np.nanmean([book_r("conflict", s, f"book_conflict_{s}")[0] for s in CONF])
        return float(fr), float(br)
    if attribute.startswith("shape_"):
        slug = attribute[len("shape_"):]
        fr = film_r("shape", slug, f"shape_{slug}")[0]
        br = book_r("shape", slug, f"book_shape_{slug}")[0]
        return fr, br
    if attribute == "num_narrators":
        # book-only narration count (film has no per-work narrator count in the survey)
        return float("nan"), book_r("num_narrators", "", "book_num_narrators")[0]
    # scalar structure items: film col = bare name, book col = book_ prefix
    fcol = attribute
    bcol = "book_" + ("Q622_time_linearity" if attribute == "time_linearity" else attribute)
    fr = film_r(attribute, "", fcol)[0]
    br = book_r(attribute, "", bcol)[0]
    return fr, br


def main():
    NF = pd.read_csv(P("data/validation/newattr_final.csv"))
    recomp = []
    print(f"{'attribute':26s} {'film':>14s}   {'book':>14s}")
    print(f"{'':26s} {'frozen  recomp':>14s}   {'frozen  recomp':>14s}   match")
    for _, row in NF.iterrows():
        attr = row.attribute
        fr, br = recompute_row(attr)
        ff = row.film if pd.notna(row.film) else np.nan
        bf = row.book if pd.notna(row.book) else np.nan
        def cell(frozen, rec):
            if pd.isna(frozen) and pd.isna(rec): return "  -      -   ", True
            if pd.isna(frozen): return f"  -    {rec:5.2f} ", True
            if pd.isna(rec): return f"{frozen:5.2f}   nan ", False
            return f"{frozen:5.2f}  {rec:5.2f} ", abs(frozen - rec) <= 0.02
        fs, fok = cell(ff, fr); bs, bok = cell(bf, br)
        ok = fok and bok
        print(f"{attr:26s} {fs}   {bs}   {'OK' if ok else 'XX'}")
        recomp.append({"attribute": attr, "layer": row.layer, "metric": row.metric,
                       "film": None if pd.isna(fr) else round(fr, 2), "book": None if pd.isna(br) else round(br, 2),
                       "tier": row.tier, "spine": row.spine, "cross_medium": row.cross_medium})
    out = pd.DataFrame(recomp)
    out.to_csv(P("data/validation/newattr_final_recomputed.csv"), index=False)
    if "--write" in sys.argv:
        out.to_csv(P("data/validation/newattr_final.csv"), index=False)
        print("\n[--write] regenerated data/validation/newattr_final.csv")


if __name__ == "__main__":
    main()
