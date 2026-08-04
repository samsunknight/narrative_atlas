"""GENERATION STEP (not part of reproduce.py). Stages the per-work AGGREGATE human means
for the newer atlas attributes -- setting (era/place, select-all), conflict (6 primary types),
the Vonnegut story shapes, and the scalar structure items (character development, opening hook,
time/plot linearity, ending reversal, catharsis) -- so that build_newattr_validation.py can
recompute their validation r's against the deployed atlas WITHOUT touching any rater-level data.

The raw survey exports (rater-level; carry PII) are NOT in this release. Place them under
inputs/ to re-run. What ships and what reproduce reads is only the per-work aggregate this
produces:  data/validation/newattr_human_means_film.csv  and  ..._book.csv.

Raw inputs (external, not shipped):
  film :  inputs/movie_survey_raw_2025-08.csv, inputs/movie_survey_raw_2025-09_RA.csv
          (Qualtrics exports; Q720 = survey_movie_id; per-protagonist conflict/development
           blocks; Q479/Q480 setting select-all; Q627 shape; Q622/Q623 linearity; Q632 hook;
           Q729 ending reversal; Q731 emotional release/catharsis; Q244 = 1st-protagonist dev.)
  book :  df_y_v2.csv (booknet review-annotation release: per-work aggregate `target` keyed by
           canonical_q720) + book_id_map.csv / audit_book_results.csv (canonical_q720 -> title).

Aggregation conventions (matched to how the frozen newattr_final table was built):
  * setting when/where : per-option human proportion = share of raters selecting that option.
  * conflict (6 types) : per-type human value = share of protagonist-level PRIMARY-conflict
                         responses of that type (a rater with k protagonists contributes k
                         responses; per-work = mean of the rater shares).  Film aggregates the
                         raw per-protagonist columns; book uses df_y_v2's pre-pooled `target`.
  * story shape        : per-shape human proportion = share of raters picking that shape.
  * character develop. : FILM uses the 1st-protagonist item only (Q244), matching the frozen
                         n=132 / r=0.58; encoded No=1 / Arguably=2 / Yes=3.
  * scalar structure   : linearity 1(linear)..4(nonlinear); hook / reversal / catharsis
                         No=1 / Somewhat|Arguably=2 / Yes=3 (mean over raters).

Output columns (long): work_key, attribute, option, human_mean, n_raters
  film work_key = survey_movie_id (joins to the atlas via survey_to_corpus_map.csv)
  book work_key = normalized title (joins to the atlas title column)
"""
import os, csv, re
import numpy as np, pandas as pd
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def P(*a): return os.path.join(ROOT, *a)
# Fall back to the working-tree locations if inputs/ is absent (author re-runs).
STY = os.path.expanduser("~/uoft/style_evolves")
BOOKNET = "/Users/samsunknight/uoft/booknet/review_annotation_paper/out"
FILM_RAW = [P("inputs", "movie_survey_raw_2025-08.csv"), P("inputs", "movie_survey_raw_2025-09_RA.csv")]
if not all(os.path.exists(f) for f in FILM_RAW):
    FILM_RAW = [os.path.join(STY, "data/surveys/2025-08 Movie Genome Survey_June 18, 2026_10.31.csv"),
                os.path.join(STY, "data/surveys/2025-09 Movie Genome Survey - RA_June 18, 2026_10.33.csv")]
DFY = P("inputs", "df_y_v2.csv");           DFY = DFY if os.path.exists(DFY) else f"{BOOKNET}/df_y_v2.csv"
AUDIT = P("inputs", "audit_book_results.csv"); AUDIT = AUDIT if os.path.exists(AUDIT) else os.path.join(STY, "survey_review/audit_book_results.csv")

WHEN = {"Around present day": "around_present_day", "In the near past": "in_the_near_past",
        "In the distant past": "in_the_distant_past", "In the near future": "in_the_near_future",
        "In the distant future": "in_the_distant_future"}
WHERE = {"Somewhere on Earth that really exists": "somewhere_on_earth_that_really_exists",
         "Somewhere on Earth that is imaginary": "somewhere_on_earth_that_is_imaginary",
         "Somewhere off Earth that is imaginary": "somewhere_off_earth_that_is_imaginary"}
SHAPE = {"Cinderella": "cinderella", "Rags to riches": "rags_to_riches", "Riches to rags": "riches_to_rags",
         "Man in a hole": "man_in_a_hole", "Icarus": "icarus"}
CONF = {"Protagonist vs. self": "self", "Protagonist vs. other character(s)": "other",
        "Protagonist vs. society": "society", "Protagonist vs. nature": "nature",
        "Protagonist vs. technology": "technology", "Protagonist vs. fate/luck": "fate"}
LIN = {"Completely linear": 1, "More linear than non-linear (mostly sequential, with some digressions)": 2,
       "More non-linear than linear (mostly non-sequential, with some progressions)": 3, "Completely nonlinear": 4}
YSN = {"Yes, definitely": 3, "Yes": 3, "Arguably - you could make the case, but it’s not clear-cut": 2,
       "Somewhat": 2, "Maybe": 2, "No": 1, "Not at all": 1}


def build_film():
    when = defaultdict(lambda: defaultdict(list)); where = defaultdict(lambda: defaultdict(list))
    shape = defaultdict(lambda: defaultdict(list)); conf = defaultdict(list); scal = defaultdict(lambda: defaultdict(list))
    nresp = defaultdict(int)
    for path in FILM_RAW:
        r = list(csv.reader(open(path, encoding="utf-8", errors="replace")))
        hdr, txt = r[0], r[1]; ci = {h: i for i, h in enumerate(hdr)}
        primcols = [h for h, t in zip(hdr, txt) if "primary type of conflict" in str(t) and not h.endswith("_TEXT")]
        def g(row, q): return row[ci[q]].strip() if q in ci and ci[q] < len(row) else ""
        for row in r[3:]:
            mid = g(row, "Q720")
            if not mid: continue
            nresp[mid] += 1
            wv = g(row, "Q479")
            if wv:
                sel = set(x.strip() for x in wv.split(","))
                for opt, slug in WHEN.items(): when[mid][slug].append(1 if opt in sel else 0)
            ev = g(row, "Q480")
            if ev:
                sel = set(x.strip() for x in ev.split(","))
                for opt, slug in WHERE.items(): where[mid][slug].append(1 if any(s.startswith(opt) for s in sel) else 0)
            sv = g(row, "Q627")
            if sv:
                for opt, slug in SHAPE.items(): shape[mid][slug].append(1 if sv == opt else 0)
            prim = [CONF[g(row, h)] for h in primcols if g(row, h) in CONF]
            if any(g(row, h) for h in primcols) and prim:
                tot = len(prim)
                conf[mid].append({s: prim.count(s) / tot for s in CONF.values()})
            for q, dim in [("Q622", "time_linearity"), ("Q623", "plot_linearity")]:
                if g(row, q) in LIN: scal[mid][dim].append(LIN[g(row, q)])
            for q, dim in [("Q632", "opening_hook"), ("Q729", "ending_reversal"), ("Q731", "catharsis")]:
                if g(row, q) in YSN: scal[mid][dim].append(YSN[g(row, q)])
            if g(row, "Q244") in YSN: scal[mid]["character_development"].append(YSN[g(row, "Q244")])
    rows = []
    for mid in set(list(when) + list(where) + list(shape) + list(conf) + list(scal)):
        n = nresp[mid]
        for slug, vs in when[mid].items(): rows.append((mid, "setting_when", slug, np.mean(vs), len(vs)))
        for slug, vs in where[mid].items(): rows.append((mid, "setting_where", slug, np.mean(vs), len(vs)))
        for slug, vs in shape[mid].items(): rows.append((mid, "shape", slug, np.mean(vs), len(vs)))
        if conf[mid]:
            for s in CONF.values():
                rows.append((mid, "conflict", s, float(np.mean([d[s] for d in conf[mid]])), len(conf[mid])))
        for dim, vs in scal[mid].items(): rows.append((mid, dim, "", float(np.mean(vs)), len(vs)))
    out = pd.DataFrame(rows, columns=["work_key", "attribute", "option", "human_mean", "n_raters"])
    out["human_mean"] = out.human_mean.astype(float).round(6)
    return out


def nt(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())


def build_book():
    D = pd.read_csv(DFY); D["canonical_q720"] = D.canonical_q720.astype(str)
    AB = pd.read_csv(AUDIT); AB["bid"] = AB.bid.astype(str)
    bid2t = dict(zip(AB.bid, AB.wiki_title))     # canonical_q720 -> title (atlas join key source)

    def key_series(sub, choice_filter=None):
        s = D[D.conceptual_id.str.contains(sub, regex=False)]
        if choice_filter is not None: s = s[s.choice_id.str.startswith(choice_filter)]
        return s

    def emit_prop(sub, mapping, attribute):
        for choice, slug in mapping.items():
            s = key_series(sub, choice)
            g = s.groupby("canonical_q720").agg(target=("target", "max"), n=("n_readers", "max"))
            for bid, r in g.iterrows():
                t = bid2t.get(bid)
                if t is None or pd.isna(r.target): continue
                rows.append((nt(t), attribute, slug, float(r.target), int(r.n)))

    def emit_scalar(sub, attribute, sign=1):
        s = key_series(sub, "<scalar>")
        for _, r in s.iterrows():
            t = bid2t.get(r.canonical_q720)
            if t is None or pd.isna(r.target): continue
            rows.append((nt(t), attribute, "", sign * float(r.target), int(r.n_readers)))

    def emit_cat_ordinal(sub, order, attribute):
        s = key_series(sub)
        for bid, gg in s.groupby("canonical_q720"):
            t = bid2t.get(bid)
            if t is None: continue
            m = dict(zip(gg.choice_id, gg.target)); tot = sum(m.get(c, 0) for c in order)
            if tot > 0:
                rows.append((nt(t), attribute, "", sum(m.get(c, 0) * rk for c, rk in order.items()) / tot, int(gg.n_readers.max())))

    rows = []
    emit_prop("When does the book take place", WHEN, "setting_when")
    emit_prop("Where does the book take place", WHERE, "setting_where")
    emit_prop("Which story shape best describes", {k: v for k, v in SHAPE.items() if v in ("cinderella", "rags_to_riches", "riches_to_rags")}, "shape")
    emit_prop("primary type of conflict", CONF, "conflict")
    emit_scalar("strong personal development", "character_development")
    emit_scalar("linearity in terms of plot", "plot_linearity", sign=-1)   # atlas book_plot_linearity runs opposite
    emit_scalar("linearity in terms of time", "time_linearity", sign=-1)
    emit_scalar("begin with a", "opening_hook")
    emit_scalar("how many narrators were there", "num_narrators")   # book-only narration count
    emit_cat_ordinal("major reversal-of-fortune", {"No": 1, "Somewhat": 2, "Yes": 3}, "ending_reversal")
    out = pd.DataFrame(rows, columns=["work_key", "attribute", "option", "human_mean", "n_raters"])
    out["human_mean"] = out.human_mean.astype(float).round(6)
    return out


if __name__ == "__main__":
    f = build_film(); f.to_csv(P("data/validation/newattr_human_means_film.csv"), index=False)
    b = build_book(); b.to_csv(P("data/validation/newattr_human_means_book.csv"), index=False)
    print(f"film: {len(f)} rows, {f.work_key.nunique()} works, attrs={sorted(f.attribute.unique())}")
    print(f"book: {len(b)} rows, {b.work_key.nunique()} works, attrs={sorted(b.attribute.unique())}")
