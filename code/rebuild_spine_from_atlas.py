"""Rebuild the structural spine for each medium directly from the canonical atlas.

The atlas (data/atlas/century_frame_{m}.parquet) is the validated source of truth for every
structural score; the spine CSVs are the analysis panel the geometry reads. They had drifted:
a 2026-07-28 re-score refreshed the atlas but only five spine columns were carried over, leaving
the rest at pre-rescore values. This rebuilds every spine column's values from the atlas so the two
can never disagree. The "immersive" column had held a mis-collapsed per-setting item; it is now
refilled from the work-level `immersive_setting` score (cross-medium validated, Marginal tier,
codebook "Immersive setting (avg)"), keeping the 36-attribute spine intact with corrected values.

Film and TV map by identical column name (plus the two plot-drive axes, which the atlas prefixes
with the medium). Book uses the human-curated survey_atlas_crosswalk (film_col -> book_col), since
the book atlas names its columns differently (book_Q483_fantastical, not the survey-derived name).
A few texture attributes (visual style, score, some dialogue) are scored for film/TV only; where the
crosswalk supplies no book column, the closest scored book attribute is used, and this is the one
place a cross-medium value rests on an analog rather than the same instrument.
"""
import os, re, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def P(*a): return os.path.join(ROOT, *a)
def strip(s): return re.sub(r"\s*\[newscore\]\s*", "", str(s)).strip()
def norm(s): return re.sub(r"[^a-z0-9]", "", strip(s).lower())

OUT = P("data", "corpus_rebuilt")
os.makedirs(OUT, exist_ok=True)

# ---- book column map: crosswalk film_col(norm) -> book atlas col, plus explicit overrides ----
xw = pd.read_csv(P("data/validation/survey_atlas_crosswalk.csv"))
BA = set(pd.read_parquet(P("data/atlas/century_frame_book.parquet")).columns)
xwbook = {}
for _, r in xw.iterrows():
    if pd.notna(r.film_col) and pd.notna(r.book_col) and "family" not in str(r.film_col):
        xwbook[norm(r.film_col)] = strip(r.book_col)

# explicit book sources for the five newscore additions and the two plot-drive axes
BOOK_OVR = {
    "plot_driven": "book_plot_driven", "character_driven": "book_character_driven",
    "character_development": "book_character_development", "opening_hook": "book_opening_hook",
    "time_linearity": "book_Q622_time_linearity", "plot_linearity": "book_plot_linearity",
    "ending_reversal": "book_ending_reversal",
}
# texture attributes scored for film/TV; nearest scored book attribute where one exists (analog)
BOOK_TEX = {
    "3_how_evocative_did_you_find_the_visual_style": "book_evocative_prose",
    "8_how_engaging_did_you_find_the_dialogue": "book_Q659_dlg_engaging",
    "7_now_let_s_talk_pun_intended_about_dialogue": "book_Q670_imp_dialogue",
}

def book_source(col):
    if col in BOOK_OVR: return BOOK_OVR[col]
    bc = xwbook.get(norm(col))
    if bc and bc in BA: return bc
    if col in BA: return col                      # same-name (likable/competent/proactive, newscore)
    for k, v in BOOK_TEX.items():
        if norm(col).startswith(norm(k)[:22]): return v if v in BA else None
    return None                                   # no book equivalent (side chars, score, realistic-dialogue)

def med_source(col, medium, atlas_cols):
    if "immersive" in col.lower():                # bogus per-setting item -> work-level rescore
        return "immersive_setting" if "immersive_setting" in atlas_cols else None
    if medium == "book": return book_source(col)
    if col in atlas_cols: return col              # film/TV: identical name
    pre = f"{medium}_{col}"
    return pre if pre in atlas_cols else None      # plot_driven -> film_plot_driven

def main():
    report = []
    for m in ["film", "book", "tv"]:
        A = pd.read_parquet(P(f"data/atlas/century_frame_{m}.parquet"))
        acols = set(A.columns)
        S = pd.read_csv(P(f"data/corpus/{m}_structural_1890_2025.csv"))
        idc = f"{m}_idx"
        spine = [c for c in S.columns if c not in (idc, "title", "year")]
        out = S[[idc, "title", "year"]].copy() if "title" in S.columns else S[[idc, "year"]].copy()
        Ai = A.set_index("idx")
        kept, analog, unmapped = 0, [], []
        for c in spine:
            src = med_source(c, m, acols)
            if src is None:
                out[c] = S.set_index(idc)[c].reindex(out[idc].values).values  # keep current (no atlas source)
                unmapped.append(c); continue
            out[c] = Ai[src].reindex(out[idc].values).values
            kept += 1
            if m == "book" and (c in BOOK_TEX or (norm(c) not in xwbook and c not in BOOK_OVR and src != c)):
                analog.append((c, src))
        out.to_csv(P("data", "corpus_rebuilt", f"{m}_structural_1890_2025.csv"), index=False)
        report.append((m, len(spine), kept, len(unmapped), unmapped, analog))
    for m, n, kept, nun, un, an in report:
        print(f"{m}: {n} cols | refreshed {kept} | kept-as-is {nun}: {[c[:28] for c in un]}")
        if an: print(f"    book analogs: {[(c[:24], s) for c, s in an]}")

if __name__ == "__main__":
    main()
