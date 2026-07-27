"""Regenerate the canonical film-validation table on the deployed-corpus basis.

Each survey film's per-work human ground truth is correlated against the deployed corpus
score for the SAME film, paired through the corrected survey->corpus row map. Every layer
uses the rater-inclusion rule of its originally published validation, so the table is a
like-for-like recomputation on the corrected linkage:

    structure   scalar attributes, all raters            human_means_film.csv
    mood        31 checkbox options, all raters          human_props_film_matched.csv (no_filter)
    texture     37 descriptor options, >=3 engaged        human_props_film_matched.csv (>=3_engaged)
    arc         9 protagonist arc cells, all raters        human_props_film_matched.csv (no_filter)
    arc_change  3 end-minus-beginning deltas, all raters   derived from the arc cells

Metric is Pearson r between the human value (checkbox proportion, scalar mean, or level mean)
and the deployed corpus column, across films with both present (>=10 films, non-degenerate).

Inputs (all committed): data/atlas/century_frame_film.parquet, data/validation/human_means_film.csv,
data/validation/human_props_film_matched.csv, data/validation/survey_to_corpus_map.csv.
Output: data/validation/film_validation_corpus_basis.csv, columns [attribute, layer, r, n].
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/uoft/style_evolves")
V = os.path.join(BASE, "data/validation")

CORPUS = pd.read_parquet(os.path.join(BASE, "data/atlas/century_frame_film.parquet")).reset_index(drop=True)
STRUCT = pd.read_csv(os.path.join(V, "human_means_film.csv"))
STRUCT["survey_movie_id"] = STRUCT.survey_movie_id.astype(str)
MATCH = pd.read_csv(os.path.join(V, "human_props_film_matched.csv"))
MATCH["survey_movie_id"] = MATCH.survey_movie_id.astype(str)
CMAP = pd.read_csv(os.path.join(V, "survey_to_corpus_map.csv"))
CMAP["survey_movie_id"] = CMAP.survey_movie_id.astype(str)
SID2IDX = dict(zip(CMAP[CMAP.corpus_idx >= 0].survey_movie_id, CMAP[CMAP.corpus_idx >= 0].corpus_idx.astype(int)))

MOOD = [f"mood_{m}" for m in ("Adventurous", "Bittersweet", "Bleak", "Chaotic", "Challenging", "Chilling",
        "Cozy", "Dark", "Dreamy", "Eerie", "Energetic", "Epic", "Funny", "Gritty", "Heartwarming", "Hopeful",
        "Inspirational", "Lighthearted", "Melancholic", "Mysterious", "Optimistic", "Quirky", "Raw", "Reflective",
        "Romantic", "Sad", "Surreal", "Tense", "Tragic", "Whimsical", "Wistful")]
TEXTURE = (["visual_Dark", "visual_Saturated", "visual_Muted", "visual_Gritty", "visual_Bright", "visual_Handheld",
        "visual_Slow", "visual_Stylized", "visual_Clean", "visual_Natural", "visual_Busy", "visual_Fast_cut",
        "visual_Sharp", "visual_Dreamy"]
    + ["score_Ominous", "score_Orchestral", "score_Melodic", "score_Rock_n__Roll", "score_Electronic", "score_Grand",
       "score_Dissonant", "score_Jazz", "score_Emotional", "score_Intimate", "score_Choral"]
    + ["acting_Campy", "acting_Brooding", "acting_Deadpan", "acting_Raw", "acting_Manic", "acting_Intense",
       "acting_Physical", "acting_Improvisational", "acting_Natural", "acting_Subtle", "acting_Method", "acting_Theatrical"])
ARC = [f"arc_{d}_{t}" for d in ("likable", "competent", "proactive") for t in ("Beginning", "Middle", "End")]


def pearson(xs, ys):
    """Pearson r over paired lists; NaN if <10 pairs or a side is constant."""
    if len(xs) >= 10 and np.std(xs) > 0 and np.std(ys) > 0:
        return float(np.corrcoef(xs, ys)[0, 1]), len(xs)
    return float("nan"), len(xs)


def corr_human(human_by_sid, attr):
    """Correlate a {survey_movie_id: human_value} map for `attr` against the corpus column."""
    xs, ys = [], []
    for sid, hv in human_by_sid.items():
        idx = SID2IDX.get(sid)
        if idx is None:
            continue
        cv = CORPUS.iloc[idx][attr]
        if pd.notna(cv):
            xs.append(float(cv))
            ys.append(float(hv))
    return pearson(xs, ys)


rows = []
# structure: scalar human means, all raters
for attr in [a for a in STRUCT.attribute.unique() if a in CORPUS.columns]:
    hbs = STRUCT[STRUCT.attribute == attr].set_index("survey_movie_id").human_mean.to_dict()
    r, n = corr_human(hbs, attr)
    rows.append((attr, "structure", r, n))
# mood / texture / arc-level: per-work human truth at the matched rater rules
match_by_attr = {a: g.set_index("survey_movie_id").human_value.to_dict() for a, g in MATCH.groupby("attribute")}
for attr in MOOD:
    r, n = corr_human(match_by_attr.get(attr, {}), attr)
    rows.append((attr, "mood", r, n))
for attr in TEXTURE:
    r, n = corr_human(match_by_attr.get(attr, {}), attr)
    rows.append((attr, "texture", r, n))
for attr in ARC:
    r, n = corr_human(match_by_attr.get(attr, {}), attr)
    rows.append((attr, "arc", r, n))
# arc_change: end-minus-beginning delta, human vs corpus
for dim in ("likable", "competent", "proactive"):
    hb = match_by_attr.get(f"arc_{dim}_Beginning", {})
    he = match_by_attr.get(f"arc_{dim}_End", {})
    xs, ys = [], []
    for sid in sorted(set(hb) & set(he)):
        idx = SID2IDX.get(sid)
        if idx is None:
            continue
        cb, ce = CORPUS.iloc[idx][f"arc_{dim}_Beginning"], CORPUS.iloc[idx][f"arc_{dim}_End"]
        if pd.notna(cb) and pd.notna(ce):
            xs.append(float(ce - cb))
            ys.append(float(he[sid] - hb[sid]))
    r, n = pearson(xs, ys)
    rows.append((f"arc_{dim}_change", "arc_change", r, n))

OUT = pd.DataFrame(rows, columns=["attribute", "layer", "r", "n"])
OUT.to_csv(os.path.join(V, "film_validation_corpus_basis.csv"), index=False)

for layer in ("structure", "mood", "texture", "arc", "arc_change"):
    s = OUT[OUT.layer == layer].dropna(subset=["r"])
    print(f"{layer:11s} n={len(s):2d}  median_r={s.r.median():.3f}  clear>=0.22: {int((s.r >= 0.22).sum())}/{len(s)}")
