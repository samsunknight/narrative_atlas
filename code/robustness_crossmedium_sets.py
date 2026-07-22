"""SI robustness (Supplementary Table S5): cross-medium geometry on three
attribute bases. Mirrors the certified method in replication/reproduce.py
(same corpus, same standardization on the un-windowed pool, same >=30 decade
threshold), varying only the attribute set: (1) the eight cross-medium-validated
structural attributes on the shared 1-7 scale, (2) a broader narrative-structure
set (the spine minus craft/texture attributes), (3) the full structural spine.
The 'eight' row reproduces reproduce.py exactly (var-ratio 2.03; bk-tv 1.41/1.17/1.25).
The qualitative pattern (media separate; book-tv converge to a 1990s trough) is
invariant; absolute Euclidean distances scale with the number of attributes.
Run from project root: .venv/bin/python3 code/robustness_crossmedium_sets.py"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")

F = pd.read_csv("data/corpus/film_structural_1890_2025.csv").rename(columns={"film_idx": "id"})
B = pd.read_csv("data/corpus/book_structural_1890_2025.csv").rename(columns={"book_idx": "id"})
T = pd.read_csv("data/corpus/tv_structural_1890_2025.csv").rename(columns={"tv_idx": "id"})
win = lambda d: d[(d.year >= 1915) & (d.year <= 2020)]
ATTRS = [c for c in B.columns if c not in ("id", "title", "year") and c in T.columns and c in F.columns]

VAL8_KEYS = ["science_fictional", "fantastical", "realistic_was_the_world", "world_building",
             "relatable_did_you_find", "competent_was_this_protagonist", "how_many_protagonists", "proactiv"]
VAL8 = [c for c in ATTRS if any(k in c.lower() for k in VAL8_KEYS)]
CRAFT = ["immersive", "visual", "dialogue", "score", "evocative",
         "character_writ", "setting_and_wo", "moved_by", "now_let_s"]
NARR = [c for c in ATTRS if not any(k in c.lower() for k in CRAFT)]   # spine minus craft/texture
SETS = {"eight (cross-medium)": VAL8, "narrative-structure": NARR, "full spine": ATTRS}

def metrics(cols):
    pool = pd.concat([B[cols], T[cols], F[cols]]); mu, sd = pool.mean(), pool.std().replace(0, 1)
    z = lambda df: (df[cols] - mu) / sd
    cent = lambda df, d: (z(df[(df.year >= d) & (df.year < d + 10)]).mean().values
                          if len(df[(df.year >= d) & (df.year < d + 10)]) >= 30 else None)
    cz = {m: {d: cent(df, d) for d in (1950, 1990, 2010)} for m, df in [("bk", B), ("tv", T)]}
    D = lambda d: np.linalg.norm(cz["bk"][d] - cz["tv"][d])
    C = {m: {dec: z(g).mean().values
             for dec, g in win(df).assign(dec=lambda x: (x.year // 10) * 10).groupby("dec") if len(g) >= 30}
         for m, df in [("bk", B), ("fm", F), ("tv", T)]}
    mm = {m: np.mean(list(C[m].values()), 0) for m in C}
    gr = np.mean([v for m in C for v in C[m].values()], 0)
    betw = np.mean([np.mean((mm[m] - gr) ** 2) for m in C])
    wth = np.mean([np.mean([np.mean((c - mm[m]) ** 2) for c in C[m].values()]) for m in C])
    return len(cols), betw / wth, D(1950), D(1990), D(2010)

print(f"{'attribute set':22s} {'n':>3s} {'var-ratio':>9s} {'bk-tv 50s':>9s} {'90s':>6s} {'10s':>6s}  narrows?")
for name, cols in SETS.items():
    n, vr, d50, d90, d10 = metrics(cols)
    print(f"{name:22s} {n:3d} {vr:9.2f} {d50:9.2f} {d90:6.2f} {d10:6.2f}  {'yes' if d90 < d50 else 'no'}")
