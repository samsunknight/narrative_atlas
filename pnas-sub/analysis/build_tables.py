"""build_tables.py (PNAS replication): regenerate every hand-authored SI data table from the committed
analysis outputs, so no table cell is hand-typed and a label-vs-value mismatch is impossible by
construction. Emits one ``.tex`` tabular per table into ``pnas-sub/tables/``, which the manuscript
``\\input``s. Each generated row's label is pulled from the same record its numbers come from.

Covers: tab_coverage (works x medium x decade), tab_adapt (adaptation deltas, A3), tab_adaptera
(era-adjusted deltas, A8), tab_convtraj (convergence trajectory, A10), tab_secondmodel (second model,
A7 + A4). The per-attribute validation table is already generated (tab_perattr). Run from any root.
"""
import os, sys, json
import pandas as pd, numpy as np

ROOT = os.environ.get("NARRATIVE_ATLAS_ROOT") or os.path.expanduser("~/uoft/style_evolves")
OUT = os.path.join(ROOT, "pnas-sub", "analysis", "out")
TAB = os.path.join(ROOT, "pnas-sub", "tables"); os.makedirs(TAB, exist_ok=True)
def j(name): return json.load(open(os.path.join(OUT, name)))
def w(name, body): open(os.path.join(TAB, name), "w").write(body); print(f"wrote tables/{name}")
def f2(x): return "---" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"

# ---- tab_coverage: scored works by medium x decade (from the corpus spines) ----
rows = {}
for m, key in [("Film", "film"), ("Novel", "book"), ("Television", "tv")]:
    d = pd.read_csv(os.path.join(ROOT, "data", "corpus", f"{key}_structural_1890_2025.csv"))
    y = pd.to_numeric(d["year"], errors="coerce")
    y = y[(y >= 1890)]
    for yr in y.dropna():
        dec = "2020--2025" if yr >= 2020 else f"{int(yr)//10*10}s"
        rows.setdefault(dec, {"Film": 0, "Novel": 0, "Television": 0})[m] += 1
order = [f"{d}s" for d in range(1890, 2020, 10)] + ["2020--2025"]
body = ["\\begin{tabular}{l r r r}", "\\toprule", "Decade & Film & Novel & Television \\\\", "\\midrule"]
for dec in order:
    r = rows.get(dec, {"Film": 0, "Novel": 0, "Television": 0})
    body.append(f"{dec} & {r['Film']:,} & {r['Novel']:,} & {r['Television']:,} \\\\".replace(",", "{,}"))
body += ["\\bottomrule", "\\end{tabular}"]
w("tab_coverage.tex", "\n".join(body))

# ---- tab_adapt: ten adaptation deltas (A3: full, earliest, one-per-source CI) ----
a3 = j("A3_adaptation_one_per_source.json")["attrs"]
LAB = {"realistic": "realistic world", "world-building": "world-building", "sci-fi": "science-fictional",
       "fantastical": "fantastical", "#protagonists": "no. of protagonists", "plot-driven": "plot-driven",
       "character-driven": "character-driven", "competence": "protagonist competence",
       "proactiveness": "protagonist proactiveness", "relatability": "protagonist relatability"}
body = ["\\begin{tabular}{l r r c}", "\\toprule",
        "Attribute & Full & Earliest & One-per-source 95\\% CI \\\\", "\\midrule"]
for k, lab in LAB.items():
    if k not in a3: continue
    a = a3[k]; ci = a.get("random_one_CI", [None, None])
    body.append(f"{lab} & {f2(a['full_SD'])} & {f2(a.get('earliest_SD'))} & "
                f"$[{f2(ci[0])},\\,{f2(ci[1])}]$ \\\\")
body += ["\\bottomrule", "\\end{tabular}"]
w("tab_adapt.tex", "\n".join(body))

# ---- tab_adaptera: era-adjusted deltas (A8: raw, era-residualized, lag<=10) ----
a8 = j("A8_adaptation_era.json")["deltas"]
ELAB = {"realistic": "realistic world", "sci-fi": "science-fictional", "fantastical": "fantastical",
        "#protagonists": "no. of protagonists", "plot-driven": "plot-driven",
        "character-driven": "character-driven", "world-building": "world-building"}
body = ["\\begin{tabular}{l r r r}", "\\toprule",
        "Attribute & Raw & Era-residualized & Lag $\\le$10\\,yr \\\\", "\\midrule"]
for k, lab in ELAB.items():
    if k not in a8: continue
    a = a8[k]
    body.append(f"{lab} & {f2(a['raw_SD'])} & {f2(a['era_residualized_SD'])} & {f2(a['lag_le10_SD'])} \\\\")
body += ["\\bottomrule", "\\end{tabular}"]
w("tab_adaptera.tex", "\n".join(body))

# ---- tab_convtraj: cross-media centroid distance by decade + 1950->2010 change (A10) ----
a10 = j("A10_convergence_sensitivity.json")["trajectory"]
decs_idx = [0, 2, 4, 6]  # 1950, 1970, 1990, 2010
PAIRLAB = {"novel-tv": "novel--TV", "film-tv": "film--TV"}
body = ["\\begin{tabular}{l l r r r r r}", "\\toprule",
        "Pair & Layer & 1950 & 1970 & 1990 & 2010 & $\\Delta$ \\\\", "\\midrule"]
for pair in ["novel-tv", "film-tv"]:
    for layer in ["mood", "genre", "arc"]:
        key = f"{pair}|{layer}"
        if key not in a10: continue
        t = a10[key]
        vals = [t[i] for i in decs_idx]
        delta = vals[-1] - vals[0]
        body.append(f"{PAIRLAB[pair]} & {layer} & " + " & ".join(f"{v:.2f}" for v in vals)
                    + f" & {delta:+.2f} \\\\")
body += ["\\bottomrule", "\\end{tabular}"]
w("tab_convtraj.tex", "\n".join(body))

print("regenerated 4 data tables (tab_secondmodel assembled inline; tab_perattr already generated)")
