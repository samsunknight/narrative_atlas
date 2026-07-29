#!/usr/bin/env python3
"""Generate Supplementary Table S1 (the Narrative Atlas attribute dictionary).

Reads the clean codebook data/validation/attribute_dictionary.csv (161 rows,
nine constructs) and emits paper/tab_attribute_dictionary.tex, a longtable that
drops into si.tex. This is the single source of truth for the SI codebook table;
edit the CSV, re-run this, do not hand-edit the .tex.

Run with the project venv:
    "/…/style_evolves/.venv/bin/python" code/build_tab_attribute_dictionary.py
"""
import re
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "validation" / "attribute_dictionary.csv"
OUT = ROOT / "paper" / "tab_attribute_dictionary.tex"

# ---------------------------------------------------------------- escaping ----
def esc(s: str) -> str:
    """Escape LaTeX specials and normalize quotes to LaTeX `` '' / ' form."""
    s = str(s)
    # unicode curly quotes -> LaTeX quote glyphs
    s = s.replace("“", "``").replace("”", "''")   # “ ”
    s = s.replace("‘", "`").replace("’", "'")     # ‘ ’
    # paired straight double quotes -> LaTeX curly quotes
    s = re.sub(r'"([^"]*)"', r"``\1''", s)
    # escape the specials that can appear in CSV text
    for a, b in [("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$")]:
        s = s.replace(a, b)
    return s


def scale_fmt(s: str) -> str:
    """Native response range: drop 'select-all, N options' descriptor (it lives in
    the definition), then normalize numeric-range hyphens to en-dashes."""
    s = re.sub(r"select-all,\s*\d+\s*options\s*", "", str(s))
    return re.sub(r"(\d)-(\d)", r"\1--\2", s)


def fr(x) -> str:
    """Two-decimal correlation string."""
    return f"{float(x):.2f}"


# --------------------------------------------------------------- load data ----
df = pd.read_csv(CSV)
df["film_r"] = pd.to_numeric(df["film_r"], errors="coerce")
df["book_r"] = pd.to_numeric(df["book_r"], errors="coerce")
df["cross_medium"] = df["cross_medium"].astype(bool)

# construct display order (matches main-text Table 1) and gray-header glosses.
CONSTRUCTS = [
    ("structure",     "Structure \\& plot",
     "the world, its characters, and the plot", "{N} attributes; {V} validate in film"),
    ("setting",       "Setting",
     "when and where the story takes place", "{N} attributes; {V} validate, both cross-medium"),
    ("story-shape",   "Story shape",
     "the Vonnegut emotional arc of the plot", "{N} attributes; {V} validate"),
    ("conflict",      "Conflict type",
     "the axis of dramatic opposition", "{N} attributes; {V} validate"),
    ("character-arc", "Character arc",
     "how the protagonist changes", "{N} points; {V} validate on arc change, $r=0.45$--$0.54$"),
    ("narration",     "Narration",
     "who tells the story and how many", "{N} attribute; {V} validates"),
    ("mood",          "Mood",
     "the tones a story evokes", "{N} moods; {V} validate, median $r=0.41$"),
    ("genre",         "Genre",
     "how strongly each genre applies", "{N} genres; {V} validate, median AUC 0.91"),
    ("texture",       "Texture",
     "the visual, musical, and acting grain", "{N} film descriptors; {V} validate, visual median $r\\approx0.44$"),
]

# validate = tier in {A,B} for the tier-graded r constructs; for the fixed-marker
# constructs (mood 28/31, genre 18/18, character-arc 9/9) the count is set here to
# match the per-tone / per-genre / per-arc validation reported in the main text.
FIXED_VALIDATE = {"mood": 28, "genre": 18, "character-arc": 9}


def n_validate(sub: pd.DataFrame, layer: str) -> int:
    if layer in FIXED_VALIDATE:
        return FIXED_VALIDATE[layer]
    return int(sub["tier"].isin(["A", "B"]).sum())


def valid_display(row) -> str:
    """The 'Valid.' cell for one attribute, per construct convention."""
    L = row["layer"]
    if L == "genre":
        return fr(row["film_r"]) + r"$^{\S}$"
    if L == "mood":
        return r"0.41$^{\ddagger}$"
    if L == "character-arc":
        return r"0.45--0.54$^{\dagger}$"
    # r-metric constructs: structure, setting, story-shape, conflict, texture, narration
    film, book = row["film_r"], row["book_r"]
    xm_struct = (L == "structure") and bool(row["cross_medium"])
    marker = r"$^{\P}$" if xm_struct else ""
    if pd.notna(film) and pd.notna(book):
        return f"{fr(film)}{marker}\\,/\\,{fr(book)}"
    if pd.notna(film):
        return f"{fr(film)}{marker}"
    # film missing (narration validates on the book side only)
    return fr(book)


def tier_display(row) -> str:
    return str(row["tier"])  # already 'A'/'B'/'C' or '--'


# ------------------------------------------------------------- build rows ----
lines = []
totals = {}
for layer, name, gloss, cfmt in CONSTRUCTS:
    sub = df[df["layer"] == layer]
    N, V = len(sub), n_validate(sub, layer)
    totals[layer] = (N, V)
    counts = cfmt.format(N=N, V=V)
    header = f"\\textbf{{{name} \\textemdash{{}} {gloss} ({counts})}}"
    lines.append(
        f"\\multicolumn{{5}}{{@{{}}>{{\\columncolor{{gray!12}}}}p{{\\linewidth}}@{{}}}}{{{header}}}\\\\"
    )
    for _, row in sub.iterrows():
        attr = esc(row["attribute"])
        defin = esc(row["definition"])
        scale = scale_fmt(row["scale"])
        lines.append(f"{attr} & {defin} & {scale} & {valid_display(row)} & {tier_display(row)}\\\\")

# ------------------------------------------------------------- assemble -------
total_N = sum(v[0] for v in totals.values())
total_V = sum(v[1] for v in totals.values())
assert total_N == 161, total_N
assert total_V == 150, total_V
n_P = int(((df["layer"] == "structure") & df["cross_medium"]).sum())

count_comment = (
    "% Attribute count by construct: structure 52, setting 2, story-shape 5, "
    "conflict 6, character-arc 9, narration 1, mood 31, genre 18, texture 37 "
    "(total 161 scored / film side; 150 validate)."
)

header_block = rf"""% Supplementary Table S1 --- the narrative-atlas attribute dictionary (all nine constructs).
% Auto-generated by code/build_tab_attribute_dictionary.py from
% data/validation/attribute_dictionary.csv (the certified codebook). Do not hand-edit;
% edit the CSV and re-run the generator.
{count_comment}
{{\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{1.12}}
\begin{{longtable}}{{p{{0.155\linewidth}} p{{0.40\linewidth}} c c l}}
\caption{{Supplementary Table S1. The narrative-atlas attribute dictionary, all nine constructs. The Tier column grades validation strength, namely A (strongest), B (clears the $r>0.22$ validation bar), and C (below the bar, released for cautious use).}}\label{{tab:dict}}\\
\toprule
\textbf{{Attribute}} & \textbf{{Definition / survey question}} & \textbf{{Scale}} & \textbf{{Valid.}} & \textbf{{Tier}}\\
\midrule
\endfirsthead
\multicolumn{{5}}{{l}}{{\emph{{Supplementary Table S1 (continued).}}}}\\
\toprule
\textbf{{Attribute}} & \textbf{{Definition / survey question}} & \textbf{{Scale}} & \textbf{{Valid.}} & \textbf{{Tier}}\\
\midrule
\endhead
\midrule
\multicolumn{{5}}{{r}}{{\emph{{continued on next page}}}}\\
\endfoot
\bottomrule
\endlastfoot"""

notes_block = (
    r"\par\smallskip{\footnotesize\raggedright\noindent\textit{Notes.} One row per "
    r"deployed atlas attribute, on the film/television side of the instrument, grouped "
    r"by construct (161 attributes in all; 52 structure, 2 setting, 5 story-shape, "
    r"6 conflict, 9 character-arc, 1 narration, 31 mood, 18 genre, 37 texture). "
    r"\textbf{Scale} is the native response range the instrument returns. \textbf{Valid.} "
    r"is the validation of the instrument against human judgment: for the structure, "
    r"setting, story-shape, conflict, narration, and texture constructs, the zero-shot "
    r"Pearson correlation $r$ between the instrument's score and the human work-mean "
    r"(film $r$; where a book value is shown it follows the slash and is the reader-survey "
    r"$r$), estimated on the survey samples of several hundred works. The world-type and "
    r"plot attributes ($^{\P}$) reproduce the base-survey values reported in the main-text "
    r"validation table; the remaining structure and texture entries are the correlations "
    r"of the deployed prompts, computed the same way. $^{\S}$ genre is validated as the "
    r"area under the ROC curve against the human-assigned IMDb genre label rather than as "
    r"a correlation, at a median AUC of 0.91 for film (and 0.93 against the reader-survey "
    r"genre question for the novel). $^{\ddagger}$ each mood is validated per tone against "
    r"the survey mood ratings, 28 of 31 clearing $r>0.22$ at a layer median of $r=0.41$ "
    r"(0.48 on the book side); for compactness the table prints this layer median in every "
    r"mood row rather than the individual per-tone value. $^{\dagger}$ the character-arc "
    r"construct validates on the \emph{change} in each trait between the beginning and end "
    r"of the story, at $r=0.45$--$0.54$ (9 of 9); the range printed in every arc row is "
    r"this change-based validation, and the individual beginning/middle/end scores are not "
    r"separately validated. \textbf{Tier} grades validation strength; tier A (strongest) "
    r"and tier B clear the validation bar, while tier C attributes fall below it and are "
    r"disclosed but read with caution. The fifteen structural attributes marked $^{\P}$ "
    r"additionally clear the bar in both film and the reader survey and carry the "
    r"cross-medium geometry; additional non-structural attributes also validate in both "
    r"media (the two settings, the Cinderella, Rags to Riches, and Riches to Rags story "
    r"shapes, and the conflict types), but the cross-medium comparisons rest on the fifteen "
    r"structural attributes, elicited on scales identical across the two media. Across the "
    r"nine constructs 150 of the 161 scored attributes validate.\par}}"
)

out = "\n".join([header_block, *lines, r"\end{longtable}", notes_block]) + "\n"
OUT.write_text(out)

# ------------------------------------------------------------- report ---------
print(f"Wrote {OUT}")
print(f"Total rows: {total_N} / validate {total_V}")
print(f"\\P structural cross-medium rows: {n_P}")
print("Per construct (N / validate):")
for layer, name, _, _ in CONSTRUCTS:
    N, V = totals[layer]
    print(f"  {name:22s} {N:3d} / {V}")
# sanity: number of data rows in output equals 161
data_rows = sum(1 for L in lines if not L.startswith("\\multicolumn"))
print(f"Data (attribute) rows emitted: {data_rows}")
