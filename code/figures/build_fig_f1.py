#!/usr/bin/env python3
"""FIG F1 — Resource-documentation figure for the Narrative Atlas (NHB Resource paper).
Panel (a): corpus coverage (works per medium x decade, 1890-2020s).
Panel (b): attribute taxonomy (5 layers, film sizes).
Outputs PNG + CSV to results/figures_certified/.
"""
import pandas as pd, numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while ROOT != ROOT.parent and not (ROOT / "data" / "atlas").is_dir():
    ROOT = ROOT.parent
DATA = ROOT / "data" / "atlas"
OUT  = ROOT / "results" / "figures_certified"
OUT.mkdir(parents=True, exist_ok=True)

MEDIA = ["film", "book", "tv"]
TOTALS = {}          # full per-medium totals (all years, incl. nulls dropped for decade table)
rows = []
for m in MEDIA:
    df = pd.read_parquet(DATA / f"century_frame_{m}.parquet", columns=["year"])
    TOTALS[m] = len(df)
    y = df["year"].dropna()
    dec = (np.floor(y / 10) * 10).astype(int)
    vc = dec.value_counts().sort_index()
    for d, c in vc.items():
        rows.append((m, int(d), int(c)))

tbl = pd.DataFrame(rows, columns=["medium", "decade", "n"])
pivot = tbl.pivot(index="decade", columns="medium", values="n").fillna(0).astype(int)

# Figure window: 1890-2020s
decades = list(range(1890, 2030, 10))
cov = pivot.reindex(decades).fillna(0).astype(int)[MEDIA]

# --- taxonomy (film) ---
taxo = pd.DataFrame([
    ("structure & plot", "scalar", 52, "world / character / plot"),
    ("texture",   "descriptor", 37, "visual / score / acting"),
    ("mood",      "mood",       31, "emotional tones"),
    ("genre",     "genre",      18, "genres"),
    ("character arc", "arc",     9, "protagonist change"),
    ("conflict",  "conflict",    6, "conflict types"),
    ("story shape", "shape",     5, "Vonnegut arcs"),
    ("setting",   "setting",     2, "when / where"),
    ("narration", "narration",   1, "narrators (novel)"),
], columns=["layer", "manifest_layer", "n_attrs_film", "descriptor"])

# ---------------- save counts CSV ----------------
cov_out = cov.reset_index().rename(columns={"index": "decade"})
cov_out["decade"] = decades
cov_long = cov_out.melt(id_vars="decade", var_name="medium", value_name="n_works")
cov_long.to_csv(OUT / "FIG_F1_coverage_counts.csv", index=False)
taxo.to_csv(OUT / "FIG_F1_taxonomy_counts.csv", index=False)

# ---------------- plotting ----------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#444444",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "text.color": "#222222",
    "axes.labelcolor": "#222222",
})

# ---------------- fingerprint panel (c): what a scored work looks like ----------------
import re as _re
def _norm(s): return _re.sub(r'[^a-z0-9]', '', str(s).lower())
_S = {m: pd.read_csv(DATA.parent / "corpus" / f"{m}_structural_1890_2025.csv").rename(columns={f"{m}_idx": "id"}) for m in MEDIA}
_MD = {}
for m in MEDIA:
    _a = pd.read_parquet(DATA / f"century_frame_{m}.parquet", columns=["title", "mood_Dark"])
    _MD[m] = {_norm(t): v for t, v in zip(_a.title, _a.mood_Dark)}
WORKS = [("film", "Blade Runner", 1982, "Blade Runner"), ("film", "2001: A Space Odyssey", 1968, "2001"),
         ("tv", "Breaking Bad", None, "Breaking Bad"), ("book", "Pride and Prejudice", None, "Pride & Prejudice"),
         ("book", "The Road", None, "The Road")]
FP_ATTR = [("science_fictional", "science-fictional", 7), ("realistic_was_the_world", "realistic world", 7),
           ("world_building", "world-building", 7), ("character_driven", "character-driven", 7),
           ("competent_was_this_protagonist", "competent lead", 7), ("time_linearity", "non-linear time", 7),
           ("opening_hook", "opening hook", 7), ("__dark__", "dark mood", 100)]
def _scol(df, k):
    mm = [c for c in df.columns if k in c.lower()]; return mm[0] if mm else None
_fp, _flab = [], []
for m, t, y, disp in WORKS:
    d = _S[m]; c = d[d.title.map(_norm) == _norm(t)]
    if y and len(c) > 1 and "year" in c: c = c[c.year == y]
    if not len(c): continue
    r = c.iloc[0]; _flab.append(disp); row = []
    for k, lab, mx in FP_ATTR:
        if k == "__dark__":
            row.append(_MD[m].get(_norm(t), np.nan) / 100.0)
        else:
            col = _scol(d, k); row.append((float(r[col]) - 1) / (mx - 1) if col and not pd.isna(r[col]) else np.nan)
    _fp.append(row)
FPM = np.array(_fp); FPLAB = [a[1] for a in FP_ATTR]

fig = plt.figure(figsize=(7.2, 5.0))
gs = fig.add_gridspec(2, 2, width_ratios=[1.30, 1.0], height_ratios=[1.0, 0.62], wspace=0.55, hspace=0.62,
                      left=0.115, right=0.975, top=0.92, bottom=0.10)

# ---- Panel (a): coverage heatmap ----
axA = fig.add_subplot(gs[0, 0])
# rows = media, cols = decades; color = log10(count+1)
M = cov[MEDIA].T.values.astype(float)         # 3 x n_decades
Mlog = np.log10(M + 1)
cmap = plt.get_cmap("Greys")
im = axA.imshow(Mlog, aspect="auto", cmap=cmap, vmin=0, vmax=np.log10(M.max() + 1))

axA.set_yticks(range(len(MEDIA)))
axA.set_yticklabels(["Film", "Book", "TV"])
xt = list(range(0, len(decades), 2))
axA.set_xticks(xt)
axA.set_xticklabels([f"{decades[i]}s" for i in xt], rotation=45, ha="right")
axA.set_xticks(np.arange(-.5, len(decades), 1), minor=True)
axA.set_yticks(np.arange(-.5, len(MEDIA), 1), minor=True)
axA.grid(which="minor", color="white", linewidth=0.8)
axA.tick_params(which="minor", length=0)
axA.tick_params(which="major", length=2)
for s in axA.spines.values():
    s.set_visible(False)

# annotate counts (only nonzero, readable text color)
thr = Mlog.max() * 0.55
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = int(M[i, j])
        if v == 0:
            continue
        lab = f"{v/1000:.1f}k" if v >= 1000 else f"{v}"
        col = "white" if Mlog[i, j] > thr else "#333333"
        axA.text(j, i, lab, ha="center", va="center", fontsize=5.2, color=col)

axA.set_title("a  Corpus coverage — works per medium × decade",
              fontsize=8.5, loc="left", pad=8, fontweight="bold")

# totals annotation (fig-level, clear of rotated tick labels)
tot_txt = (f"Corpus totals   Film {TOTALS['film']:,}  ·  Book {TOTALS['book']:,}  ·  "
           f"TV {TOTALS['tv']:,}   =   {sum(TOTALS.values()):,} works")
fig.text(0.115, 0.45, tot_txt, fontsize=6.6, color="#555555", ha="left")

# colorbar
cb = fig.colorbar(im, ax=axA, fraction=0.040, pad=0.04)
cb.outline.set_linewidth(0.4)
ticks = [0, 1, 2, 3, 4]
cb.set_ticks(ticks)
cb.set_ticklabels(["1", "10", "100", "1k", "10k"])
cb.ax.tick_params(labelsize=6, length=2)
cb.set_label("works (log)", fontsize=6.4)

# ---- Panel (b): attribute taxonomy ----
axB = fig.add_subplot(gs[0, 1])
tx = taxo.iloc[::-1].reset_index(drop=True)   # largest at top
ypos = np.arange(len(tx))
# restrained mono-hue ramp (dark->light) by size
ramp = plt.get_cmap("Blues_r")(np.linspace(0.12, 0.72, len(tx)))
size_order = tx["n_attrs_film"].rank(ascending=True, method="first").astype(int) - 1
colors = [ramp[i] for i in size_order]

bars = axB.barh(ypos, tx["n_attrs_film"], color=colors, edgecolor="white",
                linewidth=0.5, height=0.72)
axB.set_yticks(ypos)
axB.set_yticklabels([f"{r.layer}" for r in tx.itertuples()], fontsize=8)
axB.set_xlim(0, 108)
for i, r in enumerate(tx.itertuples()):
    axB.text(r.n_attrs_film + 1.6, i + 0.02, f"{r.n_attrs_film}", va="center",
             fontsize=8, color="#333333", fontweight="bold")
    # descriptor in gray, to the right of the count (always readable)
    off = 8 if r.n_attrs_film >= 30 else 7
    axB.text(r.n_attrs_film + off, i, r.descriptor, va="center",
             fontsize=5.8, color="#777777", style="italic")
axB.set_xlabel("attributes (film)", fontsize=7)
axB.tick_params(length=2)
for sp in ["top", "right", "left"]:
    axB.spines[sp].set_visible(False)
axB.spines["bottom"].set_color("#888888")
axB.set_xticks([0, 20, 40, 60])
axB.set_title("b  Attribute taxonomy (161)",
              fontsize=8.5, loc="left", pad=8, fontweight="bold")
axB.text(0.0, -0.32, "per work; layers scored by an LLM viewer panel",
         transform=axB.transAxes, fontsize=6.4, color="#555555")

# ---- Panel (c): fingerprint of scored works ----
axC = fig.add_subplot(gs[1, :])
imC = axC.imshow(FPM, aspect="auto", cmap=plt.get_cmap("RdPu"), vmin=0, vmax=1)
axC.set_xticks(range(len(FPLAB))); axC.set_xticklabels(FPLAB, rotation=32, ha="right", fontsize=6.6)
axC.set_yticks(range(len(_flab))); axC.set_yticklabels(_flab, fontsize=7)
axC.set_xticks(np.arange(-.5, len(FPLAB), 1), minor=True); axC.set_yticks(np.arange(-.5, len(_flab), 1), minor=True)
axC.grid(which="minor", color="white", linewidth=0.8); axC.tick_params(which="minor", length=0); axC.tick_params(which="major", length=2)
for s in axC.spines.values(): s.set_visible(False)
axC.set_title("c  A fingerprint of scored works: five titles across eight attributes",
              fontsize=8.5, loc="left", pad=8, fontweight="bold")
cbC = fig.colorbar(imC, ax=axC, fraction=0.020, pad=0.015); cbC.outline.set_linewidth(0.4)
cbC.set_ticks([0, 1]); cbC.set_ticklabels(["low", "high"]); cbC.ax.tick_params(labelsize=6, length=2)

fig.savefig(OUT / "FIG_F1_resource.png", dpi=300)
fig.savefig(OUT / "FIG_F1_resource.pdf")
print("saved to", OUT, "| fingerprint works:", _flab)
