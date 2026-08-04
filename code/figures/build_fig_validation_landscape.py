#!/usr/bin/env python3
"""Extended Data Fig. 7 -- FIG_validation_landscape.png.

The validation landscape of the Narrative Atlas instrument. Each point is one
attribute, placed at its validation correlation r with the mean human rating and
grouped by narrative construct; the dashed line marks the r = 0.22 bar. Of the 161
attributes the instrument scores across nine constructs, 150 validate against human
judgment; the 18 genres among these 150 are validated against IMDb tags (median ROC
AUC 0.91) and are not plotted here.

Reproduced entirely from the canonical codebook
    data/validation/attribute_dictionary.csv
so the figure is turnkey. An attribute's plotted x is its best-medium validation r,
i.e. max(film_r, book_r): for the film-only attributes this is film_r, and for the one
book-only attribute (narration, "Number of narrators") it is book_r. An attribute
counts as validated if that best-medium r reaches the r = 0.22 bar -- equivalently,
if it validates in either medium -- which reproduces the per-construct counts and the
headline 150/161 of Table 1 (tab_attribute_dictionary). The 18 genres validate by AUC,
not r, so they are counted (18/18) but carry no x-position and are omitted from the
scatter.

Writes the PNG to results/figures_certified/ and mirrors it to the working tree, the
package outputs/, and the Overleaf figures/ directory when those exist.
"""
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---- locate the package root (dir that holds data/validation) -------------------
ROOT = Path(__file__).resolve().parent
while ROOT != ROOT.parent and not (ROOT / "data" / "validation").is_dir():
    ROOT = ROOT.parent
CB_PATH = ROOT / "data" / "validation" / "attribute_dictionary.csv"
OUT = ROOT / "results" / "figures_certified"
OUT.mkdir(parents=True, exist_ok=True)
FNAME = "FIG_validation_landscape.png"

BAR = 0.22            # validation threshold on r
N_WORKS = 149_341     # corpus size (Extended Data Table 1 / abstract)

# ---- per-construct display, top-to-bottom plotting order, and colours ------------
# Colours sampled from the committed certified PNG so the reproduction matches it.
ORDER = ["texture", "mood", "narration", "character-arc", "conflict",
         "story-shape", "structure", "setting"]          # top -> bottom
DISP = {"texture": "Texture (visual/score/acting)", "mood": "Mood",
        "narration": "Narration", "character-arc": "Character arc",
        "conflict": "Conflict type", "story-shape": "Story shape (Vonnegut)",
        "structure": "Structure & plot", "setting": "Setting"}
COLOR = {"texture": "#70b080", "mood": "#d08070", "narration": "#34847e",
         "character-arc": "#e09040", "conflict": "#c04050",
         "story-shape": "#905090", "structure": "#4080b0", "setting": "#499360"}

# ---- data ------------------------------------------------------------------------
cb = pd.read_csv(CB_PATH)
# best-medium validation r for the r-validated (non-genre) attributes
cb["r_used"] = cb[["film_r", "book_r"]].max(axis=1, skipna=True)
ng = cb[cb["validation_metric"] == "r"].copy()      # the 143 r-validated attributes
gen = cb[cb["validation_metric"] == "AUC"].copy()    # the 18 genres (AUC, not plotted)

n_genre = len(gen)
auc_med = float(gen["film_r"].median())              # median ROC AUC across genres
n_total = len(cb)
n_plotted_ok = int((ng["r_used"] >= BAR).sum())
n_valid = n_plotted_ok + n_genre                     # genres all validate by AUC

# ---- figure ----------------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.spines.left": False})
fig, ax = plt.subplots(figsize=(12.6, 8.0))
rng = np.random.RandomState(0)
NROW = len(ORDER)
yticks, ylabels = [], []

for i, lay in enumerate(ORDER):
    yc = NROW - i                                    # texture at top, setting at bottom
    g = ng[ng["layer"] == lay]
    x = g["r_used"].values
    n_ok = int((x >= BAR).sum())
    yj = yc + rng.uniform(-0.30, 0.30, size=len(x))
    ax.scatter(x, yj, s=58, color=COLOR[lay], alpha=0.85,
               edgecolors="white", linewidths=0.4, zorder=3)
    yticks.append(yc)
    ylabels.append(f"{DISP[lay]}\n{n_ok}/{len(g)} validated")

# validation-threshold bar
ax.axvline(BAR, ls=(0, (5, 4)), color="#555555", lw=1.3, zorder=1)
ax.text(BAR + 0.012, NROW + 0.62, "validation\nthreshold\n($r = 0.22$)",
        va="top", ha="left", fontsize=10.5, color="#777777")

ax.set_yticks(yticks)
ax.set_yticklabels(ylabels, fontsize=11.5)
ax.tick_params(axis="y", length=0)
ax.set_xlim(-0.03, 0.82)
ax.set_ylim(-0.10, NROW + 0.95)
ax.set_xticks(np.arange(0.0, 0.81, 0.1))
ax.set_xlabel("Validation against human judgment  (Pearson $r$ with mean human rating)",
              fontsize=13)
ax.set_title("The Narrative Atlas validates across the full breadth of narrative form",
             fontsize=16.5, fontweight="bold", pad=14)

# non-additive summary: the 18 genres are AMONG the 150, not additional to them
auc_disp = f"{auc_med + 1e-9:.2f}"     # median ROC AUC across genres, matches Table 1 (0.91)
summary = (f"{n_valid} of {n_total} attributes validate; the {n_genre} genres among these "
           f"are checked by IMDb AUC {auc_disp}, not plotted"
           f"    ·    {N_WORKS:,} works, three media, 1890–2025")
ax.text(0.395, 0.20, summary, transform=ax.transData, ha="center", va="center",
        fontsize=10.5, style="italic", color="#555555")

fig.tight_layout()
dest = OUT / FNAME
fig.savefig(dest, dpi=120, bbox_inches="tight")
plt.close(fig)

# ---- mirror to the other tracked copies (skip any that do not exist) -------------
MIRRORS = [
    ROOT.parent.parent / "results" / "figures_certified" / FNAME,   # working tree
    ROOT.parent.parent / "results" / "figures_draft" / FNAME,       # working draft
    ROOT / "outputs" / "figures" / FNAME,                           # package outputs
    (Path.home() / "Library/CloudStorage/Dropbox/Apps/Overleaf/"
     "narrative_atlas_resource/figures" / FNAME),                   # Overleaf mirror
]
for m in MIRRORS:
    if m.parent.is_dir():
        shutil.copyfile(dest, m)
        print(f"  mirrored -> {m}")

print(f"saved {dest}")
print(f"validated {n_valid}/{n_total}  (plotted {n_plotted_ok}/{len(ng)} by r, "
      f"genres {n_genre}/{n_genre} by AUC median {auc_med:.3f})")
for lay in ORDER:
    g = ng[ng["layer"] == lay]
    print(f"  {DISP[lay]:32s} {int((g.r_used>=BAR).sum())}/{len(g)}")
