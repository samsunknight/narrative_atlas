#!/usr/bin/env python3
"""FIG (main) "A century of storytelling" -- 5-panel human-graspable lead.
Decade means 1920-2020 from the released dense atlas (century_frame_{film,book}.parquet):
 (a) plot non-linearity, novel vs film;  (b) internal character development, novel vs film;
 (c) share of films with near-future / distant-future / off-Earth settings;
 (d) fit to Cinderella vs Riches-to-Rags plot shapes (film);
 (e) centrality of the protagonist's conflict with self vs society (film).
Run from ~/uoft/style_evolves/.  Saves results/figures_draft/FIG_century_5panel.png."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.expanduser("~/uoft/style_evolves"))
OUT = "results/figures_draft"
os.makedirs(OUT, exist_ok=True)

F = pd.read_parquet("data/atlas/century_frame_film.parquet")
B = pd.read_parquet("data/atlas/century_frame_book.parquet")
DEC = list(range(1920, 2021, 10))


def dser(d, col, minn=30):
    """Decade mean of `col` over 1920-2020; NaN where a decade has <minn scored works."""
    g = d.assign(dec=(d.year // 10) * 10)
    out = []
    for de in DEC:
        s = g.loc[g.dec == de, col].dropna()
        out.append(s.mean() if len(s) >= minn else np.nan)
    return np.array(out, float)


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.9})
FILM, NOVEL = "#2471a3", "#c0392b"
fig = plt.figure(figsize=(16, 9.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.30, wspace=0.26)
axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[0, 2])
axd = fig.add_subplot(gs[1, 0]); axe = fig.add_subplot(gs[1, 1]); axf = fig.add_subplot(gs[1, 2])


def dressed(ax, title, ylab, xlab=None):
    ax.set_title(title, loc="center", fontweight="bold", fontsize=13, pad=8)
    ax.set_ylabel(ylab)
    if xlab:
        ax.set_xlabel(xlab)
    ax.set_xticks(range(1920, 2021, 20))
    ax.grid(axis="y", alpha=0.12); ax.set_axisbelow(True)


# (a) plot non-linearity: novel leads film
axa.plot(DEC, dser(F, "time_linearity"), color=FILM, lw=2.2, marker="o", ms=6, label="Film")
axa.plot(DEC, dser(B, "book_Q622_time_linearity"), color=NOVEL, lw=2.2, marker="s", ms=6, label="Novel")
dressed(axa, "a  Storytelling grows less linear", "Non-linearity of plot  (1-4)")
axa.legend(loc="upper left", frameon=False, fontsize=10)
axa.annotate("novel leads\n(modernism)", xy=(1930, 1.44), xytext=(1948, 1.34),
             color=NOVEL, fontsize=9.5, ha="left",
             arrowprops=dict(arrowstyle="->", color=NOVEL, lw=1.1))

# (b) internal character development
axb.plot(DEC, dser(F, "character_development"), color=FILM, lw=2.2, marker="o", ms=6, label="Film")
axb.plot(DEC, dser(B, "book_character_development"), color=NOVEL, lw=2.2, marker="s", ms=6, label="Novel")
dressed(axb, "b  Protagonists change more", "Internal change (1-7)")
axb.legend(loc="upper left", frameon=False, fontsize=10)

# (c) speculative settings (film), % of films
axc.plot(DEC, dser(F, "setting_when__in_the_near_future"), color="#7d3c98", lw=2.2, marker="o", ms=6, label="Near future")
axc.plot(DEC, dser(F, "setting_when__in_the_distant_future"), color="#e67e22", lw=2.2, marker="^", ms=6, label="Distant future")
axc.plot(DEC, dser(F, "setting_where__somewhere_off_earth_that_is_imaginary"), color="#229954", lw=2.2, marker="D", ms=5.5, label="Off Earth")
dressed(axc, "c  The speculative frontier widens", "Share of films  (%)")
axc.legend(loc="upper left", frameon=False, fontsize=10)

# (d) story shapes (film)
axd.plot(DEC, dser(F, "shape_cinderella"), color="#5dade2", lw=2.2, marker="o", ms=6, label="Cinderella (rise-fall-rise)")
axd.plot(DEC, dser(F, "shape_riches_to_rags"), color=NOVEL, lw=2.2, marker="s", ms=6, label="Riches to Rags (decline)")
dressed(axd, "d  Story shapes darken", "Arc-fit score  (0-100)", "Decade")
axd.legend(loc="lower left", frameon=False, fontsize=9.5)

# (e) conflict (film)
axe.plot(DEC, dser(F, "conflict_self"), color="#7d3c98", lw=2.2, marker="o", ms=6, label="vs. self (interior)")
axe.plot(DEC, dser(F, "conflict_society"), color="#148f77", lw=2.2, marker="s", ms=6, label="vs. society")
dressed(axe, "e  Conflict turns inward and social", "Conflict centrality  (0-100)", "Decade")
axe.legend(loc="upper left", frameon=False, fontsize=9.5)

# (f) endings grow less resolved: film falls, the novel holds (the down-trend)
axf.plot(DEC, dser(F, "12a_on_a_scale_of_1_entirely_unresolved_to_7_entirely_resolved"), color=FILM, lw=2.2, marker="o", ms=6, label="Film")
axf.plot(DEC, dser(B, "book_Q638_resolved"), color=NOVEL, lw=2.2, marker="s", ms=6, label="Novel")
dressed(axf, "f  Endings grow less resolved", "Ending resolution  (1-7)", "Decade")
axf.legend(loc="lower left", frameon=False, fontsize=10)

fig.suptitle("A century of storytelling  ·  the Narrative Atlas reads the elements of "
             "narrative form and how they moved, 1920–2020",
             fontweight="bold", fontsize=15, y=0.985)
fig.savefig(f"{OUT}/FIG_century_5panel.png", dpi=150, bbox_inches="tight")
print("SAVED", f"{OUT}/FIG_century_5panel.png")
for nm, ax in [("a film time_lin", dser(F, "time_linearity")), ("a novel time_lin", dser(B, "book_Q622_time_linearity")),
               ("b film char_dev", dser(F, "character_development")), ("b novel char_dev", dser(B, "book_character_development")),
               ("c near_future", dser(F, "setting_when__in_the_near_future")),
               ("d cinderella", dser(F, "shape_cinderella")), ("d riches_to_rags", dser(F, "shape_riches_to_rags")),
               ("e conflict_self", dser(F, "conflict_self")), ("e conflict_society", dser(F, "conflict_society"))]:
    print(f"  {nm:20s} 1920={ax[0]:.2f}  2020={ax[-1]:.2f}")
