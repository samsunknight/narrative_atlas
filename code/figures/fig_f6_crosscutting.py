#!/usr/bin/env python3
"""FIG F6 (NHB Resource, cross-cutting): (a) medium convergence decomposed by layer —
convergence is a MOOD phenomenon; (b) layer->layer predictability (5x5 R^2) — the five
layers are complementary, mood is downstream, arc/genre orthogonal.
Rerun from the package root. Saves PNG + CSV to results/figures_certified/."""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = "results/figures_certified"
os.makedirs(OUT, exist_ok=True)

man = pd.read_csv("data/validation/rescore_manifest.csv")
man = man[man.deploy == True]
LAY = man.set_index('attr_id')['layer'].to_dict()
F = pd.read_parquet("data/atlas/century_frame_film.parquet")
B = pd.read_parquet("data/atlas/century_frame_book.parquet")
T = pd.read_parquet("data/atlas/century_frame_tv.parquet")


def A(df, l):
    return [c for c in df.columns if LAY.get(c) == l]


# ---------- Panel (a): convergence by layer ----------
DECADES = list(range(1950, 2011, 10))


def pz(df, sh):
    return df[sh].rank(pct=True)


def cent(z, df, d):
    m = (df.year >= d) & (df.year < d + 10)
    return z[m.values].mean().values if m.sum() >= 30 else None


def dist(za, dfa, zb, dfb, d):
    ca, cb = cent(za, dfa, d), cent(zb, dfb, d)
    return np.linalg.norm(ca - cb) if ca is not None and cb is not None else np.nan


conv = {}  # (layer, pair) -> {decade: dist}
for layer in ["mood", "genre", "arc"]:
    sh = [c for c in A(F, layer) if c in B.columns and c in T.columns]
    Bz, Fz, Tz = pz(B, sh), pz(F, sh), pz(T, sh)
    for pair, (za, dfa, zb, dfb) in {
        "book-tv": (Bz, B, Tz, T),
        "film-tv": (Fz, F, Tz, T),
    }.items():
        conv[(layer, pair)] = {d: dist(za, dfa, zb, dfb, d) for d in DECADES}

# ---------- Panel (b): layer->layer predictability ----------
lys = ["scalar", "mood", "genre", "arc", "descriptor"]
nm = {"scalar": "structure", "descriptor": "texture"}
cols = {l: A(F, l) for l in lys}
D = F[[c for l in lys for c in cols[l]]].dropna()
Dz = (D - D.mean()) / D.std()
samp = Dz.sample(min(8000, len(Dz)), random_state=0)


def predR2(Xi, Yj):
    Xi = np.hstack([np.ones((len(Xi), 1)), Xi])
    r2s = []
    for k in range(Yj.shape[1]):
        y = Yj[:, k]
        b = lstsq(Xi, y, rcond=None)[0]
        yh = Xi @ b
        ss = ((y - y.mean()) ** 2).sum()
        r2s.append(1 - ((y - yh) ** 2).sum() / ss if ss > 0 else np.nan)
    return np.nanmean(r2s)


R2 = np.full((5, 5), np.nan)
for i, li in enumerate(lys):
    Xi = samp[cols[li]].values
    for j, lj in enumerate(lys):
        if li == lj:
            continue
        R2[i, j] = predR2(Xi, samp[cols[lj]].values)

labels = [nm.get(l, l) for l in lys]

# ---------- Save CSVs ----------
rows = []
for (layer, pair), dd in conv.items():
    for d, v in dd.items():
        rows.append({"panel": "a_convergence", "layer": layer, "pair": pair,
                     "decade": d, "centroid_distance": v})
convdf = pd.DataFrame(rows)
convdf.to_csv(f"{OUT}/FIG_F6_convergence_by_layer.csv", index=False)

r2df = pd.DataFrame(R2, index=[f"from_{l}" for l in labels],
                    columns=[f"predict_{l}" for l in labels])
r2df.to_csv(f"{OUT}/FIG_F6_layer_predictability_R2.csv")

# ---------- Figure ----------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

fig = plt.figure(figsize=(7.2, 3.2), dpi=300)
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.42,
                      left=0.075, right=0.965, top=0.86, bottom=0.155)

# Panel (a)
axA = fig.add_subplot(gs[0, 0])
LCOL = {"mood": "#B4433B", "genre": "#4E79A7", "arc": "#9C9C9C"}
LSTY = {"book-tv": "-", "film-tv": "--"}
LMK = {"book-tv": "o", "film-tv": "^"}
xs = np.array(DECADES)
for layer in ["mood", "genre", "arc"]:
    for pair in ["book-tv", "film-tv"]:
        ys = np.array([conv[(layer, pair)][d] for d in DECADES], float)
        axA.plot(xs, ys, color=LCOL[layer], ls=LSTY[pair], lw=1.4,
                 marker=LMK[pair], ms=3.2, mew=0, alpha=0.95, zorder=3)
axA.axvline(1990, color="0.55", lw=0.7, ls=(0, (2, 2)), zorder=1)
axA.text(1990, axA.get_ylim()[1] * 0.985, "~1990 trough", rotation=0,
         ha="center", va="top", fontsize=6.5, color="0.4")
axA.set_xlabel("Decade")
axA.set_ylabel("Centroid distance between media\n(percentile-z, shared attributes)")
axA.set_title("a  Convergence is a mood phenomenon", loc="left",
              fontsize=8.5, fontweight="bold", pad=6)
axA.set_xticks(range(1950, 2011, 10))
axA.set_xticklabels([f"'{str(d)[2:]}" for d in range(1950, 2011, 10)])
axA.set_ylim(0, None)
axA.margins(x=0.03)
# legends
from matplotlib.lines import Line2D
h_layer = [Line2D([], [], color=LCOL[l], lw=1.6, label=l.capitalize())
           for l in ["mood", "genre", "arc"]]
h_pair = [Line2D([], [], color="0.35", ls=LSTY[p], marker=LMK[p], ms=3.2,
                 mew=0, lw=1.3, label=p) for p in ["book-tv", "film-tv"]]
leg1 = axA.legend(handles=h_layer, loc="upper right", frameon=False,
                  fontsize=6.8, handlelength=1.4, labelspacing=0.25,
                  bbox_to_anchor=(1.0, 1.0))
axA.add_artist(leg1)
axA.legend(handles=h_pair, loc="upper right", frameon=False, fontsize=6.8,
           handlelength=1.9, labelspacing=0.25, bbox_to_anchor=(1.0, 0.74))

# Panel (b)
axB = fig.add_subplot(gs[0, 1])
cmap = plt.get_cmap("YlGnBu")
Rm = np.ma.masked_invalid(R2)
im = axB.imshow(Rm, cmap=cmap, vmin=0, vmax=0.75, aspect="equal")
axB.set_xticks(range(5))
axB.set_yticks(range(5))
axB.set_xticklabels(labels, rotation=35, ha="right")
axB.set_yticklabels(labels)
axB.set_xlabel("predicted layer")
axB.set_ylabel("predictor layer")
axB.set_title("b  Layers are complementary", loc="left",
              fontsize=8.5, fontweight="bold", pad=6)
for i in range(5):
    for j in range(5):
        if np.isnan(R2[i, j]):
            axB.text(j, i, "—", ha="center", va="center",
                     color="0.6", fontsize=8)
        else:
            v = R2[i, j]
            axB.text(j, i, f"{v:.2f}", ha="center", va="center",
                     color="white" if v > 0.42 else "0.15", fontsize=7)
for sp in axB.spines.values():
    sp.set_visible(False)
axB.set_xticks(np.arange(-.5, 5, 1), minor=True)
axB.set_yticks(np.arange(-.5, 5, 1), minor=True)
axB.grid(which="minor", color="white", lw=1.2)
axB.tick_params(which="minor", length=0)
cb = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.03)
cb.set_label("mean $R^2$", fontsize=7)
cb.ax.tick_params(labelsize=6.5, width=0.5, length=2)
cb.outline.set_visible(False)

fig.savefig(f"{OUT}/FIG_F6_crosscutting.png", dpi=300, bbox_inches="tight")
print("SAVED", f"{OUT}/FIG_F6_crosscutting.png")
print("\nPanel (a) convergence (centroid distance):")
for layer in ["mood", "genre", "arc"]:
    for pair in ["book-tv", "film-tv"]:
        dd = conv[(layer, pair)]
        vals = [dd[d] for d in DECADES]
        tr = np.nanargmin(vals)
        print(f"  {layer:5s} {pair:8s}: {vals[0]:.2f}(1950s) -> "
              f"trough {vals[tr]:.2f}@{DECADES[tr]} -> {vals[-1]:.2f}(2010s)")
print("\nPanel (b) R^2 (rows=from, cols=predict):", labels)
print(np.round(R2, 2))
