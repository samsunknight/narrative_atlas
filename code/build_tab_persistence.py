#!/usr/bin/env python3
"""Generate Supplementary Table S2 (trajectory persistence and cyclicity).

For each structural-spine attribute, this computes the persistence autocorrelation
phi from the film decade trajectory, the implied half-life, a drift/revert label, and,
for the few attributes whose reversion takes the form of an observable oscillation, an
implied cycle length omega from a second-order autoregression. Writing the table from the
corpus (rather than transcribing it by hand) keeps every row's label bound to the column
it is computed from, which is what a hand-authored version failed to do.

Run with the project venv:
    "/…/style_evolves/.venv/bin/python" code/build_tab_persistence.py
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus" / "film_structural_1890_2025.csv"
CODEBOOK = ROOT / "data" / "validation" / "attribute_dictionary.csv"
OUT = ROOT / "paper" / "tab_persistence.tex"

LO, HI, MINN = 1915, 2020, 30          # analysis window and per-decade film floor
DRIFT_PHI = 0.90                        # drifts vs reverts: the persistent end sits at phi>=0.90
CYC_MAX_PERIOD = 7.0                    # a cycle is reported only if short enough to recur in the window
CYC_MAX_MOD = 0.90                      # and not a near-unit-root masquerading as a slow cycle

# display labels that read more clearly than the raw codebook attribute string
LABEL_OVERRIDE = {
    "13_how_much_did_you_like_the_score": "musical score",
    "4_how_immersive_did_you_find_this_setting_by_immersive_we_mean_fully_r": "immersive setting",
}


def esc(s: str) -> str:
    return str(s).replace("&", r"\&").replace("#", r"\#").replace("_", r"\_").replace("%", r"\%")


def trajectory(F, col):
    """Standardized decade-mean trajectory of one attribute over the analysis window."""
    d = F[(F.year >= LO) & (F.year <= HI)]
    g = d.assign(dec=(d.year // 10) * 10).groupby("dec")
    dm = g[col].mean()[g.size() >= MINN]
    return ((dm - dm.mean()) / dm.std()).dropna().values


def phi_of(s):
    """Lag-one autocorrelation of the standardized decade trajectory."""
    return float(np.corrcoef(s[:-1], s[1:])[0, 1]) if len(s) >= 7 else np.nan


def ar2_cycle(s):
    """Implied cycle length (decades) and root modulus from an AR(2) fit, when its
    characteristic roots are complex; otherwise None. period = 2*pi / arccos(phi1 / 2*sqrt(-phi2))."""
    if len(s) < 6:
        return None
    y = s[2:]
    X = np.column_stack([np.ones(len(y)), s[1:-1], s[:-2]])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    _, p1, p2 = b
    if not (p1 * p1 + 4 * p2 < 0 and p2 < 0):
        return None
    period = 2 * np.pi / np.arccos(np.clip(p1 / (2 * np.sqrt(-p2)), -1, 1))
    return period, float(np.sqrt(-p2))


def robust_cycle(s):
    """Report omega only for a short cycle that survives dropping either endpoint decade,
    so the oscillation is observable within the century and not an artifact of one decade."""
    full = ar2_cycle(s)
    if full is None:
        return np.nan
    period, mod = full
    if period > CYC_MAX_PERIOD or mod > CYC_MAX_MOD:
        return np.nan
    d1, dL = ar2_cycle(s[1:]), ar2_cycle(s[:-1])
    if not (d1 and dL):
        return np.nan
    if abs(d1[0] - period) / period < 0.5 and abs(dL[0] - period) / period < 0.5:
        return period
    return np.nan


def half_life(phi):
    """Implied half-life in decades, from the two-decimal phi the table prints."""
    p = round(phi, 2)
    return float("inf") if not (0 < p < 1) else -np.log(2) / np.log(p)


F = pd.read_csv(CORPUS).rename(columns={"film_idx": "id"})
cbk = pd.read_csv(CODEBOOK)
label = dict(zip(cbk.column, cbk.attribute))
attrs = [c for c in F.columns if c not in ("id", "title", "year")]

rows = []
for c in attrs:
    s = trajectory(F, c)
    p = phi_of(s)
    if np.isnan(p):
        continue
    rows.append({
        "label": LABEL_OVERRIDE.get(c, label.get(c, c)),
        "phi": p,
        "hl": half_life(p),
        "tendency": "drifts" if p >= DRIFT_PHI else "reverts",
        "omega": robust_cycle(s),
    })
df = pd.DataFrame(rows).sort_values("phi", ascending=False).reset_index(drop=True)

n_drift = int((df.tendency == "drifts").sum())
n_cyc = int(df.omega.notna().sum())

body = []
for _, r in df.iterrows():
    hl = f"{r.hl:.1f}"
    om = f"{r.omega:.1f}" if pd.notna(r.omega) else "--"
    body.append(f"{esc(r.label)} & {r.phi:.2f} & {hl} & {r.tendency} & {om}\\\\")

caption = (
    "Supplementary Table S2. Trajectory persistence and cyclicity of the structural-spine "
    "attributes (film, decade means over 1915--2020). $\\phi$ is the lag-one autocorrelation "
    "of the standardized decade trajectory and the half-life $\\ln(0.5)/\\ln(\\phi)$ (decades) "
    "is the implied time to revert; an attribute is labelled as drifting ($\\phi\\ge0.90$, "
    "half-lives of roughly seven decades or more) or reverting. $\\omega$ is the implied cycle "
    "length in decades from a second-order autoregression with complex roots, reported only for "
    "the attributes whose oscillation is short enough to recur within the century (period at most "
    "seven decades) and stable to dropping either endpoint decade; the remaining attributes show "
    "no such cycle. Both quantities are descriptive summaries of an eleven-point decade series, "
    "not estimates of a stationary process."
)

out = (
    "% Supplementary Table S2 --- trajectory persistence (phi) and cyclicity (omega).\n"
    "% Auto-generated by code/build_tab_persistence.py from the film corpus and codebook.\n"
    "% Do not hand-edit; edit the generator and re-run. Every row's label is bound to the\n"
    "% column its phi/omega are computed from, which a hand-authored table cannot guarantee.\n"
    f"% {n_drift} drift, {len(df) - n_drift} revert; {n_cyc} carry a stable short cycle.\n"
    "\\begin{table}[H]\\centering\n"
    f"\\caption*{{{caption}}}\n"
    "\\begin{tabular}{lrrlr}\\hline Attribute & $\\phi$ & half-life (dec.) & tendency & $\\omega$ (dec.)\\\\\\hline\n"
    + "\n".join(body)
    + "\n\\hline\\end{tabular}\\end{table}\n"
)
OUT.write_text(out)
print(f"Wrote {OUT}")
print(f"{len(df)} attributes | {n_drift} drift / {len(df)-n_drift} revert | {n_cyc} with a stable cycle")
print(df.to_string(index=False))
