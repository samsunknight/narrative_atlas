"""A5 (referee response): two additions the report is right to demand.

(1) ABSOLUTE variance explained (Comment 7). The headline 1.66 is a ratio of between- to
    within-medium *centroid* variance; it does not say what fraction of *work-level* variance
    medium accounts for. We report the multivariate eta-squared (between-group SS / total SS)
    for medium and for decade on the same 16-attribute standardized basis and window as A2, plus
    medium after partialling out decade. This answers "does medium explain 0.2%, 2%, or 20%?".

(2) DIMENSION-NORMALIZED convergence (Comment 6). A4's centroid distance is a raw Euclidean norm
    over layers of unequal size (mood/genre/arc), and an L2 norm grows with dimension. We divide
    each layer's distance by sqrt(n_attrs) -> root-mean-square per-attribute distance, and recompute
    the mood-minus-genre and mood-minus-arc contrasts with the same paired bootstrap. If mood still
    converges more once every layer is on a per-attribute footing, the result is not a dimension
    artifact.

Self-checks: (i) raw A4 contrasts must reproduce the stored JSON; (ii) A2 pipeline is mirrored
exactly for the eta-squared basis. Bounded, single-threaded. Run from ~/uoft/style_evolves/.
"""
import os, json
import numpy as np, pandas as pd

ROOT = os.environ.get("NARRATIVE_ATLAS_ROOT", os.path.expanduser("~/uoft/style_evolves"))
def P(*a): return os.path.join(ROOT, *a)

# ---------- (1) absolute variance explained: mirror A2's canonical pipeline exactly ----------
F = pd.read_csv(P("data/corpus/film_structural_1890_2025.csv")).rename(columns={"film_idx": "id"})
B = pd.read_csv(P("data/corpus/book_structural_1890_2025.csv")).rename(columns={"book_idx": "id"})
T = pd.read_csv(P("data/corpus/tv_structural_1890_2025.csv")).rename(columns={"tv_idx": "id"})
import sys; sys.path.insert(0, P("code"))
from _methods import VAL16_KEYS as VALCM_KEYS
VALCM = [c for c in B.columns if any(k in c.lower() for k in VALCM_KEYS) and c in T.columns and c in F.columns]
assert len(VALCM) == 16, f"expected 16-attr basis, got {len(VALCM)}"

mu = pd.concat([B[VALCM], T[VALCM], F[VALCM]]).mean()
sd = pd.concat([B[VALCM], T[VALCM], F[VALCM]]).std().replace(0, 1)
def prep(df, m):
    d = df[(df.year >= 1915) & (df.year <= 2020)].copy()
    d["dec"] = (d["year"] // 10) * 10; d["m"] = m
    return d[["m", "dec"] + VALCM]
D = pd.concat([prep(B, "bk"), prep(F, "fm"), prep(T, "tv")], ignore_index=True)
Z = ((D[VALCM] - mu) / sd)                       # NaN kept
med = D["m"].values; dec = D["dec"].values

def eta2(zmat, groups):
    """SS-weighted multivariate eta^2 across attributes, NaN-aware (per-attribute between/total SS)."""
    ss_b = ss_t = 0.0
    for j in range(zmat.shape[1]):
        col = zmat[:, j]; ok = ~np.isnan(col)
        x = col[ok]; g = groups[ok]
        if len(x) < 2: continue
        gm = x.mean()
        ss_t += np.sum((x - gm) ** 2)
        for lev in np.unique(g):
            xi = x[g == lev]
            ss_b += len(xi) * (xi.mean() - gm) ** 2
    return ss_b / ss_t if ss_t > 0 else np.nan

Zv = Z.values
eta_med = eta2(Zv, med)
eta_dec = eta2(Zv, dec)
# medium after partialling decade: residualize each attribute on its decade mean, then eta^2 of medium
Zr = Z.copy()
for d in np.unique(dec):
    sel = dec == d
    Zr.loc[Zr.index[sel]] = (Z.loc[Z.index[sel]] - Z.loc[Z.index[sel]].mean())
eta_med_within_dec = eta2(Zr.values, med)
print("=== (1) absolute variance explained, 16-attr standardized basis, 1915-2020 ===")
print(f"  medium eta^2                 : {eta_med:.4f}  ({100*eta_med:.1f}% of work-level variance)")
print(f"  decade eta^2                 : {eta_dec:.4f}  ({100*eta_dec:.1f}%)")
print(f"  medium eta^2 | within decade : {eta_med_within_dec:.4f}  ({100*eta_med_within_dec:.1f}%)")
print(f"  ratio medium/decade eta^2    : {eta_med/eta_dec:.2f}")

# ---------- (2) dimension-normalized convergence: reuse A4 transform ----------
os.chdir(os.environ.get("NARRATIVE_ATLAS_ROOT", os.path.expanduser("~/uoft/style_evolves")))
man = pd.read_csv("data/validation/rescore_manifest.csv"); man = man[man.deploy == True]
LAY = man.set_index("attr_id")["layer"].to_dict()
FR = {m: pd.read_parquet(f"data/atlas/century_frame_{m}.parquet") for m in ["film", "book", "tv"]}
LAYERS = ["mood", "genre", "arc"]; PAIRS = {"novel-tv": ("book", "tv"), "film-tv": ("film", "tv")}
DECS = list(range(1950, 2011, 10)); MIN = 30
def attrs(df, L): return [c for c in df.columns if LAY.get(c) == L]
SH = {L: [c for c in attrs(FR["film"], L) if c in FR["book"].columns and c in FR["tv"].columns] for L in LAYERS}
NL = {L: len(SH[L]) for L in LAYERS}
print(f"\n=== (2) layer sizes: mood={NL['mood']}, genre={NL['genre']}, arc={NL['arc']} ===")
PZ = {m: {L: FR[m][SH[L]].rank(pct=True).values for L in LAYERS} for m in FR}
YR = {m: FR[m]["year"].values for m in FR}
CELL = {m: {d: np.where((YR[m] >= d) & (YR[m] < d + 10))[0] for d in DECS} for m in FR}

def dist(a, b, L, d, idx, norm):
    ia, ib = idx[a][d], idx[b][d]
    if len(ia) < MIN or len(ib) < MIN: return np.nan
    ca = np.nanmean(PZ[a][L][ia], axis=0); cb = np.nanmean(PZ[b][L][ib], axis=0)
    raw = np.linalg.norm(ca - cb)
    return raw / np.sqrt(NL[L]) if norm else raw     # /sqrt(n) => RMS per-attribute distance

def conv(idx, norm):
    out = {}
    for pn, (a, b) in PAIRS.items():
        for L in LAYERS:
            d50 = dist(a, b, L, 1950, idx, norm); d10 = dist(a, b, L, 2010, idx, norm)
            out[(pn, L)] = d50 - d10
    return out

# self-check: raw contrasts must match stored A4 JSON (novel-tv mood-minus-genre full = 0.311)
raw_pt = conv(CELL, norm=False)
chk = raw_pt[("novel-tv", "mood")] - raw_pt[("novel-tv", "genre")]
stored = json.load(open("pnas-sub/analysis/out/A4_convergence_contrasts.json"))["contrasts"]["novel-tv|mood-minus-genre|full"]["diff"]
print(f"  self-check raw novel-tv mood-minus-genre (full): {chk:.3f}  [A4 stored {stored}]  "
      f"{'OK' if abs(chk-stored)<0.01 else 'MISMATCH'}")

rng = np.random.RandomState(0); REPS = 500
def resample():
    return {m: {d: (rng.choice(CELL[m][d], len(CELL[m][d]), replace=True) if len(CELL[m][d]) else CELL[m][d])
                for d in DECS} for m in FR}
boot = [conv(resample(), norm=True) for _ in range(REPS)]
def ci(vals):
    v = np.array([x for x in vals if not np.isnan(x)])
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

norm_pt = conv(CELL, norm=True)
print("\n=== dimension-normalized convergence (RMS per-attribute, 1950->2010) ===")
res = {"layer_sizes": NL, "eta2": {"medium": round(float(eta_med), 4), "decade": round(float(eta_dec), 4),
       "medium_within_decade": round(float(eta_med_within_dec), 4)}, "normed_conv": {}, "normed_contrasts": {}}
for pn in PAIRS:
    for L in LAYERS:
        res["normed_conv"][f"{pn}|{L}"] = round(float(norm_pt[(pn, L)]), 4)
        print(f"  {pn:9s} {L:6s}: Δ(1950->2010) per-attribute = {norm_pt[(pn,L)]:+.4f}")
print("\n=== normalized contrasts: mood converges more per attribute? ===")
for pn in PAIRS:
    for other in ["genre", "arc"]:
        diff = norm_pt[(pn, "mood")] - norm_pt[(pn, other)]
        dci = ci([bt[(pn, "mood")] - bt[(pn, other)] for bt in boot])
        sig = "*" if (dci[0] > 0 or dci[1] < 0) else " "
        res["normed_contrasts"][f"{pn}|mood-minus-{other}"] = {"diff": round(float(diff), 4), "CI": dci, "excludes0": sig == "*"}
        print(f"  {pn:9s} mood - {other:5s}: {diff:+.4f}  CI{dci} {sig}")

json.dump(res, open("pnas-sub/analysis/out/A5_variance_explained_and_normed_convergence.json", "w"), indent=2)
print("\nwrote out/A5_variance_explained_and_normed_convergence.json")
