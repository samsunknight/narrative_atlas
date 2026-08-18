"""A6 (referee response, Comment 4/8): is the medium signature just a summary-writing convention?

The medium-label swap only tests the model's response to the stated label. The stronger worry is
that film / novel / television plot summaries are written to different editorial conventions
(length, event density, named-entity load, dialogue), and that those surface differences drive the
signature. We build summary-form features from the ACTUAL Wikipedia text and show:

  (a) ADAPTATION deltas survive them. For each novel-to-film pair we regress the film-minus-novel
      attribute delta on the film-minus-novel deltas in summary length, sentence length, named-entity
      density, dialogue density, and comma density, with source-clustered SEs. The intercept is the
      attribute delta net of every summary-form difference. If it keeps sign and magnitude, the
      signature is not a summary-convention artifact.

  (b) GEOMETRY survives them. We residualize each of the 16 structural attributes on the same
      work-level summary-form features and recompute the medium-vs-era variance ratio. If it stays
      near 1.66, the medium separation is not driven by summary form.

  (c) HOW MUCH medium signal the surface carries. A logistic classifier on the surface features alone
      predicts medium above chance (as expected: the three media are summarized differently) --- which
      is exactly why (a) and (b) control for it. The point is that the narrative-attribute signature is
      not reducible to that surface signal.

Self-checks: raw adaptation deltas match A3; raw ratio rounds to 1.66. No LLM calls. Run from
~/uoft/style_evolves/.
"""
import os, re, json
import numpy as np, pandas as pd

ROOT = os.environ.get("NARRATIVE_ATLAS_ROOT", os.path.expanduser("~/uoft/style_evolves")); os.chdir(ROOT)
def P(*a): return os.path.join(ROOT, *a)
def norm(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())

# ---------- summary-form features from the raw Wikipedia plot text ----------
CAP = re.compile(r"\b[A-Z][a-z]{2,}")            # capitalized token -> named-entity / character proxy
def feats(text):
    t = str(text)
    w = re.findall(r"[A-Za-z']+", t)
    nw = len(w) or 1
    nsent = max(len(re.findall(r"[.!?]+", t)), 1)
    named = len(set(CAP.findall(t)))
    dq = t.count('"') + t.count("“") + t.count("”")
    return pd.Series({"lwords": np.log(nw), "sentlen": nw / nsent,
                      "ne_density": named / nw, "dlg_density": dq / nw, "comma_density": t.count(",") / nw})

def load_feats(path, m):
    """Summary-form features per work. When the raw plot text is present, compute the features and cache
    them as derived numbers (which carry no text and are redistributable); otherwise read the shipped
    feature cache, so this control reproduces without redistributing the raw Wikipedia text."""
    cache = P("data", "validation", f"summary_features_{m}.csv")
    if not os.path.exists(P(path)):
        return pd.read_csv(cache)
    d = pd.read_parquet(P(path))
    ff = d["text"].apply(feats)
    ff["nt"] = d["title"].map(norm); ff["m"] = m
    ff = ff.dropna(subset=["nt"]).drop_duplicates("nt")
    os.makedirs(os.path.dirname(cache), exist_ok=True); ff.to_csv(cache, index=False)
    return ff

FT = load_feats("data/film_wiki_text.parquet", "fm")
BT = load_feats("data/book_wiki_text_century.parquet", "bk")
TT = load_feats("data/tv_wiki_text_century.parquet", "tv")
FCOLS = ["lwords", "sentlen", "ne_density", "dlg_density", "comma_density"]
print("=== surface-feature medium means (films vs novels vs tv) ===")
for c in FCOLS:
    print(f"  {c:14s}: film {FT[c].mean():.3f}  novel {BT[c].mean():.3f}  tv {TT[c].mean():.3f}")

# ---------- corpus scores + A3 pair matching (reuse exactly) ----------
F = pd.read_csv(P("data/corpus/film_structural_1890_2025.csv")).rename(columns={"film_idx": "id"})
B = pd.read_csv(P("data/corpus/book_structural_1890_2025.csv")).rename(columns={"book_idx": "id"})
ATTRS = [c for c in F.columns if c not in ("id", "title", "year")]
SM = {}
for c in ATTRS:
    for k, n in [("science_fictional","sci-fi"),("fantastical","fantastical"),("realistic_was_the_world","realistic"),
                 ("world_building","world-building"),("how_many_protagonists","#protagonists"),("competent","competence"),
                 ("proactive","proactiveness"),("relatable","relatability"),("plot_driven","plot-driven"),("character_driven","character-driven")]:
        if k in c: SM[n] = c
PAIRS = pd.read_csv(P("data/matched/adaptation_pairs.csv")).dropna(subset=["filmLabel", "bookLabel"])
Fn = F.assign(nt=F.title.map(norm)).drop_duplicates("nt").set_index("nt")
Bn = B.assign(nt=B.title.map(norm)).drop_duplicates("nt").set_index("nt")
FTi = FT.set_index("nt"); BTi = BT.set_index("nt")
pooled_sd = {n: pd.concat([F[a], B[a]]).astype(float).std() for n, a in SM.items()}
xs = {n: np.sign(F[a].mean() - B[a].mean()) for n, a in SM.items()}

rows = []
for _, r in PAIRS.iterrows():
    ft, bt = norm(r.filmLabel), norm(r.bookLabel)
    if ft in Fn.index and bt in Bn.index and ft in FTi.index and bt in BTi.index:
        d = {n: (Fn.loc[ft][a] - Bn.loc[bt][a]) / pooled_sd[n] for n, a in SM.items()}   # SD units
        for c in FCOLS:
            d["d_" + c] = FTi.loc[ft][c] - BTi.loc[bt][c]                                  # film-minus-novel surface delta
        d["_src"] = bt
        rows.append(d)
AD = pd.DataFrame(rows)
print(f"\npairs with summary text on both sides: {len(AD)} from {AD._src.nunique()} sources")

def clustered_ols(y, Xcols, cluster):
    """OLS with an intercept; CR1 cluster-robust SEs. Returns intercept coef + SE (and raw mean)."""
    X = np.column_stack([np.ones(len(y))] + [AD[c].values for c in Xcols])
    yv = y.values.astype(float); ok = np.isfinite(yv) & np.all(np.isfinite(X), axis=1)
    X, yv, cl = X[ok], yv[ok], cluster[ok]
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ yv
    resid = yv - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(cl):
        Xg = X[cl == g]; ug = resid[cl == g]
        meat += Xg.T @ np.outer(ug, ug) @ Xg
    G = len(np.unique(cl)); adj = G / (G - 1)
    V = adj * XtX_inv @ meat @ XtX_inv
    return beta[0], np.sqrt(V[0, 0])

print("\n=== (a) adaptation film-minus-novel deltas, RAW vs residualized on summary form ===")
print(f"{'attr':15s} {'raw Δ(SD)':>10s} {'adj Δ(SD)':>10s} {'adj SE':>8s} {'|adj/raw|':>10s}  sign-kept")
res = {"n_pairs": int(len(AD)), "n_sources": int(AD._src.nunique()), "adapt": {}}
cl = AD["_src"].values
for n in SM:
    raw = AD[n].mean()
    adj, se = clustered_ols(AD[n], ["d_" + c for c in FCOLS], cl)
    keep = np.sign(adj) == np.sign(raw)
    frac = abs(adj / raw) if raw else np.nan
    print(f"{n:15s} {raw:+10.2f} {adj:+10.2f} {se:8.2f} {frac:10.2f}  {'yes' if keep else 'FLIP'}")
    res["adapt"][n] = {"raw_SD": round(float(raw), 3), "adj_SD": round(float(adj), 3),
                       "adj_SE": round(float(se), 3), "sign_kept": bool(keep)}

# ---------- (b) geometry: variance ratio after residualizing attributes on summary form ----------
import sys; sys.path.insert(0, P("code"))
from _methods import VAL16_KEYS as VALCM_KEYS
Tc = pd.read_csv(P("data/corpus/tv_structural_1890_2025.csv")).rename(columns={"tv_idx": "id"})
VALCM = [c for c in B.columns if any(k in c.lower() for k in VALCM_KEYS) and c in Tc.columns and c in F.columns]
assert len(VALCM) == 16

def attach_feats(df, FTbl):
    d = df.assign(nt=df.title.map(norm)).merge(FTbl[FCOLS + ["nt"]], on="nt", how="left")
    return d
Fa, Ba, Ta = attach_feats(F, FT), attach_feats(B, BT), attach_feats(Tc, TT)

def ratio(frames, resid_on_form):
    parts = []
    for df, m in frames:
        d = df[(df.year >= 1915) & (df.year <= 2020)].copy(); d["m"] = m; d["dec"] = (d.year // 10) * 10
        parts.append(d)
    D = pd.concat(parts, ignore_index=True)
    Z = D[VALCM].astype(float)
    mu, sd = Z.mean(), Z.std().replace(0, 1)
    Z = (Z - mu) / sd
    if resid_on_form:
        Xf = D[FCOLS].astype(float)
        Xf = (Xf - Xf.mean()) / Xf.std().replace(0, 1)
        Xf = Xf.fillna(0.0); Xm = np.column_stack([np.ones(len(Xf)), Xf.values])
        for a in VALCM:
            y = Z[a].values; ok = np.isfinite(y)
            b = np.linalg.pinv(Xm[ok].T @ Xm[ok]) @ Xm[ok].T @ y[ok]
            Z.loc[Z.index[ok], a] = y[ok] - Xm[ok] @ b            # residual (medium-form removed)
    med, dec = D["m"].values, D["dec"].values
    DECS = list(range(1910, 2021, 10)); MIN = 30
    C = {m: {} for m in ["bk", "fm", "tv"]}
    Zv = Z.values
    for m in C:
        for d in DECS:
            sel = (med == m) & (dec == d)
            if sel.sum() >= MIN: C[m][d] = np.nanmean(Zv[sel], axis=0)
    C = {m: v for m, v in C.items() if v}
    medmean = {m: np.mean(list(v.values()), axis=0) for m, v in C.items()}
    grand = np.mean([c for m in C for c in C[m].values()], axis=0)
    between = np.mean([np.mean((medmean[m] - grand) ** 2) for m in C])
    within = np.mean([np.mean([np.mean((c - medmean[m]) ** 2) for c in C[m].values()]) for m in C])
    return between / within

frames = [(Fa, "fm"), (Ba, "bk"), (Ta, "tv")]
r_raw = ratio(frames, False); r_adj = ratio(frames, True)
print(f"\n=== (b) medium-vs-era variance ratio ===")
print(f"  raw (self-check ~1.66)          : {r_raw:.3f}")
print(f"  after residualizing on summary form: {r_adj:.3f}")
res["ratio_raw"] = round(float(r_raw), 3); res["ratio_resid_on_form"] = round(float(r_adj), 3)

# ---------- (c) how much medium signal do surface features carry? ----------
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    S = pd.concat([FT, BT, TT], ignore_index=True)
    Xs = ((S[FCOLS] - S[FCOLS].mean()) / S[FCOLS].std()).fillna(0.0).values
    ys = S["m"].values
    acc = cross_val_score(LogisticRegression(max_iter=500, multi_class="multinomial"),
                          Xs, ys, cv=5, scoring="balanced_accuracy").mean()
    print(f"\n=== (c) medium predictable from surface features alone: balanced acc {acc:.3f} (chance 0.333) ===")
    res["surface_medium_balanced_acc"] = round(float(acc), 3)
except Exception as ex:
    print("skimpy sklearn:", ex)

json.dump(res, open(P("pnas-sub/analysis/out/A6_summary_convention_controls.json"), "w"), indent=2)
print("\nwrote out/A6_summary_convention_controls.json")
