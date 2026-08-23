"""A19 (referee response): rerun the century fingerprint for television restricted to SERIES only, to
show the large television historical shifts are not an artifact of the changing unit-type mixture.

A07 estimates each attribute's standardized early(1950s-60s)->late(2000s-10s) shift per medium on the
robust-core history attributes. Television is the medium with the largest shifts and the most
composition change (series, episodes, television films in shifting proportions). A15 already showed the
cross-media convergence survives series-only; this module applies the same control to the one-medium
fingerprint. We recompute A07's television shifts on the full television corpus and on series-only
(unit type held constant across decades, so any surviving shift is within-series, not a mixture
effect), using the identical estimator, window, endpoints, and bootstrap, and report both side by side.

Bounded, single-threaded. Writes pnas-sub/analysis/out/A19_tv_series_only_fingerprint.json. Run from the root.
"""
import os, json
import numpy as np, pandas as pd

ROOT = os.environ.get("NARRATIVE_ATLAS_ROOT", os.path.expanduser("~/uoft/style_evolves")); os.chdir(ROOT)
import sys; sys.path.insert(0, os.path.join(ROOT, "code"))
from _methods import build_registries, load_spine, SPINE_TO_CANONICAL, WINDOW

reg = build_registries()
cb = pd.read_csv("atlas_canonical/codebook.csv")
CONSTRUCT = dict(zip(cb.canonical_column, cb.construct))
VAL_R = dict(zip(cb.canonical_column, cb.heldout_r))
inv = {v: k for k, v in SPINE_TO_CANONICAL.items()}
EARLY, LATE = (1950, 1970), (2000, 2020)

# television frame + spine, with unit type joined (same source and rule A15 uses)
FR_tv = pd.read_parquet("data/atlas/century_frame_tv.parquet").rename(columns={"idx": "id"})
SP_tv = load_spine("tv")
au = pd.read_csv("pnas-sub/analysis/out/tv_unit_audit.csv")[["tv_idx", "unit_type"]]
FR_tv = FR_tv.merge(au, left_on="id", right_on="tv_idx", how="left")
U = SP_tv.merge(FR_tv[[c for c in FR_tv.columns if c not in SP_tv.columns or c == "id"]], on="id", how="inner")
print(f"tv unified n={len(U)}; series n={(U['unit_type']=='series').sum()}")

def col_in(a, cols):
    return a if a in cols else (inv[a] if inv.get(a) in cols else None)

def fingerprint(frame):
    yr = pd.to_numeric(frame["year"], errors="coerce")
    win = (yr >= WINDOW[0]) & (yr <= WINDOW[1])
    core = [r["attribute"] for r in reg["history_core_within_tv"]]
    out = {}
    for a in core:
        c = col_in(a, set(frame.columns))
        if c is None: continue
        x = pd.to_numeric(frame[c], errors="coerce")
        mu, sd = x[win].mean(), x[win].std()
        if not sd or np.isnan(sd): continue
        z = (x - mu) / sd
        e = z[(yr >= EARLY[0]) & (yr < EARLY[1])].dropna()
        l = z[(yr >= LATE[0]) & (yr < LATE[1])].dropna()
        if len(e) < 30 or len(l) < 30: continue
        delta = float(l.mean() - e.mean())
        rng = np.random.RandomState(0)
        bd = np.array([l.sample(len(l), replace=True, random_state=rng.randint(1 << 31)).mean()
                       - e.sample(len(e), replace=True, random_state=rng.randint(1 << 31)).mean()
                       for _ in range(300)])
        p = 2 * min((bd > 0).mean(), (bd < 0).mean())
        out[a] = {"construct": CONSTRUCT.get(a), "delta_sd": round(delta, 3),
                  "ci": [round(float(np.percentile(bd, 2.5)), 3), round(float(np.percentile(bd, 97.5)), 3)],
                  "p": round(float(p), 4), "n_early": int(len(e)), "n_late": int(len(l))}
    return out

full = fingerprint(U)
series = fingerprint(U[U["unit_type"] == "series"])

# side-by-side on the attributes both compute, with sign-and-significance agreement
common = [a for a in full if a in series]
comp = []
for a in sorted(common, key=lambda a: -abs(full[a]["delta_sd"])):
    f, s = full[a], series[a]
    same_sign = np.sign(f["delta_sd"]) == np.sign(s["delta_sd"])
    both_sig = (f["p"] < 0.05) and (s["p"] < 0.05)
    comp.append({"attribute": a, "construct": f["construct"],
                 "full_delta": f["delta_sd"], "full_ci": f["ci"],
                 "series_delta": s["delta_sd"], "series_ci": s["ci"],
                 "same_sign": bool(same_sign), "both_sig": bool(both_sig)})

n_sig_full = sum(1 for a in common if full[a]["p"] < 0.05)
agree = sum(1 for c in comp if c["same_sign"] and c["both_sig"])
res = {"note": "A07 television fingerprint recomputed series-only; identical estimator/window/endpoints.",
       "n_common_attrs": len(common), "n_sig_full": n_sig_full,
       "n_sign_and_sig_agree": agree, "attributes": comp}

print(f"\ncommon robust-core tv attrs: {len(common)} | significant in full: {n_sig_full} | "
      f"sign+significance agree series-only: {agree}")
print(f"{'attribute':28} {'full':>7} {'series':>8}  agree")
for c in comp:
    print(f"  {c['attribute'][:26]:26} {c['full_delta']:7.3f} {c['series_delta']:8.3f}  "
          f"{'yes' if c['same_sign'] and c['both_sig'] else '.'}")

os.makedirs("pnas-sub/analysis/out", exist_ok=True)
json.dump(res, open("pnas-sub/analysis/out/A19_tv_series_only_fingerprint.json", "w"), indent=2)
print("\nwrote pnas-sub/analysis/out/A19_tv_series_only_fingerprint.json")
