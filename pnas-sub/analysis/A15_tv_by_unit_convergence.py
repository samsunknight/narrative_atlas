"""A15 (PNAS rebuild, Phase 2 / T3): rerun cross-media convergence within television unit type.

The television corpus mixes scope (series, episodes, television films). Because that mixture shifts
over the century, a convergence measured on the full television corpus could be an artifact of the
changing mixture rather than a change in television itself. This module reruns A4's convergence
exactly --- mood/genre/arc, percentile-rank within medium, medium x decade centroids, convergence =
distance(1950) minus distance(later), and the paired mood-minus-genre / mood-minus-arc contrasts ---
but restricts the television side to one unit type at a time. Film and novel are unchanged.

The decisive control is SERIES-ONLY: holding the television unit type constant at "series" across
every decade fixes the composition, so any surviving convergence is within-series, not a mixture
effect. If mood still converges most under series-only (and under high-confidence, dropping the
non-television "other"/"unknown" matches), the television result is not an artifact of unit mixture.

Bounded, single-threaded. Writes analysis/out/A15_tv_by_unit_convergence.json. Run from the root.
"""
import os, json
import numpy as np, pandas as pd

ROOT = os.environ.get("NARRATIVE_ATLAS_ROOT", os.path.expanduser("~/uoft/style_evolves")); os.chdir(ROOT)
import sys; sys.path.insert(0, os.path.join(ROOT, "code"))
man = pd.read_csv("data/validation/rescore_manifest.csv"); man = man[man.deploy == True]
LAY = man.set_index("attr_id")["layer"].to_dict()
FR = {m: pd.read_parquet(f"data/atlas/century_frame_{m}.parquet") for m in ["film", "book", "tv"]}
LAYERS = ["mood", "genre", "arc"]
PAIRS = {"novel-tv": ("book", "tv"), "film-tv": ("film", "tv")}
DECS = list(range(1950, 2011, 10))
MIN = 30

# join television unit type onto the tv frame (idx <-> tv_idx)
au = pd.read_csv("pnas-sub/analysis/out/tv_unit_audit.csv")[["tv_idx", "unit_type", "confidence"]]
FR["tv"] = FR["tv"].merge(au, left_on="idx", right_on="tv_idx", how="left")

# Layers sourced from their declared registries (canonical), not manifest prefix-matching.
from _methods import build_registries
_reg = build_registries()
SH = {L: [a for a in [r["attribute"] for r in _reg[f"convergence_{L}"]]
          if all(a in FR[m].columns for m in FR)] for L in LAYERS}

# TV subsets: full corpus, then unit-type restrictions. Series-only fixes the composition.
def tv_mask(name):
    u = FR["tv"]["unit_type"].astype(str)
    c = FR["tv"]["confidence"].astype(str)
    if name == "full":            return np.ones(len(u), bool)
    if name == "series_only":     return (u == "series").values
    if name == "scripted":        return u.isin(["series", "miniseries", "tv_film"]).values
    if name == "high_confidence": return ((c == "wikidata") & u.isin(
        ["series", "miniseries", "tv_film", "episode", "special", "serial_anthology", "season"])).values
    if name == "episode_only":    return (u == "episode").values
    raise ValueError(name)

SUBSETS = ["full", "series_only", "scripted", "high_confidence", "episode_only"]

def build(tvsub):
    """A4's transform with the tv frame restricted to a row subset; returns the ranked layer matrices,
    decade-cell indices, and per-medium arrays needed by dist()."""
    frames = {"film": FR["film"], "book": FR["book"], "tv": FR["tv"].loc[tvsub].reset_index(drop=True)}
    PZ = {m: {L: frames[m][SH[L]].rank(pct=True).values for L in LAYERS} for m in frames}
    YR = {m: frames[m]["year"].values for m in frames}
    CELL = {m: {d: np.where((YR[m] >= d) & (YR[m] < d + 10))[0] for d in DECS} for m in frames}
    return PZ, CELL

def dist(PZ, idx, a, b, L, d):
    ia, ib = idx[a][d], idx[b][d]
    if len(ia) < MIN or len(ib) < MIN: return np.nan
    ca = np.nanmean(PZ[a][L][ia], axis=0); cb = np.nanmean(PZ[b][L][ib], axis=0)
    return float(np.linalg.norm(ca - cb))

def stats(PZ, CELL):
    out = {}
    for pn, (a, b) in PAIRS.items():
        for L in LAYERS:
            d50 = dist(PZ, CELL, a, b, L, 1950); d90 = dist(PZ, CELL, a, b, L, 1990)
            d10 = dist(PZ, CELL, a, b, L, 2010)
            out[(pn, L)] = {"d1950": d50, "d1990": d90, "d2010": d10,
                            "conv_trough": (d50 - d90) if not (np.isnan(d50) or np.isnan(d90)) else np.nan,
                            "conv_full": (d50 - d10) if not (np.isnan(d50) or np.isnan(d10)) else np.nan}
    return out

def ci(vals):
    v = np.array([x for x in vals if x is not None and not np.isnan(x)])
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] if len(v) >= 20 else [None, None]

res = {"note": "A4 convergence rerun within television unit type; film/novel unchanged. "
               "conv = distance(1950) - distance(later); + = converged.",
       "subsets": {}, "min_cell": MIN, "reps": 300}
REPS = 300
for name in SUBSETS:
    mask = tv_mask(name)
    PZ, CELL = build(mask)
    pt = stats(PZ, CELL)
    # shared paired bootstrap: resample works within medium x decade, recompute all layers
    rng = np.random.RandomState(0)
    boot = []
    for _ in range(REPS):
        rs = {m: {d: (rng.choice(CELL[m][d], len(CELL[m][d]), replace=True) if len(CELL[m][d]) else CELL[m][d])
                  for d in DECS} for m in CELL}
        boot.append(stats(PZ, rs))
    tv_n = {d: int(len(CELL["tv"][d])) for d in DECS}
    # Serialize BOTH horizons: conv_full (2010 endpoint) and conv_trough (1990 endpoint, BEFORE the
    # post-2000 series-page surge -- the temporal-inflation control the plan rests the TV decision on).
    entry = {"tv_cell_n": tv_n, "by_pair_layer": {}, "contrasts": {}}
    for pn in PAIRS:
        for L in LAYERS:
            s = pt[(pn, L)]
            entry["by_pair_layer"][f"{pn}|{L}"] = {
                "conv_full": None if np.isnan(s["conv_full"]) else round(s["conv_full"], 3),
                "conv_full_CI": ci([bt[(pn, L)]["conv_full"] for bt in boot]),
                "conv_trough": None if np.isnan(s["conv_trough"]) else round(s["conv_trough"], 3),
                "conv_trough_CI": ci([bt[(pn, L)]["conv_trough"] for bt in boot])}
        for horizon, key in [("full", "conv_full"), ("trough", "conv_trough")]:
            for other in ["genre", "arc"]:
                dp = pt[(pn, "mood")][key] - pt[(pn, other)][key]
                dci = ci([bt[(pn, "mood")][key] - bt[(pn, other)][key] for bt in boot])
                excl = dci[0] is not None and (dci[0] > 0 or dci[1] < 0)
                entry["contrasts"][f"{pn}|mood-minus-{other}|{horizon}"] = {
                    "diff": None if np.isnan(dp) else round(dp, 3), "CI": dci, "excludes0": bool(excl)}
    res["subsets"][name] = entry
    # console summary: both endpoints
    print(f"\n=== TV subset: {name}  (tv works/decade: {list(tv_n.values())}) ===")
    for pn in PAIRS:
        for horizon in ["full", "trough"]:
            mg = entry["contrasts"][f"{pn}|mood-minus-genre|{horizon}"]
            ma = entry["contrasts"][f"{pn}|mood-minus-arc|{horizon}"]
            ep = "2010" if horizon == "full" else "1990(pre-surge)"
            print(f"  {pn:9s} [{ep:15s}] mood-genre {mg['diff']} CI{mg['CI']} {'*' if mg['excludes0'] else ' '} | "
                  f"mood-arc {ma['diff']} CI{ma['CI']} {'*' if ma['excludes0'] else ' '}")

# decision helper: does mood-converges-most survive across the meaningful subsets?
def survives(name, horizon):
    e = res["subsets"][name]
    ks = [f"{pn}|mood-minus-{o}|{horizon}" for pn in PAIRS for o in ["genre", "arc"]]
    if any(e["contrasts"][k]["diff"] is None for k in ks):
        return None
    return all(e["contrasts"][k]["excludes0"] and e["contrasts"][k]["diff"] > 0 for k in ks)
res["mood_converges_most_survives"] = {
    h: {n: survives(n, h) for n in ["full", "series_only", "scripted", "high_confidence"]}
    for h in ["full", "trough"]}

os.makedirs("pnas-sub/analysis/out", exist_ok=True)
json.dump(res, open("pnas-sub/analysis/out/A15_tv_by_unit_convergence.json", "w"), indent=2)
print("\nmood-converges-most survives:", res["mood_converges_most_survives"])
print("wrote out/A15_tv_by_unit_convergence.json")
