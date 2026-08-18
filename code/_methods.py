"""Canonical feature basis for the Narrative Atlas, defined once and imported by reproduce.py and
the figure generators so the structural spine and the cross-medium set never drift between scripts.
Any script that needs the spine or the sixteen cross-medium attributes imports them from here rather
than hardcoding its own list."""
import os, pandas as pd

ROOT = os.environ.get("NARRATIVE_ATLAS_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project/package root; code/ sits under it
WINDOW = (1915, 2020)   # the analysis window; decades outside it are too sparse to estimate a centroid

# The sixteen cross-medium-validated structural attributes, on a scale identical across all three
# media: the four world-type attributes, cast size, the four protagonist qualities, the two
# plot-drive axes, immersive setting, and the five plot-structure additions (development, hook, time
# and plot linearity, peripeteia). Matched by substring against the shared column names.
VAL16_KEYS = ["science_fictional", "fantastical", "realistic_was_the_world", "world_building",
              "relatable_did_you_find", "competent_was_this_protagonist", "how_many_protagonists",
              "proactiv", "plot_driven", "character_driven", "immersive", "character_development",
              "opening_hook", "time_linearity", "plot_linearity", "ending_reversal"]

def load_spine(medium):
    """The released structural corpus for one medium, keyed by a common `id` column."""
    idx = {"film": "film_idx", "book": "book_idx", "tv": "tv_idx"}[medium]
    return pd.read_csv(os.path.join(ROOT, "data", "corpus", f"{medium}_structural_1890_2025.csv")).rename(columns={idx: "id"})

def spine_attrs(df):
    """The structural-spine attribute columns of a corpus frame (everything but the identifiers)."""
    # exclude identifiers; the medium idx columns are listed defensively so the spine is 36
    # even if a caller reads the corpus CSV directly without load_spine (which renames idx to id).
    drop = ("id", "title", "year", "dec", "medium", "loglen", "film_idx", "book_idx", "tv_idx")
    return [c for c in df.columns if c not in drop]

def cross_medium(df_or_cols):
    """The sixteen cross-medium columns present in a frame or column list, in column order."""
    cols = df_or_cols if isinstance(df_or_cols, (list, tuple)) else list(df_or_cols.columns)
    return [c for c in cols if any(k in c.lower() for k in VAL16_KEYS)]

# ---------------------------------------------------------------------------
# Claim-specific attribute registries (PNAS rebuild, Phase 0 / P0.2)
#
# The twelve named registries below are DERIVED from the canonical codebook, never hand-typed, so
# every analysis declares which basis it uses and no script quietly switches bases. The only curated
# lists that live here are the two analytical choices that are not recoverable from the codebook
# alone: which ten shared-scale attributes carry the adaptation contrast, and which four robust
# trajectories serve as the historical callouts. Both are defined once, here.
# ---------------------------------------------------------------------------

# The ten shared-survey-scale attributes used for the novel-to-film within-pair contrast: the four
# world types, cast size, the competent/proactive/relatable protagonist qualities, and the two
# plot-drive axes. A subset of the sixteen, restricted to those scored on an identical scale in both
# the film and novel surveys. Matched by substring against the shared column names.
ADAPTATION10_KEYS = ["science_fictional", "fantastical", "realistic_was_the_world", "world_building",
                     "how_many_protagonists", "competent_was_this_protagonist", "proactiv",
                     "relatable_did_you_find", "plot_driven", "character_driven"]

# The four robust historical trajectories used as interpretable callouts in the century figure,
# each the strongest robust-core mover in a distinct construct. concept -> codebook substring key.
TEMPORAL_CALLOUTS = {"protagonist_transformation": "character_development",
                     "future_setting_era": "setting_when",
                     "ending_resolution": "entirely_unresolved_to_7_entirely_resolved",
                     "darkening": "mood_Dark"}

# release_class values that are not scored narrative attributes; excluded from the scored/validated
# counts so the codebook's raw row count reconciles to the manuscript's reported totals.
NON_NARRATIVE_CLASSES = ("reception",)

_CONSTRUCT_TO_LAYER = {"structure": "structure", "mood": "mood", "genre": "genre",
                       "character-arc": "arc"}

# Known spine-column -> codebook canonical_column naming exceptions. During the spine rebuild every
# structural column was renamed to its codebook canonical name except immersive setting, which kept
# its raw survey-question name; this alias documents that one residual mismatch so the spine still
# joins cleanly to the codebook. (A candidate for a later corpus-column rename, tracked in the
# canonical numbers ledger.)
SPINE_TO_CANONICAL = {
    "4_how_immersive_did_you_find_this_setting_by_immersive_we_mean_fully_r": "immersive_setting",
}

def _spine_to_canon(col, canon_set):
    """Map a spine column to its codebook canonical_column, via the documented alias when the spine
    kept a non-canonical name."""
    if col in canon_set:
        return col
    return SPINE_TO_CANONICAL.get(col)

def load_codebook():
    """The live canonical codebook (one row per scored attribute), read only from the authoritative
    file so a stale ``.bak_*`` sibling can never slip into a registry."""
    return pd.read_csv(os.path.join(ROOT, "atlas_canonical", "codebook.csv"))

def _media_set(s):
    return set(str(s).replace(" ", "").split(","))

def reconcile_counts(cb=None):
    """The headline attribute counts, reconciled from the codebook by an explicit rule so the raw
    row count (which includes non-narrative reception items) is never quoted by mistake."""
    if cb is None:
        cb = load_codebook()
    narrative = cb[~cb.release_class.isin(NON_NARRATIVE_CLASSES)]
    return {
        "codebook_rows": int(len(cb)),
        "reception_excluded": int(cb.release_class.isin(NON_NARRATIVE_CLASSES).sum()),
        "scored_attributes": int(len(narrative)),
        "validated_attributes": int((narrative.heldout_tier != "drop").sum()),
        "robust_core_attributes": int((narrative.release_class == "core").sum()),
        "cross_medium_eligible": int(narrative.cross_medium.fillna(False).astype(bool).sum()),
        "rule": "scored = codebook rows minus reception; validated = scored with heldout_tier != drop; "
                "robust-core = scored with release_class == core.",
    }

def _row_meta(r, registry, reason, raw_level_ok):
    """One registry row: the analytical fields plus the codebook validation metadata, so a basis is
    self-documenting (construct, layer, scale, validation r and interval, tier, readability)."""
    def num(x):
        return None if pd.isna(x) else round(float(x), 3)
    construct = r.get("construct")
    return {
        "registry": registry, "attribute": r["canonical_column"], "construct": construct,
        "layer": _CONSTRUCT_TO_LAYER.get(construct, construct), "scale": r.get("scale"),
        "media": r.get("media"), "film_r": num(r.get("film_r")), "book_r": num(r.get("book_r")),
        "heldout_tier": r.get("heldout_tier"), "heldout_r": num(r.get("heldout_r")),
        "heldout_ci_lo": num(r.get("heldout_ci_lo")), "heldout_ci_hi": num(r.get("heldout_ci_hi")),
        "release_class": r.get("release_class"), "readability": r.get("readability"),
        "cross_medium": bool(r.get("cross_medium")),
        "raw_cross_level_allowed": bool(raw_level_ok), "within_medium_rank_allowed": True,
        "reason": reason,
    }

def build_registries(cb=None):
    """The twelve claim-specific registries, each a list of self-documenting rows derived from the
    codebook. Structural bases (geometry, structure-layer convergence, adaptation) match the curated
    key lists against the shared spine columns so their membership is exact; the remaining registries
    are codebook filters on construct, release class, media coverage, and readability."""
    if cb is None:
        cb = load_codebook()
    narrative = cb[~cb.release_class.isin(NON_NARRATIVE_CLASSES)].copy()
    by_col = {r["canonical_column"]: r for _, r in narrative.iterrows()}

    # structural bases: resolve the curated key lists against the real spine columns (exact 16 / 10),
    # mapping each spine column to its codebook row and failing loudly if any fails to resolve, so a
    # silent drop (as the immersive-setting naming mismatch once caused) can never recur.
    spine_cols = spine_attrs(load_spine("film"))
    canon_set = set(by_col)

    def structural_rows(keys, name, reason, expected):
        cols = [c for c in spine_cols if any(k in c.lower() for k in keys)]
        rows, unresolved = [], []
        for c in cols:
            canon = _spine_to_canon(c, canon_set)
            if canon is None or canon not in by_col:
                unresolved.append(c)
            else:
                rows.append(_row_meta(by_col[canon], name, reason, True))
        if unresolved:
            raise ValueError(f"{name}: spine columns with no codebook row: {unresolved}")
        if len(rows) != expected:
            raise ValueError(f"{name}: expected {expected} attrs, resolved {len(rows)}")
        return rows

    reg = {}
    reg["geometry_shared16"] = structural_rows(VAL16_KEYS, "geometry_shared16",
        "identical-scale structural spine for medium-vs-era geometry", 16)
    reg["adaptation_shared10"] = structural_rows(ADAPTATION10_KEYS, "adaptation_shared10",
        "shared-survey-scale attributes for the novel-to-film within-pair contrast", 10)

    # convergence layers: structure = the geometry spine; mood/genre/arc = construct, all-three-media
    reg["convergence_structure"] = [dict(r, registry="convergence_structure",
        reason="structural layer for the all-layer convergence audit (identical-scale)")
        for r in reg["geometry_shared16"]]
    for layer, construct in [("mood", "mood"), ("genre", "genre"), ("arc", "character-arc")]:
        sub = narrative[(narrative.construct == construct)
                        & (narrative.heldout_tier != "drop")
                        & (narrative.media.apply(lambda m: {"film", "book", "tv"} <= _media_set(m)))]
        reg[f"convergence_{layer}"] = [_row_meta(r, f"convergence_{layer}",
            f"{layer} layer for the all-layer convergence audit (rank-within-medium)", False)
            for _, r in sub.iterrows()]

    # within-medium historical fingerprints: robust-core attributes validated for that medium
    for medium, key in [("film", "film"), ("novel", "book"), ("tv", "tv")]:
        sub = narrative[(narrative.release_class == "core")
                        & (narrative.heldout_tier != "drop")
                        & (narrative.media.apply(lambda m: key in _media_set(m)))]
        reg[f"history_core_within_{medium}"] = [_row_meta(r, f"history_core_within_{medium}",
            f"robust-core attribute validated for {medium}; within-medium century change", False)
            for _, r in sub.iterrows()]

    shared = narrative[(narrative.release_class == "core") & (narrative.heldout_tier != "drop")
                       & (narrative.media.apply(lambda m: {"film", "book", "tv"} <= _media_set(m)))]
    reg["history_shared_crossmedium"] = [_row_meta(r, "history_shared_crossmedium",
        "robust-core, all-three-media; cross-medium direction-of-change comparison",
        bool(r.get("cross_medium"))) for _, r in shared.iterrows()]

    for flag in ["summary-readable", "summary-imputed"]:
        sub = narrative[(narrative.release_class == "core") & (narrative.heldout_tier != "drop")
                        & (narrative.readability == flag)]
        name = "summary_readable_core" if flag == "summary-readable" else "summary_imputed_core"
        reg[name] = [_row_meta(r, name,
            f"robust-core {flag} attribute; separates read-from-summary vs imputed layers", False)
            for _, r in sub.iterrows()]
    return reg

if __name__ == "__main__":
    import json
    cb = load_codebook()
    counts = reconcile_counts(cb)
    reg = build_registries(cb)
    rows = [row for rs in reg.values() for row in rs]
    out = os.path.join(ROOT, "pnas-sub", "analysis", "out")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "attribute_registry.csv"), index=False)
    json.dump({"counts": counts, "registry_sizes": {k: len(v) for k, v in reg.items()}},
              open(os.path.join(out, "registry_summary.json"), "w"), indent=2)
    print("reconciled counts:", json.dumps(counts, indent=2))
    print("\nregistry sizes:")
    for k, v in reg.items():
        print(f"  {k:28s} {len(v)}")
    print("\nwrote analysis/out/attribute_registry.csv  +  registry_summary.json")
