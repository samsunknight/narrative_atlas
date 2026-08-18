"""A1 (PNAS Phase 2, P0.1): freeze the claim-specific attribute bases in one machine-readable file,
so every script, figure, and prose number declares which basis it uses and no analysis quietly
switches bases. Each attribute carries its held-out tier and film/book validation r from the canonical
codebook. Run from ~/uoft/style_evolves/.
"""
import os, sys, json
import numpy as np, pandas as pd

ROOT = os.environ.get("NARRATIVE_ATLAS_ROOT", os.path.expanduser("~/uoft/style_evolves")); os.chdir(ROOT)
sys.path.insert(0, "code")
cb = pd.read_csv("atlas_canonical/codebook.csv")
F = pd.read_csv("data/corpus/film_structural_1890_2025.csv").rename(columns={"film_idx": "id"})
B = pd.read_csv("data/corpus/book_structural_1890_2025.csv").rename(columns={"book_idx": "id"})
T = pd.read_csv("data/corpus/tv_structural_1890_2025.csv").rename(columns={"tv_idx": "id"})

# Import the canonical bases from the one module that owns them; fail loudly if the import breaks
# rather than falling back to a hardcoded copy that could silently drift.
from _methods import VAL16_KEYS, ADAPTATION10_KEYS, TEMPORAL_CALLOUTS
shared_structural_primary = [c for c in B.columns if any(k in c.lower() for k in VAL16_KEYS)
                             and c in T.columns and c in F.columns]
adaptation_primary = [c for c in F.columns for k in ADAPTATION10_KEYS if k in c]
adaptation_primary = list(dict.fromkeys(adaptation_primary))
# non-structural bases (setting/mood/layers) live in the DENSE century-frame atlas, not the
# structural-spine CSVs; structural bases (character_development, ending-resolved) are in both.
FR = pd.read_parquet("data/atlas/century_frame_film.parquet")
BR = pd.read_parquet("data/atlas/century_frame_book.parquet")
TR = pd.read_parquet("data/atlas/century_frame_tv.parquet")
temporal_robust = {}
for concept, key in TEMPORAL_CALLOUTS.items():
    hits = [c for c in FR.columns if key in c]
    temporal_robust[concept] = hits[0] if hits else None
# layer sets (shared across media) via the deployed manifest
man = pd.read_csv("data/validation/rescore_manifest.csv"); man = man[man.deploy == True]
LAY = man.set_index("attr_id")["layer"].to_dict()
def layer_shared(L): return [c for c in FR.columns if LAY.get(c) == L and c in BR.columns and c in TR.columns]

def meta(col):
    r = cb[cb.canonical_column == col]
    if r.empty: return {"tier": None, "film_r": None, "book_r": None, "construct": None}
    r = r.iloc[0]
    fr, br = r.get("film_r"), r.get("book_r")
    return {"tier": r.get("heldout_tier"), "construct": r.get("construct"),
            "film_r": None if pd.isna(fr) else round(float(fr), 3),
            "book_r": None if pd.isna(br) else round(float(br), 3)}

reg = {
  "note": "Claim-specific attribute bases for the PNAS analyses. Each script/figure must name the basis it uses.",
  "shared_structural_primary": {"n": len(shared_structural_primary),
      "used_for": "medium-vs-era 1.66 ratio; cross-medium geometry; crystallization core",
      "attrs": {c: meta(c) for c in shared_structural_primary}},
  "adaptation_primary": {"n": len(adaptation_primary),
      "used_for": "novel-to-film within-pair contrast (10 shared-scale attrs)",
      "attrs": {c: meta(c) for c in adaptation_primary}},
  "temporal_robust": {"n": len([v for v in temporal_robust.values() if v]),
      "used_for": "Figure 3 four robust trajectories (all tier A, cross-medium)",
      "attrs": {k: (v, meta(v)) for k, v in temporal_robust.items()}},
  "mood_shared": {"attrs": layer_shared("mood")},
  "genre_shared": {"attrs": layer_shared("genre")},
  "arc_shared": {"attrs": layer_shared("arc")},
  "television": {"status": "inherited", "note": "no separate human survey; film-validated visual-medium "
                 "instrument applied unchanged; no unit-type metadata available (corpus reads series-level); "
                 "read as a model-anchored extension. Dated (28 undated of 32,223), entirely post-1950."},
}
os.makedirs("pnas-sub/analysis/out", exist_ok=True)
json.dump(reg, open("pnas-sub/analysis/out/attribute_registry.json", "w"), indent=2)
print(f"shared_structural_primary: {len(shared_structural_primary)} attrs")
print(f"adaptation_primary: {len(adaptation_primary)} attrs")
print("temporal_robust:", {k: (v, meta(v)["tier"]) for k, v in temporal_robust.items()})
print(f"mood_shared {len(layer_shared('mood'))} | genre_shared {len(layer_shared('genre'))} | arc_shared {len(layer_shared('arc'))}")
print("wrote out/attribute_registry.json")
