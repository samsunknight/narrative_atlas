"""Vocabulary + drift guard for the atlas tier columns (see TIER_VOCAB.md).

Asserts that the codebook carries both the frozen legacy `tier` and the current `heldout_tier`, that
every value falls in the documented vocabulary, and that the downstream-facing (`atlas_canonical/`) and
replication-build (`rep_build/narrative_atlas/atlas_canonical/`) copies of the codebook are identical.
Run after any codebook edit so the tier vocabulary can never change, or drift between copies, silently.
"""
import os, sys, hashlib
import pandas as pd

LEGACY = {"Headline", "Validated", "Marginal", "A", "B", "C", "--"}   # `C` here = below the bar
HELDOUT = {"A", "B", "C", "drop"}                                     # `C` here = marginal

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CANDIDATES = [
    os.path.join(REPO, "atlas_canonical", "codebook.csv"),
    os.path.join(REPO, "rep_build", "narrative_atlas", "atlas_canonical", "codebook.csv"),
]

def _check_one(path):
    d = pd.read_csv(path)
    assert "tier" in d.columns, f"{path}: missing frozen legacy `tier` column"
    assert "heldout_tier" in d.columns, f"{path}: missing `heldout_tier` column"
    bad_l = set(d["tier"].dropna().astype(str)) - LEGACY
    bad_h = set(d["heldout_tier"].dropna().astype(str)) - HELDOUT
    assert not bad_l, f"{path}: legacy `tier` has undocumented values {bad_l} (see TIER_VOCAB.md)"
    assert not bad_h, f"{path}: `heldout_tier` has undocumented values {bad_h} (see TIER_VOCAB.md)"
    return hashlib.md5(open(path, "rb").read()).hexdigest()

def main():
    present = [p for p in CANDIDATES if os.path.exists(p)]
    if not present:
        print("check_tiers: no codebook found"); sys.exit(1)
    hashes = {p: _check_one(p) for p in present}
    if len(set(hashes.values())) > 1:
        print("TIER GUARD FAILED — the codebook copies have drifted:")
        for p, h in hashes.items():
            print(f"  {h[:10]}  {p}")
        sys.exit(1)
    print(f"TIER GUARD PASSED — {len(present)} codebook copy/copies, vocabulary documented, no drift.")

if __name__ == "__main__":
    main()
