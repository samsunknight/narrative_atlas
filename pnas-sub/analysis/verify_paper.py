"""verify_paper.py (PNAS replication): recompute every reported number on the canonical partition and
print it beside its source and the manuscript locations that must carry it, then fail if the manuscript
disagrees. Regenerates the canonical numbers ledger from the freshly-written analysis outputs (so the
values are recomputed, not read from a cache), then runs the manuscript-vs-ledger consistency check.

Run after the analysis scripts have written analysis/out/*.json (run_all.sh does this):
    NARRATIVE_ATLAS_ROOT=<package> python pnas-sub/analysis/verify_paper.py
"""
import os, sys, json, subprocess

ROOT = os.environ.get("NARRATIVE_ATLAS_ROOT") or os.path.expanduser("~/uoft/style_evolves")
AN = os.path.join(ROOT, "pnas-sub", "analysis")
PY = sys.executable

# 1. regenerate the ledger from the current analysis outputs
subprocess.run([PY, os.path.join(AN, "A00_canonical_numbers.py")], check=True,
               env={**os.environ, "NARRATIVE_ATLAS_ROOT": ROOT})
ledger = json.load(open(os.path.join(AN, "out", "canonical_numbers.json")))

print("\n=== every reported number, its value, source, and manuscript locations ===")
for key, e in ledger.items():
    locs = ", ".join(f"{f}×{n}" for f, n in (e.get("manuscript_locations") or {}).items())
    print(f"  {key:28s} = {str(e['value']):>10s}  [{e['source']}]  in: {locs or '(prose scope only)'}")

# 2. manuscript must agree with the recomputed ledger (orphan + straggler guard)
print()
r = subprocess.run([PY, os.path.join(AN, "check_numbers.py")],
                   env={**os.environ, "NARRATIVE_ATLAS_ROOT": ROOT})
sys.exit(r.returncode)
