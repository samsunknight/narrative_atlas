#!/usr/bin/env bash
# =============================================================================
# Narrative Atlas replication package -- single-command driver.
#
# Runs, in order:
#   1. STRUCTURAL SPINE REBUILD   code/rebuild_spine_from_atlas.py
#        rebuilds data/corpus_rebuilt/{film,book,tv}_structural_1890_2025.csv from the
#        canonical atlas parquet, so the analysis spine can never drift from the scores.
#   2. NUMBERS REPLICATION        reproduce.py
#        regenerates every headline number in the paper from shipped data. This also runs
#        the embedded guards (prompt-provenance, spine==atlas drift, tier vocabulary).
#   3. F3 JOINT-STRUCTURE EXHIBIT code/joint_structure.py
#        rebuilds the 44x44 human/machine correlation-matrix agreement (off-diag r,
#        PC1 Tucker congruence, coupling ratio) from shipped data.
#
# Figures/tables: the manuscript's figure and table GENERATORS live in the full working
# tree (they use absolute repo paths and a results/ scaffold not shipped here); this
# package ships the RENDERED exhibits under outputs/figures/. run_all.sh therefore verifies
# the NUMBERS and the one self-contained joint-structure exhibit, not the figure PNGs.
#
# Usage:  bash run_all.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# locate a python: prefer the project venv, else python3 on PATH
PY=""
for cand in ../../.venv/bin/python3 ../.venv/bin/python3 .venv/bin/python3; do
    if [ -x "$cand" ]; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then PY="$(command -v python3)"; fi
echo "Using python: $PY"
echo

echo "=== [1/3] Rebuilding structural spine from the canonical atlas ==="
"$PY" code/rebuild_spine_from_atlas.py
echo

echo "=== [2/3] Reproducing every paper number (reproduce.py) ==="
"$PY" reproduce.py
echo

echo "=== [3/3] F3 joint-structure exhibit (joint_structure.py) ==="
"$PY" code/joint_structure.py
echo

echo "Done."
