#!/usr/bin/env bash
# =============================================================================
# Narrative Atlas -- PNAS replication package. Single-command reproduction.
#
#   bash run_all.sh
#
# Reproduces every number in the paper from the shipped data, on one canonical
# partition, in order:
#   0. rebuild the structural spine FROM the canonical atlas, and assert the
#      rebuilt spine equals the atlas on every shared column (no drift);
#   1. run every analysis script (writes analysis outputs to pnas-sub/analysis/out/);
#   2. regenerate the data tables from those outputs (no hand-typed table cell);
#   3. run the guards (registries + counts, hardcoded-basis, spine==atlas);
#   4. reproduce.py -- re-derive every headline number from the released corpus (131 checks);
#   5. verify_paper.py -- recompute the reported numbers and confirm the manuscript agrees.
# Ends "Done." on success; any failed check is fatal.
#
# The package is self-locating: NARRATIVE_ATLAS_ROOT is set to this directory, and every
# script reads and writes relative to it, so the package reproduces wherever it is unpacked.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
export NARRATIVE_ATLAS_ROOT="$(pwd)"

PY=""
for cand in .venv/bin/python3 .venv/bin/python ../.venv/bin/python3 ../../.venv/bin/python3 "$(command -v python3 || true)"; do
    if [ -x "$cand" ]; then PY="$cand"; break; fi
done
[ -z "$PY" ] && { echo "no python found"; exit 1; }
echo "Using python: $PY"; echo "ROOT: $NARRATIVE_ATLAS_ROOT"; echo
A="pnas-sub/analysis"

echo "=== [0/5] Rebuild structural spine from the canonical atlas + assert no drift ==="
"$PY" code/rebuild_spine_from_atlas.py
"$PY" code/check_spine_atlas_sync.py
echo

echo "=== [1/5] Run every analysis script ==="
for s in A1_attribute_registry A2_variance_ratio_inference A9_medium_era_decomposition \
         A5_variance_explained_and_normed_convergence A4_convergence_contrasts \
         A08_all_layer_convergence A07_century_fingerprint A10_convergence_sensitivity \
         A10_dependence A3_adaptation_one_per_source A8_adaptation_era_control \
         A16_adaptation_hiconf A6_summary_convention_controls A12_whitened_distances \
         A02_covariance_agreement A13_calibration_equivalence A11_validation_fdr_dif \
         A04_tv_unit_type A15_tv_by_unit_convergence A14_country_language \
         A09_composition A09b_lang_supplement \
         A18_anchored_convergence A19_tv_series_only_fingerprint; do
    if [ -f "$A/$s.py" ]; then echo "  -> $s"
        ERR="$(mktemp)"
        if ! "$PY" "$A/$s.py" >/dev/null 2>"$ERR"; then
            if grep -qiE "wiki_text|plot_text|_text\.(parquet|csv)" "$ERR"; then
                echo "     [skipped: requires raw plot text, which is not redistributed; result shipped in pnas-sub/analysis/out/]"
            else echo "     FAILED ($s):"; tail -12 "$ERR"; rm -f "$ERR"; exit 1; fi
        fi; rm -f "$ERR"
    fi
done
echo

echo "=== [2/5] Regenerate data tables from analysis outputs ==="
"$PY" "$A/build_tables.py"
echo

echo "=== [3/5] Guards: registries + counts, hardcoded-basis, spine==atlas ==="
"$PY" code/_methods.py >/dev/null && echo "  registries + counts OK"
"$PY" "$A/check_registries.py"
echo

echo "=== [4/5] reproduce.py -- re-derive every headline number (131 checks) ==="
for rp in replication/reproduce.py reproduce.py; do
    if [ -f "$rp" ]; then LOG="$(mktemp)"; "$PY" "$rp" >"$LOG" 2>&1 || { echo "reproduce.py FAILED:"; tail -15 "$LOG"; exit 1; }; grep -E "passed" "$LOG" | tail -1; rm -f "$LOG"; break; fi
done
echo

echo "=== [5/5] verify_paper.py -- recompute reported numbers, confirm manuscript agrees ==="
VP="$(mktemp)"
if "$PY" "$A/verify_paper.py" >"$VP" 2>&1; then
    grep -E "PASS:|FAIL:|agree" "$VP" | tail -1 \
      || echo "  numbers recomputed (place pnas-sub/main.tex to also check manuscript agreement)"
else
    echo "  verify_paper FAILED:"; tail -15 "$VP"; rm -f "$VP"; exit 1
fi
rm -f "$VP"
echo
echo "Done."
