#!/usr/bin/env python3
"""Master reproduction driver for the Narrative Atlas package.
Regenerates every headline NUMBER, FIGURE, and TABLE from the released data/.
Run from the package root:   python code/run_all.py
Does NOT re-run the LLM scoring step (the released scored corpus IS the data;
the scoring code is in code/rescore_oneat.py + code/PROMPT.md for reference)."""
import subprocess, sys, os, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
for d in ["results/figures","results/figures_certified","results/cool","results/tables",
          "results/reception","results/layers/genre","results/layers/structure","outputs"]:
    os.makedirs(d, exist_ok=True)
PY = sys.executable
ok, bad = [], []
def run(kind, script):
    if not os.path.exists(script): return
    r = subprocess.run([PY, script], capture_output=True, text=True)
    if r.returncode == 0: ok.append(script); print(f"[ok  ] {kind:8} {script}")
    else:
        bad.append(script)
        err = (r.stderr.strip().splitlines() or ["?"])[-1]
        print(f"[FAIL] {kind:8} {script}  ::  {err[:120]}")
run("numbers", "reproduce.py")
for f in sorted(glob.glob("code/figures/*.py")):
    if "build_all_figures" in f: continue
    run("figure", f)
for f in ["code/robustness_crossmedium_sets.py","code/run_subsets.py","code/reception_bootstrap.py"]:
    run("table", f)
print(f"\n=== run_all: {len(ok)} ok, {len(bad)} failed ===")
if bad: print("FAILED:", *bad, sep="\n  ")
