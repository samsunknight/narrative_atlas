#!/usr/bin/env python3
"""Regenerate EVERY figure used in the paper (main + SI + Extended Data) from the released data.
Run from the project root:  .venv/bin/python3 code/build_all_figures.py
Verifies each paper-referenced figure exists and was (re)written by this run."""
import subprocess, sys, os, time, re, glob
ROOT=os.path.dirname(os.path.abspath(__file__))+"/.."; os.chdir(ROOT)
PY=sys.executable
GENERATORS=[
 "results/figures_certified/build_fig_f1.py",   # FIG_F1_resource
 "code/build_fig_f2.py",                         # FIG_F2_validation
 "code/fig_f6_crosscutting.py",                  # FIG_F6_crosscutting
 "code/build_levels_diff.py",                    # MAIN_density_grid, LEVELS_*, DIFF_*
 "code/build_missing_figs.py",                   # FIG_adaptation, FIG_crystallization, SUPP_1d_shifts, SUPP_crystallization_heatmap, SUPP_highstat_planes
 "code/build_all_exhibits.py",                   # ED3_era_predictability, ED4_rarefied_variety, ED5_continuum
 "code/build_convergence_twophase.py",           # fig_convergence_twophase
 "code/build_main_figures.py",                   # FIG2_stylespace
 "code/layer_genre_story.py",                    # FIG_genre_lifecycles
 "code/cool_descriptives_2.py",                  # FIG_production_code_crossmedium
]
t0=time.time()
for g in GENERATORS:
    if not os.path.exists(g): print(f"[MISS] {g} not found"); continue
    r=subprocess.run([PY,g],capture_output=True,text=True)
    print(f"[{'ok ' if r.returncode==0 else 'FAIL'}] {g}"+("" if r.returncode==0 else "  :: "+(r.stderr.strip().splitlines() or [''])[-1]))

# verify every figure the paper references exists and is fresh (rewritten during this run)
tex="".join(open(f).read() for f in ["paper/P0_resource_draft.tex","paper/si.tex"] if os.path.exists(f))
refs=sorted(set(re.findall(r'/([^/}]+\.png)', tex)))
print("\n=== VERIFY: every paper figure present + regenerated this run ===")
missing=stale=0
for f in refs:
    hits=[p for p in glob.glob(f"results/**/{f}",recursive=True) if "deprecated" not in p]
    if not hits: print(f"  [MISSING] {f}"); missing+=1; continue
    fresh=any(os.path.getmtime(p)>=t0 for p in hits)
    if not fresh: print(f"  [STALE]   {f}  (not rewritten this run)"); stale+=1
print(f"\n{len(refs)} figures referenced | missing {missing} | stale {stale} | "
      f"{'ALL REGENERATED' if missing==stale==0 else 'GAPS REMAIN'} | {time.time()-t0:.0f}s")
