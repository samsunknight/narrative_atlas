"""Guard against the spine-vs-atlas drift that the atlas is the source of truth for.

The geometry reads the structural spine CSVs; those must never disagree with the canonical atlas
parquet on any shared column. This re-derives each spine column's atlas source with the same mapping
`rebuild_spine_from_atlas.py` uses and asserts the shipped spine equals the atlas value per work.
Exits non-zero on any drift, so it can gate `run_all` / a pre-push check. A 2026-07 re-score refreshed
the atlas but left the spine stale; this makes that failure impossible to ship silently."""
import os, sys, importlib.util
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def P(*a): return os.path.join(ROOT, *a)

# reuse the exact atlas->spine mapping from the rebuild script
spec = importlib.util.spec_from_file_location("_rebuild", P("code", "rebuild_spine_from_atlas.py"))
rb = importlib.util.module_from_spec(spec); spec.loader.exec_module(rb)

TOL = 0.02
bad = []
for m in ("film", "book", "tv"):
    A = pd.read_parquet(P(f"data/atlas/century_frame_{m}.parquet")).set_index("idx")
    S = pd.read_csv(P(f"data/corpus/{m}_structural_1890_2025.csv"))
    idc = f"{m}_idx"; acols = set(A.columns)
    spine = [c for c in S.columns if c not in (idc, "title", "year")]
    Si = S.set_index(idc)
    for c in spine:
        src = rb.med_source(c, m, acols)
        if src is None: continue                       # no atlas source (book-only analog columns)
        a = pd.to_numeric(A[src].reindex(Si.index), errors="coerce")
        s = pd.to_numeric(Si[c], errors="coerce")
        both = a.notna() & s.notna()
        if both.sum() == 0: continue
        d = float((a[both] - s[both]).abs().mean())
        if d > TOL:
            bad.append((m, c[:40], src[:40], round(d, 3)))

if bad:
    print("SPINE-ATLAS DRIFT DETECTED (rebuild the spine with rebuild_spine_from_atlas.py):")
    for m, c, src, d in bad: print(f"  [{m}] {c} vs atlas {src}: mean|diff|={d}")
    sys.exit(1)
print("spine == atlas on every mapped column (film/book/tv). No drift.")
