# Reproducing the paper's numbers

One command reproduces all 96 checks across the five layers:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python reproduce.py
```

This regenerates `outputs/check_report.txt`, the authoritative table of every reproduced
number (target vs reproduced, per check). Do not maintain a second hand-typed table — read the
report.

## How to read the report

- `[R]` = **re-derived from raw** — recomputed in `reproduce.py` from the shipped data (corpus
  counts, structure/genre/texture validation, adaptation, reception, convergence,
  crystallization, variance ratio, genre lifecycles, Production-Code DiD, geometry).
- `[A]` = **asserted vs shipped** — the mood median r (0.39) and character-arc change r
  (0.37–0.48). Their per-rater human ratings are not shipped; the driver asserts against the
  computed sweep values (`data/validation/mood_numbers.json`, `arc_findings.json`). Never
  conflated with re-derived checks.
- Final line: `N/N passed`. This release: **96/96**.

`reproduce.py` also writes `outputs/gen_tab_corpus_rows.tex`, the corpus-coverage table rows
emitted directly from the data so the paper's counts cannot drift from the release by hand.

## One documented correction

`adaptation fantastical delta` re-derives to **0.58** (t = 8.6, 95% CI [0.45, 0.71]) from the
released corpus; the identical within-pair method reproduces the sci-fi delta exactly (0.38),
which pins the estimator. An earlier draft reported 0.41 (stale); the paper reports 0.58. See
the NOTES block at the foot of `outputs/check_report.txt`. No tolerances were widened to force
a pass.
