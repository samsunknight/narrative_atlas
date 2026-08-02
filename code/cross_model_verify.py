"""Cross-model robustness check reported in the Supplementary Information.

The deployed instrument is a single model, gpt-4o-mini. To show the atlas does not hinge on
that one model, the full corpus was re-scored with a second, independently trained frontier
model, Anthropic Claude Haiku 4.5, using the identical one-attribute-per-call prompts. For every
attribute we then correlate the two models' scores across works, within each medium, and ask
whether the agreement clears the same r >= 0.22 floor the paper uses to validate an attribute
against human judgment.

This script recomputes the summary statistics the SI reports from the per-attribute agreement
table (cross_model_agreement.csv), whose provenance is: for each medium and attribute, the
Pearson r between the Haiku 4.5 re-score (data/cross_model/atlas_haiku_full.parquet, long form
medium/idx/attr/value, means over the two ensemble prompts where an attribute was deployed as an
average) and the deployed gpt-4o-mini atlas, joined per work; book attributes are matched to
their film-side equivalents so a single r is reported per shared attribute. The raw Haiku scores
are shipped so the per-attribute r's can be recomputed in full against the deployed atlas.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AGREE = os.path.join(ROOT, "data", "cross_model", "cross_model_agreement.csv")

# The values stated in the Supplementary Information cross-model paragraph.
EXPECTED = {
    "film": {"n": 33, "median": 0.66, "clear": 33},
    "tv":   {"n": 33, "median": 0.65, "clear": 33},
    "book": {"n": 21, "median": 0.68, "clear": 21},
}
FLOOR = 0.22

a = pd.read_csv(AGREE)
print(f"{len(a)} attribute-by-medium pairs\n")
ok = True
for medium, g in a.groupby("medium"):
    med, mn, clear, n = g.r.median(), g.r.min(), int((g.r >= FLOOR).sum()), len(g)
    exp = EXPECTED[medium]
    hit = (n == exp["n"]) and (round(med, 2) == exp["median"]) and (clear == exp["clear"])
    ok &= hit
    print(f"  {medium:5s}  n={n:2d}  median r={med:.3f}  min r={mn:.3f}  "
          f"clear {FLOOR}: {clear}/{n}   [{'ok' if hit else 'MISMATCH'}]")

med_all, min_all, clear_all = a.r.median(), a.r.min(), int((a.r >= FLOOR).sum())
print(f"\n  all    n={len(a)}  median r={med_all:.3f}  min r={min_all:.3f}  "
      f"clear {FLOOR}: {clear_all}/{len(a)}")
assert clear_all == len(a), "not every pair clears the validation floor"
assert ok, "a per-medium summary does not match the reported value"
print("\nDone. Cross-model agreement reproduces the Supplementary Information numbers.")
