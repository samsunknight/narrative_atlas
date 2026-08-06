# Attribute tier vocabulary — READ BEFORE FILTERING ON `tier`

The codebook carries **two** tier columns. They use overlapping letters with **different meanings**, so
read this before writing any `tier`-based selection.

## `tier` — LEGACY, frozen
The original mixed scheme: `Headline`, `Validated`, `Marginal`, `A`, `B`, `C`, `--`. In this column
**`C` means *below the validation bar*** (6 attributes). This column is **frozen** for backward
compatibility: neither the held-out re-validation nor the later significance re-grade changed it, so
existing downstream code that reads `tier` (e.g. `tier != "C"` to drop the 6 unusable attributes)
behaves exactly as before.

## `heldout_tier` — CURRENT, recommended
An attribute **validates** when its held-out correlation with human judgment is significantly positive:
its 95% confidence interval lower bound (`heldout_ci_lo`) is above zero. The tier grades the strength of
that evidence by the same lower bound (genre validates by ROC AUC and is tier `A`):

| value | meaning | n (of 219 released) |
|---|---|---|
| `A` | robust — CI lower bound > 0.30 | 92 |
| `B` | validated — CI lower bound > 0.22 | 31 |
| `C` | **marginal** — significantly positive (CI lower bound > 0) but below 0.22 | 75 |
| `drop` | **not validated** — CI lower bound ≤ 0 (not significantly positive) | 21 |

`A` + `B` = the **robustly-validated core, 123 attributes** (CI lower bound above 0.22); `A` + `B` + `C`
= the **195 validated** attributes (main descriptive set: 216 scored, 195 validated). This is the same
rule reproduce.py and the paper use — a point-r cut (e.g. `r > 0.22`) is **not** the release rule and
should not be imposed, since significance is n-dependent (a lower-n attribute must reach a higher point
correlation to clear the interval).

## ⚠ The collision that will bite you
**`C` is below-bar in `tier` (6 attrs) but *marginal-but-validated* in `heldout_tier` (75 attrs).** A
filter written as `tier != "C"` and silently re-pointed at `heldout_tier` flips from excluding 6 unusable
attributes to excluding 75 usable-but-marginal ones **while admitting the 21 that should be dropped**.
Nothing errors; the numbers just move. When migrating:
- To keep the old behaviour, keep reading `tier`.
- To adopt the current tiers, read `heldout_tier` and use **`heldout_tier in {"A","B"}`** for the
  robustly-validated core (123 attrs), treat `C` as marginal/exploratory (validated but weak), and
  exclude `drop`. To cut on your own bar, use `heldout_ci_lo` directly (`> 0` = validated,
  `> 0.22` = robust). Do **not** reuse a `!= "C"` rule.
- Per-attribute `heldout_r`, `heldout_ci_lo`, `heldout_ci_hi` are in the codebook; full detail in
  `attribute_registry.csv`.

## Guard
Run `python atlas_canonical/check_tiers.py` after any codebook edit. It asserts both columns are present,
every value is in the documented vocabulary, and the downstream and replication-build copies agree.
