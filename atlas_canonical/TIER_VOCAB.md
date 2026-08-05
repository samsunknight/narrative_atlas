# Attribute tier vocabulary — READ BEFORE FILTERING ON `tier`

The codebook carries **two** tier columns. They use overlapping letters with **different meanings**, so
read this before writing any `tier`-based selection.

## `tier` — LEGACY, frozen
The original mixed scheme: `Headline`, `Validated`, `Marginal`, `A`, `B`, `C`, `--`. In this column
**`C` means *below the validation bar*** (6 attributes). This column is **frozen** for backward
compatibility: the held-out re-validation (Aug 2026) did **not** change it, so existing downstream code
that reads `tier` (e.g. `tier != "C"` to drop the 6 unusable attributes) behaves exactly as before.

## `heldout_tier` — CURRENT, recommended
The held-out re-validation scheme (see `HELDOUT_REVALIDATION.md`, `attribute_registry.csv`):
| value | meaning | n |
|---|---|---|
| `A` | robust — CI lower bound > 0.30 | 81 |
| `B` | validated — CI lower bound > 0.22 | 45 |
| `C` | **marginal** — point estimate clears 0.22 but its CI touches it | 53 |
| `drop` | below the bar — do not use as a validated measure | 7 |

## ⚠ The collision that will bite you
**`C` is below-bar in `tier` (6 attrs) but *marginal* in `heldout_tier` (53 attrs).** A filter written as
`tier != "C"` and silently re-pointed at `heldout_tier` flips from excluding 6 unusable attributes to
excluding 53 usable-but-marginal ones **while admitting the 7 that should be dropped**. Nothing errors;
the numbers just move. When migrating:
- To keep the old behaviour, keep reading `tier`.
- To adopt the held-out tiers, read `heldout_tier` and use **`heldout_tier in {"A","B"}`** for the
  robustly-validated core (126 attrs), treat `C` as marginal/exploratory, and exclude `drop`.
  Do **not** reuse a `!= "C"` rule.
- Per-attribute `heldout_r`, `heldout_ci_lo`, `heldout_ci_hi` are in the codebook; full detail in
  `attribute_registry.csv`.

## Guard
Run `python atlas_canonical/check_tiers.py` after any codebook edit. It asserts both columns are present,
every value is in the documented vocabulary, and the downstream and replication-build copies agree.
