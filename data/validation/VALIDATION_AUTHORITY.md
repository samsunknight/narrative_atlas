# Validation authority — read this before quoting any validation r

**The single authoritative source of per-attribute validation is the codebook**
`data/validation/attribute_dictionary.csv` (and its identical copies
`atlas_canonical/codebook.csv` / `validation_summary.csv`). It carries the **post-workshop,
deployed** correlations — the r's of the prompts that actually scored the corpus.

## The rule that keeps getting broken

Prompts were **workshopped** (a prompt bake-off that rescued marginal attributes with better
wording). The deployed atlas uses the **winning** prompts, and the codebook records **their**
validation r's. Any other file that shows a **lower** number, or marks a codebook-validated
attribute as `fail`, is **pre-workshop and superseded** — it predates the rescue.

> **Direction is always toward the higher, rescued number. A file showing a lower r or a `fail`
> for an attribute the codebook validates is stale. Do not "correct" the codebook down to it.**

Concrete example that has caused repeated confusion: the ending emotional-release attribute
(pathos; survey item Q731). Pre-workshop it scored **0.22 ("fail")**; the workshop rescued it and
the deployed/codebook value is **0.36**. The 0.22 is the stale one.

## Quarantined pre-workshop files

Moved to `_preworkshop_stale/` so they cannot be mistaken for authoritative:

- `dropped_attr_validation.csv` — original (pre-rescue) validation verdicts. For attributes the
  workshop did not touch it agrees with the codebook (e.g. anagnorisis 0.325); for rescued
  attributes it shows the **lower pre-workshop** value (e.g. emotional_release 0.22).
- `dropped_attr_llm_scores.csv` — the pre-workshop val-set LLM scores. These correlate near zero
  with the deployed atlas columns **because they are a different (older) prompt**, not because the
  atlas is broken.

If you need to check an attribute's validation, read the codebook. If you need the deployed prompt,
read `rescore_manifest.csv`. Do not reach into `_preworkshop_stale/`.
