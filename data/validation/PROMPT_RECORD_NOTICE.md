# rescore_manifest.csv — LIVE input, pending regeneration (do not delete)

This file is still read by `reproduce.py` and ~25 analysis/figure scripts, so it cannot be moved.
It carries some **truncated** prompt strings and is therefore NOT the prompt source of truth —
`atlas_canonical/prompts.csv` is. In the post-rescore cascade this file is **regenerated from
`atlas_canonical/prompts.csv`** so it becomes consistent (de-truncated) with the canonical set;
until then, treat `atlas_canonical/prompts.csv` as authoritative for any prompt question.
