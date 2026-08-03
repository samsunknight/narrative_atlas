# Deprecated / quarantined files

These files are retained for provenance only. **Do not use them.** They are stale,
alignment-fragile recomputes that disagree with the certified validation. The
authoritative validation sources are `data/validation/attribute_dictionary.csv`
(the canonical 161-row codebook), `data/validation/newattr_final.csv`, and
`data/validation/arc_change_validation.csv`.

- `data/validation/film_validation_corpus_basis.csv` — a corpus-basis recompute of
  per-attribute validation r. Its structure/mood/texture r's happen to agree with the
  codebook, but the file is alignment-fragile and its `arc`/`newattr` region wrongly
  shows peripeteia (`ending_reversal`) and `plot_linearity` as failing to validate.
  The codebook and `newattr_final.csv` score these at film r=0.29 and 0.24 (both
  validate). `reproduce.py` no longer reads this file; it reads the codebook and the
  arc-change values in `data/validation/arc_change_validation.csv` instead.
