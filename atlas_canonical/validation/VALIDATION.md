# Validation — the bar and the tiers

Every attribute in the atlas is validated against human ratings of the same works. The record is
`validation_summary.csv` (one row per attribute); this file explains how to read it.

## The bar

An attribute **validates** when the LLM instrument's score, applied to a plot summary, tracks the
human work-mean on a held-out validation set:

- Rating and count attributes: **Pearson r > 0.22** between model score and human mean (equivalently
  r² ≥ 0.05), computed on the medium(s) where the item was asked.
- Genre attributes: **ROC AUC ≥ 0.75** against IMDb genre tags.

The validation set is the works for which enough independent human raters answered the item to give a
stable work-mean (`n_human_validation` in the summary; typically ~144 for the recovered attributes).
`validated = True` when the tier is A/B/Headline/Validated, or when max(film_r, book_r) ≥ 0.22.

## The tiers

Two tier vocabularies appear, from two phases of the project; they mean the same thing:

| tier | vocabulary | reading |
|---|---|---|
| Headline / A | strong | r well above the bar; carries a headline finding |
| Validated / B | clears | r above the 0.22 bar |
| Marginal | at the floor | r at or just above 0.22 in at least one medium; interpret with care |
| C | below | does not clear the bar; not in the atlas unless retained for a documented reason |

**Marginal-tier attributes clear the bar by a smaller margin.** Several recovered attributes
(anagnorisis, inevitability, plot complexity, the unreliable narrator, narration switches) validate
but sit closer to the r > 0.22 floor than the headline attributes, and some validate in one medium
only. They are labeled `Marginal` so that this is visible; a plot summary carries them less strongly
than the concrete world and character attributes, so read them with that margin in mind.

## Cross-model agreement

Beyond the human validation, a second model (Claude Haiku 4.5) re-scores a ~23,000-work subsample on
the identical prompts. Agreement above the same r > 0.22 floor indicates the reading is a property of
the work rather than one model's idiosyncrasy; the per-attribute agreement is in
`../../rep_build/narrative_atlas/data/cross_model/`. This is corroboration alongside the human
validation, not a substitute for it.

## Media coverage

Attributes were not all asked in every medium. `media` in the summary lists where each was asked and
scored; `film_r` / `book_r` give the per-medium validation correlations (TV inherits the film
instrument and is validated where a TV human sample exists). An attribute asked only of readers
(e.g. the narration/point-of-view battery) validates on the book side only.
