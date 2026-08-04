# Provenance — from the raw survey to the scored atlas

Every attribute in the atlas is the output of the same five-stage pipeline. The chain below holds
for all 186 columns; the per-attribute prompt and validation correlation are in
`../validation/validation_summary.csv`, and the per-attribute source column in `../codebook.csv`.

## The chain

1. **Raw human survey.** Readers and viewers were assigned a complete novel or film and rated it on
   a battery of narrative attributes.
   - the book reader survey — 714 readers (HumanReader).
   - the film viewer surveys — two rounds that pool to 225 viewers (HumanViewer).

   Respondent-level data are governed by the survey's consent and are not part of this release; only
   aggregates flow downstream.

2. **Per-work human targets.** Each attribute is reduced to a work-level human mean (select-all items
   to a 0–100 proportion, rating items to a mean), with rater counts and reliability. Only these
   per-work aggregates flow downstream.

3. **Validated instrument.** For each attribute an LLM answers the *same* survey question from the
   work's plot summary, and the answer is kept only where it correlates with the human mean above the
   validation bar (see `../validation/VALIDATION.md`). The exact prompt for every attribute is the
   `prompt` field of `validation_summary.csv`, which also records prompt selection and the bake-off.

4. **Full-corpus scoring.** The validated prompt is applied to every work with an English Wikipedia
   plot summary, deployed model **`gpt-4o-mini-2024-07-18`** (temperature 0, JSON response,
   8,000-character plot truncation). The plot text is the English Wikipedia database dump; each work
   is linked to its own plot text by normalized title+year, not by row position.

5. **Cross-model check.** A randomized ~1/6 subsample (23,262 works, all three media) is re-scored on
   the identical prompts with an independently trained model, **`claude-haiku-4-5-20251001`**, to
   confirm the century-scale readings are properties of the works, not one model's idiosyncrasies.

6. **This atlas.** The full-corpus scores are assembled into `../data/atlas_{film,book,tv}.parquet`,
   with column names normalized to one shared scheme, unscored placeholder items dropped, the
   duplicate narrator-count resolved, and each attribute tagged with its construct and status.

## Model-inferred protagonist demographics (`released_demographic`)

The ten `identity_*` columns are produced by the same pipeline — the model infers a protagonist's
race, religion, ethnicity, gender identity, sexual identity, disability, socioeconomic status,
citizenship, language, and marginalized-group membership from the plot summary, validated against
readers' and viewers' own perceptions of the same. They are **released for completeness and not
analyzed** in any paper. These are sensitive inferences drawn from sparse textual cues; a plot
summary rarely states a character's demographics, so both the human labels and the model scores are
perception-based approximations. They should not be treated as ground truth about a work's
characters, and any downstream use must carry an explicit ethics and limitations statement.

## What is NOT in the atlas

Free-text responses, respondent identifiers and demographics, and survey items that do not reduce to
one work-level scalar (per-narrator repeats, branching conditionals) are excluded by design, as are
the evaluative importance and enjoyment survey items that the instrument does not score.
