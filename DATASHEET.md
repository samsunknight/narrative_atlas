# Datasheet — the Narrative Atlas

## 1. Motivation & composition
The atlas is 149,341 narrative works (94,140 films, 22,978 novels, 32,223 television
programs), each scored on 162 attributes of narrative form across five layers (structure,
mood, genre, character arc, texture). The dataset is `data/atlas/century_frame_{film,book,tv}.parquet`
(one row per work; `idx`, `title`, `year`, `decade`, `medium`, and the layer-prefixed
attribute scores). A 30-attribute structural spine with column names shared across media
(used for adaptation, convergence, and crystallization) is at
`data/corpus/{film,book,tv}_structural_1890_2025.csv`. Works span 1890–2025 (capped at
2025); main analyses use 1915–2020. Titles are English Wikipedia page titles and may include
a disambiguation suffix (`"Casablanca (film)"`).

## 2. How the scores were generated (data generation, not replication)
Each attribute is scored by `gpt-4o-mini` at temperature 0 (greedy decoding, so an identical
summary returns an identical score), reading the work's English Wikipedia plot summary and
answering the exact human-survey question for that attribute on its ordinal scale (typically
1–5 or 1–7; moods, genres, and texture on a 0–100 intensity scale). The model receives no
title/date/genre metadata beyond the summary text. Re-running the scoring is **not** part of
replication and requires an OpenAI key (read from `OPENAI_API_KEY`; no key is shipped). The
model settings, prompt format, and scoring harness are in `code/` (`PROMPT.md`,
`rescore_oneat.py`, `rescore_corpus.py`), and the exact deployed prompt for every attribute is
in `data/validation/rescore_manifest.csv`.

## 3. Validation / human anchoring
Attributes are anchored to two human surveys approved by the University of Toronto Research
Ethics Board (protocol 46547): 751 readers (book survey; the HumanReader corpus) and 225
viewers (two film surveys; the HumanViewer corpus). The raw per-respondent responses are
**not** redistributed here (see `PRIVACY.md`); they remain with the companion HumanReader and
HumanViewer dataset papers. This package ships (a) the processed per-attribute machine-vs-human
statistics (`{movie,book}_attribute_validation.csv`, `T2_validation.csv`,
`genre_validation_layer.csv`, `rescore_manifest.csv`) and (b) the intermediate **per-work
human means** (`human_means_{film,book}.csv`): for each work × attribute, the mean human
rating and the rater count. Joining the means to the model scores reproduces the published
validation correlations exactly (see `reproduce.py`). For film the model side is the validated
per-work scoring in `film_llm_validation_scores.csv`; for book it is the released corpus
column. The eight attributes clearing r > 0.22 in **both** media (`cross_medium == True` in the
codebook) carry all cross-medium comparisons.

## 4. Matched external keys (frozen snapshots in `data/matched/`)
- **imdb_film_ratings.csv** — film → IMDb average rating and vote count, matched by
  normalized title + release year against the IMDb public dataset.
- **imdb_film_genres.csv** — film → IMDb genre tags (same match), for the genre-recovery AUCs.
- **adaptation_pairs.csv** — Wikidata "based on" (P144) film→literary-work pairs; matched to
  the corpus by normalized title yields the within-work adaptation set (437 pairs).

## 5. Known limitations
- **Summaries, not works.** Attributes are read from contemporary Wikipedia plot summaries,
  whose length and conventions drift over time; craft/texture attributes validate weakest. The
  driver residualizes the escalation index on summary length as a robustness check.
- **Coverage selection.** Wikipedia covers notable works, and notability is selected
  differently across eras, so per-decade samples are not random.
- **Validation asymmetry.** Stronger for film than for books; television inherits the film
  viewer-validation (no separate TV survey).
- **Television dating.** A television work is a single Wikipedia entry (a series or a notable
  episode). The `year` field is unreliable at both tails: undated entries (concentrated in the
  most recent decade) are placed at the corpus endpoint, and the ~46 entries dated before 1950
  are mislabeled — a year taken from a source film or the story's setting, or a non-television
  work (a novel, play, or film) misclassified as television — since broadcast television begins
  only in the late 1940s. Read television temporal trends on the **1950–2020 window** and do
  not trust an individual early TV year.

## 6. Reproducibility status
Every headline number reproduces from this package via `reproduce.py` (**96/96**). `[R]`
checks (corpus counts, structure/genre/texture validation, adaptation, reception, convergence,
crystallization, variance ratio, genre lifecycles, the Production-Code difference-in-differences,
and style-space geometry) are re-derived from the shipped tables. `[A]` checks (the mood
median r and the character-arc change r) are asserted against the shipped sweep values
(`mood_numbers.json`, `arc_findings.json`), since the per-rater mood/arc ratings are not
redistributed. No tolerances were widened to force a pass; the one documented correction (the
adaptation fantastical delta re-derives to 0.58, matching the paper) is noted in the report.

## 7. Distribution & license
See `LICENSE`. Survey-derived ratings are governed by the companion HumanReader / HumanViewer
releases; IMDb and Wikidata content are subject to their respective terms.
