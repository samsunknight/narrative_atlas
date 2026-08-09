# Datasheet — the Narrative Atlas

## 1. Motivation & composition
The atlas is 149,341 narrative works (94,140 films, 22,978 novels, 32,223 television
programs), each scored on 216 attributes of narrative form across eleven constructs (structure
& plot, setting, story shape, conflict, character, character arc, narration, mood, genre, texture). The dataset is `data/atlas/century_frame_{film,book,tv}.parquet`
(one row per work; `idx`, `title`, `year`, `decade`, `medium`, and the layer-prefixed
attribute scores). A 36-attribute structural spine with column names shared across media
(used for adaptation, convergence, and crystallization) is at
`data/corpus/{film,book,tv}_structural_1890_2025.csv`. Works span 1890–2025 for film and television; the novel corpus reaches back further (earliest
2025 cap; a tail of novels predates 1890, minimum year 1800). Main analyses use 1915–2020. Titles are English Wikipedia page titles and may include
a disambiguation suffix (`"Casablanca (film)"`).

**Cleaning.** Each work's medium is its Wikipedia classification. Before release we removed
644 works (0.4% of the corpus) whose entry identified a different medium — chiefly theatrical
films that had been carried in the television set (titles disambiguated as `"… (film)"`) —
together with the pre-1950 television entries, which carry a year taken from a source film or
the story's setting, or are non-television works misclassified as television, since broadcast
television begins only in the late 1950s. The removals were: television −617, book −20, film
−7. This is the only row-level filtering; no work is dropped for its scores.

## 2. How the scores were generated (data generation, not replication)
Each attribute is scored by `gpt-4o-mini-2024-07-18` at temperature 0 (greedy decoding, so an identical
summary returns an identical score), reading the work's English Wikipedia plot summary and
answering the exact human-survey question for that attribute on its ordinal scale (typically
1–5 or 1–7; moods, genres, and texture on a 0–100 intensity scale). The model receives no
title/date/genre metadata beyond the summary text. Re-running the scoring is **not** part of
replication and requires an OpenAI key (read from `OPENAI_API_KEY`; no key is shipped). The
model settings and prompt format are in `code/PROMPT.md`. The single authoritative prompt record is
`atlas_canonical/prompts.csv` — one row per atlas column × medium, giving the exact `deployed_prompt`
and its `canonical_prompt` — and `atlas_canonical/verify_prompts.py` is the guard that binds those
prompts to the shipped scores (it fails the build if any prompt is truncated, missing, or drifted).
`data/validation/rescore_manifest.csv` is a legacy/secondary registry of the same prompts.

## 3. Validation / human anchoring
Of the 216 main descriptive attributes, **198 are measured by the human survey** (structure & plot 51,
setting 2, story shape 5, conflict 6, character 8, character-arc 9, narration 6, mood 31, texture 37) and
**18 — the genre layer — against IMDb category tags** (AUC), since the survey carried only a
coarse genre checklist; **195 clear the validation bar**. The released dataset additionally carries
three reception attributes (documented in
the codebook `status` column), which sit outside the main descriptive constructs and are not analyzed. The survey attributes
are anchored to two human surveys approved by the University of Toronto Research
Ethics Board (protocol 46547): 714 readers (book survey; the HumanReader corpus) and 225
viewers (two film surveys; the HumanViewer corpus). The raw per-respondent responses are
**not** redistributed here (see `PRIVACY.md`); they will be released with the companion HumanReader and
HumanViewer dataset papers. This package ships (a) the processed per-attribute machine-vs-human
statistics (`genre_validation_layer.csv`, `rescore_manifest.csv`, and the per-attribute film and
book validation correlations in the codebook `attribute_dictionary.csv`) and (b) the intermediate
**per-work human means** (`human_means_book.csv`): for each work × attribute, the mean human
rating and the rater count. Joining the book means to the model scores reproduces the published
book-side validation correlations exactly (see `reproduce.py`); the film-side correlations are the
deployed-corpus values recorded in the codebook. The sixteen structural attributes validated in **both** media
carry the cross-medium geometry; the codebook's `cross_medium == True` flag marks these along
with the settings, story shapes, and the vs-self and vs-nature conflict types that also validate cross-medium.

**Recommended-use tiers.** The `tier` column grades each attribute by human-anchoring
strength: **A** (strongest), **B** (validated), and **C** (exploratory, released for
description and flagged for cautious use). The tiers map to how an attribute should be used.
Cross-medium film-versus-novel geometry uses the sixteen structural attributes validated in
both film and the novel (the codebook's `cross_medium` flag marks these plus the settings,
story shapes, and the vs-self and vs-nature conflict types that also validate cross-medium); within-medium and
film/television analyses may use the wider film-validated set (tiers A and B); tier C is
suited to description rather than headline inference. An attribute validates when its held-out
correlation with human judgment is significantly positive (95% CI excluding zero); a stricter robust
core (tiers A/B) requires the CI lower bound to clear r = 0.22.

## 4. Matched external keys (frozen snapshots in `data/matched/`)
- **imdb_film_ratings.csv / imdb_film_genres.csv** — film → IMDb average rating and vote
  count, and film → IMDb genre tags, matched by normalized title + release year against IMDb's
  public non-commercial dataset via `code/rebuild_imdb_match.py`; they remain subject to IMDb's
  terms. They drive the genre-recovery AUCs and the reception layer.
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
  only in the late 1950s. Read television temporal trends on the **1950–2020 window** and do
  not trust an individual early TV year.

## 6. Reproducibility status
Every headline number reproduces from this package via `reproduce.py` (**148/148** turnkey; **162/162** once the IMDb match files are rebuilt, see §4). `[R]`
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

*Prompt note:* six deployed prompts retain a verbatim "plot and reception" survey preamble; no reception text is ever supplied (plot summary only), and deleting the clause leaves scores unchanged (mean Δr 0.003).

## Full scored atlas (data release)

The complete per-work scored atlas (film/book/tv `*_atlas.csv` + pooled `century_frame.parquet`, ~106 MB) is distributed as a GitHub Release rather than tracked in git:

  https://github.com/samsunknight/narrative_atlas/releases/tag/v1.0-data

The repo itself ships the code, the `reproduce.py` harness (148/148 turnkey, 162/162 with the IMDb match files rebuilt), and the reproduce-scale data; download the release assets into `data/atlas/` for the full dataset.
