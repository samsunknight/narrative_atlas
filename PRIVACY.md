# Privacy and human-subjects statement

## No personally identifiable information is distributed

This package contains **no** individual survey responses, respondent identifiers,
demographics, free text, IP addresses, or platform (Reddit / Qualtrics / Prolific)
metadata. Every file was screened for these before release.

The human anchor enters the package only in **aggregated** form:

- `human_means_film.csv`, `human_means_book.csv` — for each work × attribute, the **mean**
  human rating and the rater count `n_raters`. No respondent rows, no respondent keys.
- `film_llm_validation_scores.csv`, `reliability_halves.csv`,
  `{movie,book}_attribute_validation.csv`, `T2_validation.csv` — model scores and
  already-summarized validation statistics (correlations, counts).

These aggregates are what make the validation independently re-derivable from this package
alone (joining the per-work means to the model scores reproduces the published validation
correlations inside `reproduce.py`).

## The `n_raters = 1` case

Rater counts per work × attribute are small (median 2, up to 6 for film and 8 for book), and
some are 1. Where `n_raters = 1`, that work's mean equals a single anonymous rater's ordinal
response to a question **about a public work** (e.g. "how science-fictional is this film,
1–7"). This carries no re-identification risk: there is no rater identifier, no demographic
field, and no key linking one rater's responses across works, so an individual's response set
cannot be reconstructed and no person can be identified. This is a standard aggregate release.

## Provenance of the human data

Ratings come from two surveys approved by the University of Toronto Research Ethics Board
(protocol 46547): a book survey (751 readers; the HumanReader corpus) and two film surveys
(225 viewers; the HumanViewer corpus). Participants gave informed consent and were
compensated. The **raw** per-respondent responses remain with those companion dataset papers
and are **not** part of this release.

## Works and external content

Work titles are English Wikipedia page titles (public), and may include disambiguation
suffixes such as `"Casablanca (film)"`. No plot-summary text is redistributed — only the
numeric attribute scores derived from it. IMDb-derived ratings/genres and Wikidata-derived
adaptation pairs are frozen public-source snapshots, subject to their providers' terms.
