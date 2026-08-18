# Validation files — what is current, what is stale (READ BEFORE TRUSTING ANY r)

Written 2026-07-09 after stale artifacts here misled an analysis into thinking moods/genres/arcs were unvalidated. They are NOT. Guide:

## ⚠ The `rescore_manifest.csv` `r` column is NOT the validation record
It is populated only for **scalar (111) and descriptor (77)** attributes. It is **empty for all mood (31), genre (18), and arc (9)** attributes. **Empty `r` here does NOT mean "unvalidated."** The real validation r's for moods/genres/arcs live in `~/uoft/style_evolves/ATLAS_VALIDATION_MASTER.md §11` (the comprehensive sweep):
- moods validate (film 24–28/31, median r≈0.39; book 27/31)
- genres 10/10 film (median 0.68, AUC 0.75–0.93), 8/8 book
- descriptors 65/94 from plot
- character-arc *change* r=0.37–0.48 (§11a)
Final validated set ≈ **150 film / 90 book** (≥Marginal). Cite §11t, not §0/§3–6 of that doc (those are the pre-sweep snapshot, also stale).

## Files here
- `movie_attribute_validation.csv`, `book_attribute_validation.csv` — **BASE-30 only** (the pre-sweep 30-attribute battery). Superseded by the sweep for the full atlas; still referenced by the old `validate_movies.py`/`validate_books.py` exhibits. Do NOT read these as "the atlas validation" — they cover ~30 of ~150 attributes.
- `film_llm_validation_scores.csv`, `human_means_{film,book}.csv` — validation ground truth (LLM scores + human survey means).
- `genre_validation.csv` (in `results/tables/`) — genre AUCs (0.75–0.78 vs IMDb tags) — proof genres validate.
- `atlas_master_matrix.csv` — the concept × source × medium validity matrix (validates_via per attribute). Better single source than the manifest `r` column for cross-layer validity.
- `rescore_manifest.csv` — the DEPLOY manifest (attr × prompt × tier × media × deploy). Its `tier` column is meaningful; its `r` column is partial (see above).

## SI-robustness derived aggregates (added 2026-07-10; consumed by replication/reproduce.py)
Two PII-free aggregates were derived from local raw sources that are NOT shipped (raw Wikipedia plot text; raw per-rater survey rows). They are built by `replication/build_validation_aggregates.py` and let the SI robustness numbers be recomputed from the release:
- `summary_lengths.csv` — `idx, medium, n_char`: character length of each work's plot summary, joined to the atlas by normalized title+year (idx == corpus id). Feeds the summary-length control and the surface-feature baseline (SI §S1.4).
- `reliability_halves.csv` — `attribute, medium, r_halfsplit, n_raters, n_works`: film per-rater split-half half-mean correlation (500-partition average, seed 0). The driver applies Spearman–Brown `2r/(1+r)` to recover the `r^2` reliability ceiling (SI §S1). Only the aggregate correlation ships — never individual rater rows.

⚠ **The SI robustness numbers do not all reproduce from the released corpus.** The persistence spectrum (Tables S2/S3), the escalation rise and surface `r^2`(sci-fi) (§S1.4), and the genre-composition decomposition (§S1.5) were computed on the **superseded** `deprecated/data/corpus/*_structural_century.csv` (pre-rescoring; non-sci-fi films mis-scored ~4/7 on sci-fi). The released, corrected corpus gives materially different values — see the "SI-TEXT RECONCILIATION NEEDED" ledger printed by `reproduce.py`.

## The lesson
Before concluding an attribute is unvalidated, check `ATLAS_VALIDATION_MASTER.md §11` + `atlas_master_matrix.csv`, NOT the manifest `r` column or the base-30 CSVs.
