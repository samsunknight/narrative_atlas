## Replication Package for "A Human-Validated Atlas of Narrative Form across a Century of Literature, Film, and Television"

This repository contains the code and released data to reproduce every headline number, figure, and table in the paper. The atlas is a scored corpus of 149,341 works (94,140 films, 22,978 novels, and 32,223 television programs) spanning 1890 to 2025, each read on more than 140 attributes of narrative form across five layers (structure, mood, genre, character arc, and texture), of which 129 clear the validation bar. Scores are produced by a language model reading each work's English Wikipedia plot summary and answering the same questions put to human raters; the human anchor is two surveys of 714 readers and 225 viewers. A single driver, `reproduce.py`, regenerates 95 checked quantities from the released tables alone; a further 14 IMDb-dependent checks re-enable once the IMDb match files are rebuilt (see Data Availability).

---

## Data Availability Statement

The package is self-contained for reproduction: every number in `reproduce.py` is recomputed from the tables shipped under `data/`. Three categories of input sit outside this repository, and none is required to reproduce the results.

- **The full scored atlas.** The per-medium dense tables (`data/atlas/century_frame_{film,book,tv}.parquet`, one row per work and one column per attribute) are the released dataset and are included here. The larger merged and CSV forms are distributed as GitHub Release assets rather than tracked in git, to keep the repository small; download them into `data/atlas/` for the complete dataset.
- **The raw human ratings.** The atlas ships per-work *mean* human ratings only (`data/validation/human_means_{film,book}.csv`); no individual responses or rater identifiers are included. The rater-level survey data will be released, under their own research-ethics terms, with the companion dataset paper (the HumanReader and HumanViewer corpora).
- **The raw Wikipedia plot text.** The language-model scoring reads each work's plot summary from a Wikipedia snapshot. The scored corpus is the released output; the raw text is not redistributed here.
- **IMDb ratings and genres.** IMDb's data is under a non-commercial license, so the two files `data/matched/imdb_film_{ratings,genres}.csv` are not redistributed. The 14 checks that use them (genre recovery against IMDb tags, and the reception analysis) are skipped by default, leaving 95 checks that run turnkey. To enable them, download IMDb's public non-commercial dataset and run `code/rebuild_imdb_match.py`, which rejoins IMDb to the corpus by the shipped title and year and writes the two files back; `reproduce.py` then runs all 109 checks.

---

## Software Requirements

Python 3.11 or later. Dependencies are pinned in `requirements.txt`:

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python reproduce.py
```

---

## Contents

```
reproduce.py            single driver; reproduces every headline number (95 checks turnkey; +14 IMDb-dependent, see Data Availability)
requirements.txt        Python dependencies
code/                   analysis code (see "Pipeline" below)
data/
  corpus/               structural spine, column names shared across media
    {film,book,tv}_structural_1890_2025.csv
  atlas/                the released dataset, one dense table per medium
    century_frame_{film,book,tv}.parquet
  validation/           the human anchor and the codebook
    attribute_dictionary.csv        141-row codebook (layer, attribute, column, scale,
                                    validation r, recommended-use tier, cross_medium flag)
    human_means_{film,book}.csv     per-work MEAN human ratings (no individual responses)
    film_llm_validation_scores.csv, book_attribute_validation.csv, reliability_halves.csv,
    genre_validation_layer.csv, rescore_manifest.csv, summary_lengths.csv,
    mood_numbers.json, arc_findings.json
  matched/              frozen external keys
    adaptation_pairs.csv            (IMDb match files not shipped; rebuild via code/rebuild_imdb_match.py)
  derived/              adaptation_deltas.csv
outputs/                check_report.txt (reproduce.py); figures and tables regenerate
                        under results/ (code/run_all.py) and are not tracked in git
```

`MANIFEST.md` gives a file-by-file inventory with provenance; `DATASHEET.md` documents the dataset and its recommended-use tiers; `LICENSE` states the release terms.

---

## Pipeline

The analysis runs in three stages, and only the last is required to reproduce the paper.

1. **Generation** (not re-run here). The corpus is scored by the language model, and the book validation is computed from the raw survey. These steps are provided for transparency as `code/rescore_corpus.py`, `code/rescore_oneat.py`, and `code/validate_books.py`, with the exact model, settings, and prompt in `code/PROMPT.md`. Each requires inputs held outside this release (the raw plot text, an API key, or the rater-level survey) and each is headed with a note to that effect. Their outputs are the shipped tables under `data/`, so reproduction does not invoke them.

2. **Released intermediates.** The scored corpus, the per-work human means, and the matched external keys are the tables in `data/`, and they are the sole inputs to reproduction.

3. **Reproduction.** `reproduce.py` recomputes every headline quantity from the tables in `data/` and writes `outputs/check_report.txt`; `code/run_all.py` additionally regenerates every figure and table.

---

## Reproducing the Results

To reproduce the checked numbers:

```
.venv/bin/python reproduce.py
```

This prints one line per quantity (`[PASS/FAIL][R/A] label  target=X  reproduced=Y`) and a final `95/95 passed` (or `109/109` once the IMDb match files are rebuilt), and writes the same to `outputs/check_report.txt`. A check tagged `[R]` is re-derived from the released tables; a check tagged `[A]` is asserted against a shipped result artifact for the two layers (mood, character arc) whose raw ratings are not redistributed here.

To regenerate the figures and tables:

```
.venv/bin/python code/run_all.py
```

which runs `reproduce.py`, the figure builders in `code/figures/`, and the table scripts (`robustness_crossmedium_sets.py`, `run_subsets.py`, `reception_bootstrap.py`), writing to `outputs/`.

---

## Output Files

`reproduce.py` writes `outputs/check_report.txt` (the full pass/fail log). `code/run_all.py` regenerates the paper's figures and the CSV and TeX table fragments under `results/`. Neither `outputs/` nor `results/` is tracked in git, since both are regenerated from the released `data/` tables, which are the single source of truth.
