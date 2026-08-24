# Replication Package for "Measuring narrative form across a century reveals persistent medium signatures amid convergence"

This repository contains the code and released data to reproduce every number, table, and figure in the paper with a single command. The atlas is a scored corpus of 149,341 works (94,140 films, 22,978 novels, and 32,223 television programs) spanning 1890 to 2025, each read on 216 attributes of narrative form across eleven constructs (structure and plot, setting, story shape, conflict, character, character arc, narration, mood, genre, tone, and texture), of which 195 clear the validation bar. Scores are produced by a language model reading each work's English Wikipedia plot summary and answering the same questions put to human raters; the human anchor is two surveys of 711 readers and 225 viewers.

Reproduction is one command:

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
bash run_all.sh
```

`run_all.sh` rebuilds the structural spine from the canonical atlas and asserts it has not drifted, runs every analysis script (writing its outputs under `pnas-sub/analysis/out/`), regenerates the data tables from those outputs, runs the drift guards, re-derives every headline number from the released corpus through `reproduce.py` (117 checks on the shipped data; a further 14 require IMDb ratings and genres, which are not redistributed), and finally `verify_paper.py` recomputes each reported number and confirms the manuscript agrees. It ends `Done.`; any failed check is fatal. The package is self-locating: it sets `NARRATIVE_ATLAS_ROOT` to its own directory, so it reproduces wherever it is unpacked.

The clean, shared-column dataset for reuse lives in `atlas_canonical/`; the dense per-medium frames under `data/atlas/` are the reproduction inputs.

---

## Data Availability Statement

The package is self-contained for reproduction: every number in `reproduce.py` is recomputed from the tables shipped under `data/`. A few categories of input sit outside this repository, and none is required to reproduce the results.

- **The full scored atlas.** The per-medium dense tables (`data/atlas/century_frame_{film,book,tv}.parquet`, one row per work and one column per attribute) are the reproduction inputs, and the clean shared-column copies are in `atlas_canonical/data/`. The larger merged and CSV forms are distributed as GitHub Release assets rather than tracked in git, to keep the repository small.
- **The raw human ratings.** The atlas ships per-work *mean* human ratings only (`data/validation/human_means_book.csv`); no individual responses or rater identifiers are included. The rater-level survey data will be released, under their own research-ethics terms, with the companion dataset (the HumanReader and HumanViewer corpora).
- **The raw Wikipedia plot text.** The language-model scoring reads each work's plot summary from a Wikipedia snapshot. The scored corpus is the released output; the raw text is not redistributed here.
- **IMDb ratings and genres.** The two files `data/matched/imdb_film_{ratings,genres}.csv` are matched from IMDb's public non-commercial dataset by the shipped title and year via `code/rebuild_imdb_match.py`, and remain subject to IMDb's terms. They drive the genre-recovery and reception checks.

---

## Software Requirements

Python 3.11 or later. Dependencies are pinned in `requirements.txt`:

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
bash run_all.sh
```

---

## Contents

```
run_all.sh              single driver; reproduces every number, table, and figure (one command)
reproduce.py            headline verifier called by run_all.sh (117 checks on shipped data; 131 with IMDb metadata)
requirements.txt        Python dependencies
atlas_canonical/        the clean, shared-column dataset for reuse (start here)
  data/atlas_{film,book,tv}.parquet
  prompts.csv           the single authoritative prompt record (one row per column x medium:
                        deployed_prompt + canonical_prompt)
  verify_prompts.py     the prompt guard binding shipped prompts to shipped scores
                        (run by reproduce.py before any check)
  codebook.csv          219-row codebook (canonical name, construct, status, r, tier, media, scale)
  validation/           validation_summary.csv + VALIDATION.md
  provenance/           PROVENANCE.md, QUESTIONS.md, questions_and_prompts.csv
  README.md
code/                   the analysis suite: the canonical module, the A-series analysis scripts, and the reference verifiers (see "Pipeline" below)
data/
  corpus/               structural spine, column names shared across media
    {film,book,tv}_structural_1890_2025.csv
  atlas/                the reproduction inputs, one dense frame per medium
    century_frame_{film,book,tv}.parquet
  validation/           the human anchor and the codebook
    attribute_dictionary.csv        219-row codebook (layer, attribute, column, scale,
                                    validation r, recommended-use tier, cross_medium flag)
    human_means_book.csv            per-work MEAN human ratings (no individual responses)
    survey_atlas_crosswalk.csv, reliability_halves.csv, darkening_mask_film.csv,
    book_darkindex_pairs.csv, genre_validation_layer.csv, genre_reception.csv,
    rescore_manifest.csv, summary_lengths.csv, book_taxonomy_validation.csv,
    arc_change_validation.csv, newattr_final.csv, mood_numbers.json, arc_findings.json
  matched/              frozen external keys
    adaptation_pairs.csv, imdb_film_{ratings,genres}.csv (matched via code/rebuild_imdb_match.py)
outputs/                check_report.txt (written by reproduce.py)
```

`MANIFEST.md` gives a file-by-file inventory with provenance; `DATASHEET.md` documents the dataset and its recommended-use tiers; `LICENSE` states the release terms.

---

## Pipeline

The analysis runs in three stages, and only the last is required to reproduce the paper.

1. **Generation** (not re-run here). The corpus is scored by the language model. The exact model, settings, and prompt are in `code/PROMPT.md`; scoring requires the raw plot text and an API key, both held outside this release. Its outputs are the shipped tables under `data/`, so reproduction does not invoke it.

2. **Released intermediates.** The scored corpus, the per-work human means, and the matched external keys are the tables in `data/`, and they are the sole inputs to reproduction. `code/rebuild_spine_from_atlas.py` rebuilds the shared-scale structural spine from the atlas and the survey crosswalk, `code/rebuild_imdb_match.py` rejoins IMDb's public dataset, and `code/check_spine_atlas_sync.py` guards the spine against drift.

3. **Reproduction.** `reproduce.py` (called by `run_all.sh`) recomputes every headline quantity from the tables in `data/` and writes `outputs/check_report.txt`.

---

## Reproducing the Results

```
.venv/bin/python reproduce.py
```

This prints one line per quantity (`[PASS/FAIL][R/A] label  target=X  reproduced=Y`) and a final `117/117 passed` on the shipped data (14 further IMDb-dependent checks run only where IMDb ratings and genres are present, for 131 in total), and writes the same to `outputs/check_report.txt`. A check tagged `[R]` is re-derived from the released tables; a check tagged `[A]` is asserted against a shipped result artifact for the two layers (mood, character arc) whose raw ratings are not redistributed here.

---

## Output Files

`reproduce.py` writes `outputs/check_report.txt` (the full pass/fail log). `outputs/` is not tracked in git, since it is regenerated from the released `data/` tables, which are the single source of truth.
