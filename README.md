# The Narrative Atlas

A century of narrative style across film, book, and television, scored on more than 140
attributes of narrative form and anchored to human judgment. This repository is the
self-contained replication package and data release for the paper *The Narrative Atlas*.

The atlas covers **149,341 works** (94,140 films, 22,978 novels, 32,223 television
programs), 1890–2025, each scored on **142 attributes** across five layers — structure,
mood, genre, character arc, and texture — of which **130 clear the validation bar**. Scores
are produced by a language model reading each work's English Wikipedia plot summary and
answering the same questions put to human raters; the human anchor is two surveys (751
readers, 225 viewers).

## What is here

```
reproduce.py            one script; reproduces every headline number (96 checks)
requirements.txt        Python 3.11+ dependencies
code/                   how the scores were generated (transparency; not needed to reproduce)
  PROMPT.md             model, settings, prompt format
  rescore_oneat.py, rescore_corpus.py   the gpt-4o-mini scoring harness
data/
  atlas/                THE dataset — one dense table per medium
    century_frame_{film,book,tv}.parquet   one row per work; idx, title, year, decade,
                        medium, and one column per attribute (mood_*, genre_*, arc_*,
                        the visual/score/acting/dialogue texture columns, and the
                        structural survey items)
  corpus/               the 30-attribute structural spine, column names shared across
    {film,book,tv}_structural_1890_2025.csv   media (used for adaptation / convergence)
  validation/           the human anchor and the codebook
    attribute_dictionary.csv   the 142-row codebook (layer, attribute, column, scale,
                        validation r, tier, cross_medium flag)
    human_means_{film,book}.csv   per-work MEAN human ratings (no individual responses)
    film_llm_validation_scores.csv, genre_validation_layer.csv, rescore_manifest.csv,
    reliability_halves.csv, summary_lengths.csv, {movie,book}_attribute_validation.csv,
    T2_validation.csv, mood_numbers.json, arc_findings.json
  matched/              frozen external keys
    imdb_film_{ratings,genres}.csv, adaptation_pairs.csv
outputs/                written by reproduce.py (check_report.txt)
```

See `MANIFEST.md` for a file-by-file inventory with provenance, `DATASHEET.md` for the
dataset datasheet, and `PRIVACY.md` for the privacy and human-subjects statement.

## Using the atlas

```python
import pandas as pd
film  = pd.read_parquet("data/atlas/century_frame_film.parquet")
adict = pd.read_csv("data/validation/attribute_dictionary.csv")

# the 8 cross-medium-validated attributes that carry film-vs-novel claims:
cross = adict[adict.cross_medium].column.tolist()
casablanca = film.loc[film.title == "Casablanca (film)", cross]

# a genre's century-long lifecycle:
western = film.assign(dec=film.year // 10 * 10).groupby("dec")["genre_Western"].mean()
```

Titles are English Wikipedia page titles and may carry a disambiguation suffix
(`"Casablanca (film)"`). Cross-medium comparisons should keep to the eight
both-media-validated attributes (`cross_medium == True`); within-medium and
film/television analyses can use the wider film-validated set. The `tier` field grades
human-anchoring strength — A (strongest), B (clears the r>0.22 bar), C (below it) — and
tier C is for description, not headline inference. The codebook's `column` field gives the exact parquet
column for the clean-named layers and the eight cross-medium attributes; the remaining
structural rows are named by their survey question and matched through `definition`.

## Reproducing the paper

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python reproduce.py
```

This prints, and writes to `outputs/check_report.txt`, one line per check
(`[PASS/FAIL][R/A] label  target=X  reproduced=Y`) and a final `96/96 passed`. `[R]` checks
are re-derived from the shipped tables; `[A]` checks (the mood and character-arc validation
r) are asserted against the shipped sweep values, since the per-rater mood/arc ratings are
not redistributed. The one documented correction (the adaptation fantastical delta re-derives
to 0.58, matching the paper) is noted at the foot of the report; no tolerances were widened.

## Scope of this release

The scores themselves are the data; re-running the language-model scoring is **not** part of
replication and is not needed to reproduce any number here (the scoring harness and the exact
prompts are in `code/` and `data/validation/rescore_manifest.csv` for transparency). The raw,
per-respondent human
survey responses are **not** redistributed — they live with the companion HumanReader and
HumanViewer dataset papers — and this package ships only aggregated per-work means. See
`PRIVACY.md`.

## License

Code (`reproduce.py`) under MIT; data under CC BY 4.0, subject to the upstream terms of IMDb,
Wikidata, and the companion survey releases. See `LICENSE`.

## Figures

All figures in the paper (main text, Extended Data, and Supplementary) are provided as PNGs in
`outputs/figures/`, and the code that generates them is in `code/figures/`.

Regenerate the core figures from the released data:

```bash
python code/figures/build_fig_f1.py           # corpus coverage + attribute taxonomy
python code/figures/build_fig_f2.py           # human validation, all five layers
python code/figures/build_missing_figs.py     # adaptation, crystallization (line + heatmap), 1-D shifts, strongest planes
python code/figures/build_levels_diff.py      # style-space density grid + per-plane SI figures
python code/figures/fig_f6_crosscutting.py    # layer convergence + complementarity
```

These read only the released data (`data/corpus/*_structural_1890_2025.csv`, `data/atlas/*.parquet`,
`data/validation/*`, `data/derived/adaptation_deltas.csv`) and write to `outputs/figures/`. The
crystallization line reproduces `reproduce.py`'s `crys()` exactly (film 0.25 in the 1910s to 0.39 by
the 1980s). The remaining layer figures (genre lifecycles, the Production-Code tone comparison, the
Extended-Data exhibits, the PCA style-space map) additionally use analysis intermediates produced by
the scoring/reception pipeline; their generators are included in `code/figures/` and the rendered
figures are in `outputs/figures/`.
