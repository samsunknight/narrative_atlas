# Narrative Atlas — Authoritative Dataset (v1)

The Narrative Atlas: every work in the corpus scored on a human-validated instrument of narrative
form, for film, the novel, and television. This folder is the authoritative release; read the
dataset from here.

## What is here

```
atlas_canonical/
  codebook.csv                 one row per attribute: canonical name, construct, status, r, tier, media, scale, definition, source column
  data/
    atlas_film.parquet         94,140 films   × attributes   (idx, title, year, decade, medium, + attributes)
    atlas_book.parquet         22,978 novels  × attributes
    atlas_tv.parquet           32,223 TV works × attributes
  validation/
    validation_summary.csv     per attribute: validated?, tier, film_r, book_r, metric, human-N, prompt
    VALIDATION.md              how validation was done and how to read the tiers
  provenance/
    PROVENANCE.md              the lineage of every column, from the raw survey to the scored atlas
```

## The one thing to know: column names are canonical and shared across media

The same attribute has the **same column name** in all three media files (`plot_driven`,
`narrator_unreliable`, `inevitability`, …), unprefixed. Where an attribute does not apply to a
medium, the column is simply absent from that medium's file. Join film, book, and TV on the shared
column names.

## How to use it (downstream papers)

```python
import pandas as pd
book = pd.read_parquet("atlas_canonical/data/atlas_book.parquet")
cb   = pd.read_csv("atlas_canonical/codebook.csv")
# take only the main descriptive constructs, validated attributes:
use = cb[(cb.status=="main") & (cb.validated)].canonical_column
X   = book[[c for c in use if c in book.columns]]
```

## Attribute status — read before using

Each attribute carries a **status** in the codebook that governs how it may be used:

| status | count | meaning |
|---|---|---|
| `main` | 173 | Descriptive attributes of narrative form, in one of ten constructs. The atlas proper. |
| `appendix_reception` | 3 | Reception/salience judgments (how much an audience enjoyed the work; which elements mattered most). Released and validated, but **evaluative, not descriptive** — kept out of the main descriptive constructs and used only in a supplementary analysis. |
| `released_demographic` | 10 | Model-inferred protagonist demographics (race, religion, ethnicity, gender identity, sexual identity, disability, socioeconomic status, citizenship, language, marginalized-group membership). **Released for completeness but not analyzed.** These are sensitive, error-prone inferences from a plot summary; see `provenance/PROVENANCE.md` and the ethics note before any use. |

## Constructs (main, descriptive)

structure (51), texture (37), mood (31), genre (18), character-arc (9), **character (8)**,
conflict (6), **narration (6)**, story-shape (5), setting (2) — **ten constructs, 173 attributes,
164 of them validated** (r > 0.22, or ROC AUC ≥ 0.75 for genre).

## Provenance and validation

Every column traces from the raw reader/viewer survey through a validated LLM instrument to the
full corpus; see `provenance/PROVENANCE.md` for the lineage and `validation/VALIDATION.md` for the
bar and the tiers.
