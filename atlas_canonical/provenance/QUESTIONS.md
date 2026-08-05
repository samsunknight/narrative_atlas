# Narrative Atlas — Questions & Prompts Provenance

`questions_and_prompts.csv` is the "which questions were used" table for the
Narrative Atlas. It has **one row per attribute for all 186 canonical attributes** in
`atlas_canonical/codebook.csv`, and for each gives:

> **Prompt authority:** the single authoritative record of the exact LLM prompts is
> `atlas_canonical/prompts.csv` (one row per atlas column × medium, with `deployed_prompt`
> and `canonical_prompt`), guarded by `atlas_canonical/verify_prompts.py`. The `llm_prompt`
> column below is a convenience view; on any discrepancy, `prompts.csv` governs.


| column | meaning |
|---|---|
| `canonical_column` | the atlas attribute id (matches the codebook) |
| `construct` | the codebook construct family (genre, mood, texture, structure, …) |
| `media` | media the attribute is scored on (film / book / tv) |
| `survey_question` | the **verbatim human survey question** the attribute derives from |
| `response_scale` | the human response format, plus which option/pole/arc-point this attribute picks out |
| `survey_source` | which survey file/question the verbatim text came from |
| `llm_prompt` | the **verbatim LLM scoring prompt** used to score the attribute corpus-wide |
| `prompt_source` | which file the LLM prompt came from |

## How the human survey question maps to the deployed LLM prompt

Two human instruments were fielded, a film viewer survey and a book reader survey. Each atlas attribute began as an
item on one of these surveys. To score the whole corpus (works no human rated), each survey item was
rewritten as a short LLM prompt that scores one work from its plot summary and returns
`JSON {"v": number}`. The corpus deployment used **gpt-4o-mini-2024-07-18, temperature 0,
`response_format=json_object`, max_tokens 20**, system message = the prompt, user message =
`"{title}\n{plot_text[:8000]}"`.

The rewrite generally **preserves the survey wording and scale**. Examples:

- *Likability* — survey Q710 "How likable was this protagonist? By likable, we mean easy to
  relate to, empathize with, and root for." → prompt "Based only on this {m}'s plot, answer: How
  likable was this protagonist? … 1-5."
- *Select-all taxonomies* (genre, mood, visual/score/acting texture, marginalized-identity) — the
  survey asks one select-all question; the atlas scores **each option as its own attribute**
  (`response_scale` names the option). The prompt asks a 0-100 intensity for that single option.
- *Setting when/where* — the survey select-all becomes one 0-100 "out of 100 viewers, how many
  would say it takes place {option}" prompt **per option**.
- *Two-prompt ensembles* — `plot_linearity` and `ending_reversal` were deployed as the **average of
  two prompts**; both are stored in `llm_prompt`, joined by `||ENSEMBLE-AVG||`.

`{m}` in a prompt is the medium token, substituted at deployment to **movie / TV show / book**.

## Sources (in the priority order used)

**LLM prompts** (`prompt_source`):
1. `atlas_canonical/validation/validation_summary.csv` — 161 attributes (`{m}`-templated form; the
   original instrument, before the 25 coverage-expansion attributes drawn from sources 2–3, for 186
   in all — 173 main descriptive plus 13 reception/demographic).
2. the verified deployed-prompt set — 23 residual attributes, harvested verbatim from the deployed
   request chunks (medium-substituted; the film variant is quoted unless the attribute is book-only).
3. a select-all template — 2 attributes (`setting_when`, `setting_where`), with the option phrases
   listed in the cell.

**Survey questions** (`survey_source`): verbatim text is taken from the raw survey headers (row 2 =
question text), matched to each attribute by exact-column slug (for the 32 structure attributes whose
id is a slugified survey header) or by a distinctive question phrase (for family/taxonomy and
coverage-expansion attributes), and confirmed against the shipped `survey_atlas_crosswalk.csv`
attribute↔question↔qid mapping.

## Coverage

- **Survey question present: 178 / 186.**
- **LLM prompt present: 186 / 186.**

The **8 attributes with no survey question** are all genre attributes not offered on the survey:
`genre_Animation, genre_Documentary, genre_War, genre_Historical, genre_Musical, genre_Adventure,
genre_Crime, genre_Family`. The Movie Genome Survey's genre item (Q724) offered only 10 genres
(Romance, Horror, Science Fiction, Western, Drama, Mystery, Thriller, Fantasy, Action, Comedy). These
8 additional genres were never human-surveyed; they are LLM-scored 0-100 and validated against the
corresponding **IMDb genre label** (ROC AUC), not against a human survey response. Their
`survey_question` is intentionally blank and the reason is recorded in `survey_source`.

No prompt or question was fabricated: any cell that a source did not supply is left blank and flagged.
