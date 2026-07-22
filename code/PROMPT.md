# How the atlas was scored — model, prompt, and settings

This documents how the attribute scores were generated, so the measurement is auditable.
**Re-running the scoring is not part of replication** — the released scored corpus
(`data/atlas/`, `data/corpus/`) is the data, and every headline number reproduces from it via
`reproduce.py` without any model call.

## Model & decoding
- Model: `gpt-4o-mini-2024-07-18` (OpenAI).
- Temperature: 0 (greedy decoding), so the same summary returns the same score on every call.
- Input: the work's English Wikipedia plot summary only (truncated to 8,000 characters). No
  title, date, or genre metadata is provided beyond the title string used as a header.
- Output: a single value on the attribute's ordinal scale, returned as JSON `{"v": number}`;
  non-conforming replies are dropped and retried.
- `gpt-4o-mini` was benchmarked against `gpt-4o` on the validation works and recovers
  comparable validation correlations at a fraction of the cost.

## The prompts
Every deployed attribute carries its exact prompt in
`data/validation/rescore_manifest.csv` (the `prompt` column, all 254 rows), together with its
layer, response scale (`lo`,`hi`), validation `r`, and tier. The question text and anchors are
taken verbatim from the human-survey item for that attribute; the noun changes by medium
(`{m}` → "movie" / "book" / "TV show"). Example (science-fictional world):

```
Consider this {m}. How science-fictional was the world of the movie? By science-fictional we
mean a world … Respond 1 (not at all) to 7 (extremely). Return ONLY JSON {"v": number}.
```

## Batching
- **Validation works:** one attribute per call (accuracy priority) — see `rescore_oneat.py`.
- **Full corpus:** manifest-driven scoring of every deployed attribute — see
  `rescore_corpus.py`. The two settings track human judgment equally well, so the batching
  choice does not affect what is measured.

## Scripts in this folder (method reference; not runnable standalone)
- `rescore_oneat.py` — one-attribute-per-call scoring (the validated method).
- `rescore_corpus.py` — full-corpus scoring from the manifest.

Both read the API key from the environment (`OPENAI_API_KEY`; no key ships here) and take the
plot-summary text tables as input. Those plot-text tables are **not** redistributed (bulk
Wikipedia text; the derived scores are the released data), so the paths are left as
placeholders at the top of each script. To re-score, set `OPENAI_API_KEY` and point the
`PARQ` / registry placeholders at your own plot-text tables.
