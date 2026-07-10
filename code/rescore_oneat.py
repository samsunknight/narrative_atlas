# ---------------------------------------------------------------------------
# METHOD REFERENCE (not part of replication). Documents how the atlas attribute
# scores were generated with gpt-4o-mini (temperature 0). The scores themselves
# are the released data (data/atlas/, data/corpus/); re-running is optional and
# needs an API key (OPENAI_API_KEY in the environment) plus the plot-text inputs,
# which are NOT redistributed (bulk Wikipedia text). Every deployed per-attribute
# prompt is shipped in data/validation/rescore_manifest.csv (the `prompt` column).
# ---------------------------------------------------------------------------

"""Re-score the FULL corpus ONE-AT-A-TIME (the validated method, matching 05_mini_full.py),
all three media. Resumable (per work-x-concept cache in /tmp). Concept set is read straight
from the released corpus so the new columns line up for downstream recompute.

Usage:
  OPENAI_API_KEY=... PILOT=25 WORKERS=24 python3 rescore_oneat.py film      # small pilot
  OPENAI_API_KEY=... WORKERS=32 python3 rescore_oneat.py film book tv       # full run
"""
import pandas as pd, json, os, time, re, sys
from concurrent.futures import ThreadPoolExecutor
key = os.environ["OPENAI_API_KEY"]
from openai import OpenAI
client = OpenAI(api_key=key)

# Attribute/question registry (survey-item definitions). NOT redistributed; the deployed
# per-attribute prompts are in data/validation/rescore_manifest.csv.
REG = json.load(open("<registry.json>"))
PILOT = int(os.environ.get("PILOT", "0"))
WORKERS = int(os.environ.get("WORKERS", "24"))
CORP_DIR = "data/corpus"
# Plot-summary text inputs (NOT redistributed). Point these at your own {medium}
# plot-text tables, each with a title column and a text/plot/summary column.
PARQ = {"film": "<film_plot_text.parquet>",
        "book": "<book_plot_text.parquet>",
        "tv":   "<tv_plot_text.parquet>"}
CORPFILE = {"film": "film_structural_1890_2025.csv",
            "book": "book_structural_1890_2025.csv",
            "tv":   "tv_structural_1890_2025.csv"}

def scale_of(c):
    enc = c.get("option_encoding") or {}
    vs = [v for v in enc.values() if isinstance(v, (int, float))]
    return (min(vs), max(vs)) if vs else (1, 7)

def concepts_for(medium):
    cols = pd.read_csv(os.path.join(CORP_DIR, CORPFILE[medium]), nrows=1).columns.tolist()
    return [c for c in cols if c in REG]

def one(f, medium, title, text, cid):
    c = REG[cid]; lo, hi = scale_of(c); q = re.sub(r"\s+", " ", str(c["q"])).strip()
    sysmsg = (f'Rate this {medium} on ONE survey question from its Wikipedia plot and reception. '
              f'Question: {q} [{lo}=low to {hi}=high]. Return ONLY JSON {{"v": number}}.')
    for k in range(4):
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18", temperature=0,
                response_format={"type": "json_object"}, max_tokens=20,
                messages=[{"role": "system", "content": sysmsg},
                          {"role": "user", "content": f'{medium}: "{title}"\n{str(text)[:8000]}'}])
            json.dump(json.loads(r.choices[0].message.content), open(f, "w")); return
        except Exception:
            time.sleep(2 * (k + 1))

_MEDIA = sys.argv[1:] or ["film", "book", "tv"]
if os.environ.get("REVERSE"): _MEDIA = _MEDIA[::-1]
for medium in _MEDIA:
    CONC = concepts_for(medium)
    CACHE = f"/tmp/oneat_{medium}"; os.makedirs(CACHE, exist_ok=True)
    d = pd.read_parquet(os.path.expanduser(PARQ[medium])).reset_index(drop=True)
    tc = [c for c in d.columns if c in ("text", "plot", "summary")][0]
    tt = [c for c in d.columns if c in ("title", "name")]
    d["__t"] = d[tt[0]] if tt else d.index.astype(str)
    d = d[d[tc].notna() & (d[tc].astype(str).str.len() > 300)].reset_index(drop=True)
    if os.environ.get("REVERSE"): d = d.iloc[::-1]
    if PILOT:
        d = d.head(PILOT)
    jobs = [(f"{CACHE}/{i}__{ci}.json", medium, d.loc[i, "__t"], d.loc[i, tc], cid)
            for i in d.index for ci, cid in enumerate(CONC)
            if not os.path.exists(f"{CACHE}/{i}__{ci}.json")]
    print(f"[{medium}] works={len(d)} concepts={len(CONC)} calls_to_make={len(jobs)}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(lambda a: one(*a), jobs))
    dt = time.time() - t0
    rows = []
    for i in d.index:
        row = {"id": i, "title": d.loc[i, "__t"], "year": d.loc[i, "year"] if "year" in d else None}
        for ci, cid in enumerate(CONC):
            f = f"{CACHE}/{i}__{ci}.json"
            if os.path.exists(f):
                try: row[cid] = json.load(open(f)).get("v")
                except Exception: pass
        rows.append(row)
    out = f"/tmp/{medium}_structural_century_oneat.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    rate = len(jobs) / max(dt, 1)
    print(f"[{medium}] {len(jobs)} calls in {dt/60:.1f} min ({rate:.1f}/s) -> {out}", flush=True)
