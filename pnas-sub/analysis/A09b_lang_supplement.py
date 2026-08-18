"""A09b: language-field supplement for the composition pull.

A09's query read original language from P364 only, which is rarely set for books (books carry language
in P407), so book language coverage came back near-zero and the language-restricted ratios were
unreliable. This supplement re-queries P407 (language of work) for every matched title whose language is
still unknown, merges it into A09's cache, and recomputes the composition ratios. Resumable/cached.
Run from the style_evolves root (repeatedly until COMPLETE), then it rewrites A09_composition.json.
"""
import os, sys, json, time, urllib.parse, urllib.request
import numpy as np, pandas as pd

ROOT = os.environ.get("NARRATIVE_ATLAS_ROOT", os.path.expanduser("~/uoft/style_evolves")); os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "code"))
from _methods import VAL16_KEYS, load_spine, WINDOW
OUT = os.path.join(ROOT, "pnas-sub", "analysis", "out")

cache = json.load(open(os.path.join(OUT, "A09_comp_cache.json")))
info = cache["info"]  # title -> {lang, country}
# titles still missing a language, restricted to those in the corpus (we only need in-window works)
missing = sorted(t for t, d in info.items() if not d.get("lang"))
print(f"[a09b] {len(info)} matched; {len(missing)} still missing language -> query P407")

ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "style-evolves-research/1.0 (samsun.knight@rotman.utoronto.ca) academic-metadata",
           "Accept": "application/sparql-results+json"}
BATCH = 150
SUPP = os.path.join(OUT, "A09b_p407_done.json")
done = set(json.load(open(SUPP))) if os.path.exists(SUPP) else set()
def esc(t): return t.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").replace("\t", " ").replace("\r", " ")
def run_query(q, tries=3):
    data = urllib.parse.urlencode({"query": q, "format": "json"}).encode(); last=None
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(ENDPOINT, data=data, headers=HEADERS), timeout=90) as r:
                return json.load(r)
        except Exception as e:
            last=e; time.sleep([5,15,30][min(k,2)])
    raise last

todo = [t for t in missing if t not in done]
BUDGET = float(os.environ.get("A09B_BUDGET_S", "3000"))
start = time.time()
for bi in range(0, len(todo), BATCH):
    if time.time()-start > BUDGET: break
    chunk = todo[bi:bi+BATCH]
    values = " ".join(f'"{esc(t)}"@en' for t in chunk)
    q = f"""SELECT ?name ?langLabel WHERE {{
  VALUES ?name {{ {values} }}
  ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> ; schema:name ?name .
  OPTIONAL {{ ?item wdt:P407 ?lang. }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}"""
    try:
        res = run_query(q)
    except Exception as e:
        print("[a09b] batch failed:", e); time.sleep(2); continue
    for b in res["results"]["bindings"]:
        nm = b.get("name", {}).get("value")
        if nm and b.get("langLabel"):
            info.setdefault(nm, {"lang": None, "country": None})["lang"] = b["langLabel"]["value"]
    done |= set(chunk)
    json.dump(cache, open(os.path.join(OUT, "A09_comp_cache.json"), "w"))
    json.dump(sorted(done), open(SUPP, "w"))
    if (bi//BATCH) % 20 == 0: print(f"[a09b] {len(done)}/{len(todo)} P407-queried", flush=True)

still = [t for t in todo if t not in done]
if still:
    print(f"RESUME: {len(still)} titles remain for P407", flush=True); sys.exit(0)
print(f"[a09b] P407 supplement complete; {sum(1 for d in info.values() if d.get('lang'))} now have language")

# ---- recompute composition ratios with the corrected language ----
SP = {m: load_spine(m) for m in ["film", "book", "tv"]}
VALCM = [c for c in SP["book"].columns if any(k in c.lower() for k in VAL16_KEYS)
         and c in SP["tv"].columns and c in SP["film"].columns]
def tag(df):
    d = df.copy(); y = pd.to_numeric(d.year, errors="coerce")
    d = d[(y >= WINDOW[0]) & (y <= WINDOW[1])].copy()
    d["lang"] = d.title.astype(str).map(lambda t: (info.get(t) or {}).get("lang"))
    d["country"] = d.title.astype(str).map(lambda t: (info.get(t) or {}).get("country"))
    d["dec"] = (pd.to_numeric(d.year) // 10 * 10)
    return d
D = {m: tag(SP[m]) for m in SP}
def ratio(frames):
    mu = {m: {dc: frames[m][frames[m].dec == dc][VALCM].mean() for dc in sorted(frames[m].dec.dropna().unique())
              if (frames[m].dec == dc).sum() >= 30} for m in frames}
    cent = {m: pd.concat(mu[m].values(), axis=1).mean(axis=1) for m in frames}
    grand = pd.concat(cent.values(), axis=1).mean(axis=1)
    between = np.mean([np.sum((cent[m] - grand) ** 2) for m in frames])
    within = np.mean([np.mean([np.sum((mu[m][dc] - cent[m]) ** 2) for dc in mu[m]]) for m in frames])
    return float(between / within) if within else np.nan
specs = {"full_frame": round(ratio(D), 3),
         "english_original": round(ratio({m: D[m][D[m].lang == "English"] for m in D}), 3),
         "metadata_matched": round(ratio({m: D[m][D[m].lang.notna()] for m in D}), 3)}
coverage = {m: {"n_window": int(len(D[m])), "lang_known": int(D[m].lang.notna().sum()),
                "english_share": round(float((D[m].lang == "English").mean()), 3)} for m in D}
res = {"n_titles_matched": sum(1 for d in info.values() if d), "coverage": coverage,
       "medium_era_ratio_specs": specs,
       "note": "Original language from P407 (language of work) with P364 fallback, so books --- which "
               "rarely set P364 --- are covered. Estimand = English-Wikipedia-documented works, not a "
               "production census."}
json.dump(res, open(os.path.join(OUT, "A09_composition.json"), "w"), indent=2)
print("corrected composition ratios:", json.dumps(specs))
print("coverage:", json.dumps({m: coverage[m]["lang_known"] for m in coverage}))
