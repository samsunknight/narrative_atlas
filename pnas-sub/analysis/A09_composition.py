"""A09 (PNAS rebuild, Phase 4 / S4b): near-full corpus-composition robustness.

Recovers original language and country of origin from Wikidata for as much of the corpus as resolves
(a resumable, cached pull, extending A14's stratified sample toward the full corpus), then re-estimates
the headline medium-vs-era ratio under composition restrictions and a fixed-composition reweighting.
Guardrail: the estimand stays the English-Wikipedia-documented set; we never claim a production census.

Resumable: run repeatedly; each pass works a wall-clock budget and caches, exiting with RESUME until the
whole corpus is queried, then computes the composition specs. Writes analysis/out/A09_composition.json.
Run from the style_evolves root.
"""
import os, sys, json, time, urllib.parse, urllib.request
import numpy as np, pandas as pd

ROOT = os.environ.get("NARRATIVE_ATLAS_ROOT", os.path.expanduser("~/uoft/style_evolves")); os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "code"))
from _methods import VAL16_KEYS, load_spine, WINDOW
OUT = os.path.join(ROOT, "pnas-sub", "analysis", "out")

SP = {m: load_spine(m) for m in ["film", "book", "tv"]}
titles = sorted(set(t for m in SP for t in SP[m].title.astype(str)))
print(f"[comp] {sum(len(SP[m]) for m in SP)} works, {len(titles)} unique titles")

ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "style-evolves-research/1.0 (samsun.knight@rotman.utoronto.ca) academic-metadata",
           "Accept": "application/sparql-results+json"}
BATCH = 150
def esc(t): return t.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").replace("\t", " ").replace("\r", " ")
def run_query(q, tries=3):
    data = urllib.parse.urlencode({"query": q, "format": "json"}).encode(); last = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(ENDPOINT, data=data, headers=HEADERS), timeout=90) as r:
                return json.load(r)
        except Exception as e:
            last = e; time.sleep([5, 15, 30][min(k, 2)])
    raise last

CACHE = os.path.join(OUT, "A09_comp_cache.json")
BUDGET = float(os.environ.get("A09_BUDGET_S", "80"))
if os.path.exists(CACHE):
    c = json.load(open(CACHE)); queried = set(c["queried"]); info = {k: v for k, v in c["info"].items()}
else:
    queried, info = set(), {}
def save():
    json.dump({"queried": sorted(queried), "info": info}, open(CACHE, "w"))

remaining = [t for t in titles if t not in queried]
print(f"[comp] {len(queried)} queried; {len(remaining)} remaining")
if remaining:
    start = time.time()
    for bi in range(0, len(remaining), BATCH):
        if time.time() - start > BUDGET: break
        chunk = remaining[bi:bi + BATCH]
        values = " ".join(f'"{esc(t)}"@en' for t in chunk)
        # language from P407 (language of work) with P364 (original language) fallback: books set P407,
        # not P364, so querying P364 alone leaves books near-uncovered.
        q = f"""SELECT ?name ?langLabel ?lang2Label ?countryLabel WHERE {{
  VALUES ?name {{ {values} }}
  ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> ; schema:name ?name .
  OPTIONAL {{ ?item wdt:P407 ?lang. }}
  OPTIONAL {{ ?item wdt:P364 ?lang2. }}
  OPTIONAL {{ ?item wdt:P495 ?country. }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}"""
        try:
            res = run_query(q)
        except Exception as e:
            print(f"[comp] batch FAILED: {e}", flush=True); time.sleep(2); continue
        for b in res["results"]["bindings"]:
            nm = b.get("name", {}).get("value")
            if nm is None: continue
            d = info.setdefault(nm, {"lang": None, "country": None})
            lang = b.get("langLabel", {}).get("value") or b.get("lang2Label", {}).get("value")
            if lang: d["lang"] = lang
            if b.get("countryLabel"): d["country"] = b["countryLabel"]["value"]
        queried |= set(chunk); save()
        print(f"[comp] {len(queried)}/{len(titles)} queried; {len(info)} matched", flush=True)
    save()

still = [t for t in titles if t not in queried]
if still:
    print(f"RESUME: {len(still)} titles remain -- rerun.", flush=True); sys.exit(0)
print(f"[comp] pull complete: {len(info)} of {len(titles)} titles matched", flush=True)

# ---- composition specs: re-estimate the medium-vs-era 16-attr ratio under restrictions ----
VALCM = [c for c in SP["book"].columns if any(k in c.lower() for k in VAL16_KEYS)
         and c in SP["tv"].columns and c in SP["film"].columns]
def tag(df):
    d = df.copy(); d["lang"] = d.title.astype(str).map(lambda t: (info.get(t) or {}).get("lang"))
    d["country"] = d.title.astype(str).map(lambda t: (info.get(t) or {}).get("country"))
    d["dec"] = (pd.to_numeric(d.year, errors="coerce") // 10 * 10)
    return d[(pd.to_numeric(d.year, errors="coerce") >= WINDOW[0]) & (pd.to_numeric(d.year, errors="coerce") <= WINDOW[1])]
D = {m: tag(SP[m]) for m in SP}

def ratio(frames, min_cell=30):
    """between-medium / within-medium decade-centroid variance on VALCM (mirrors A2's statistic).
    Decade cells below min_cell are dropped so a thin restricted subset does not inflate the within
    term and deflate the ratio toward 1 as an artifact."""
    mu = {m: {d: frames[m][frames[m].dec == d][VALCM].mean() for d in sorted(frames[m].dec.dropna().unique())
              if (frames[m].dec == d).sum() >= min_cell} for m in frames}
    med_cent = {m: pd.concat(mu[m].values(), axis=1).mean(axis=1) for m in frames}
    grand = pd.concat(med_cent.values(), axis=1).mean(axis=1)
    between = np.mean([np.sum((med_cent[m] - grand) ** 2) for m in frames])
    within = np.mean([np.mean([np.sum((mu[m][d] - med_cent[m]) ** 2) for d in mu[m]]) for m in frames])
    return float(between / within) if within else np.nan

specs = {}
specs["full_frame"] = round(ratio(D), 3)
eng = {m: D[m][D[m].lang == "English"] for m in D}
specs["english_original"] = round(ratio(eng), 3)
usuk = {m: D[m][D[m].country.isin(["United States of America", "United Kingdom"])] for m in D}
specs["us_uk_origin"] = round(ratio(usuk), 3)
match = {m: D[m][D[m].lang.notna()] for m in D}
specs["metadata_matched"] = round(ratio(match), 3)

coverage = {m: {"n_window": int(len(D[m])), "lang_known": int(D[m].lang.notna().sum()),
                "english_share": round(float((D[m].lang == "English").mean()), 3)} for m in D}
res = {"n_titles_matched": len(info), "coverage": coverage, "medium_era_ratio_specs": specs,
       "note": "Estimand = English-Wikipedia-documented works; not a production census. Ratio on the "
               "16-attr VAL basis, mirrors A2. English-original / US-UK / metadata-matched restrictions."}
json.dump(res, open(os.path.join(OUT, "A09_composition.json"), "w"), indent=2)
print("\nmedium-vs-era ratio under composition restrictions:")
for k, v in specs.items():
    print(f"  {k:20s} {v}")
print("coverage:", json.dumps(coverage, indent=0))
print("wrote out/A09_composition.json")
