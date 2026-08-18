"""A04 (PNAS rebuild, Phase 2 / T1): classify each television work by UNIT TYPE.

Television is scored with the film-validated instrument and its corpus mixes entries of different
scope --- long-running series, single miniseries, individual episodes, television films. Because the
mixture shifts over the century (Wikipedia documents far more series-level pages after ~2000), a
convergence result read off the full television corpus can be an artifact of that changing mixture.
This module assigns a defensible unit type to every television work so the convergence analysis can
be rerun within unit type and reweighted to a fixed mixture.

Method: for each enwiki article title we read Wikidata ``instance of`` (P31) via the public SPARQL
endpoint (the same title->entity linkage A14 uses, ~99% matched), map the P31 labels to a unit type,
and fall back to the title's parenthetical marker where Wikidata has no usable P31; anything left is
``unknown`` (its share is reported, not hidden). This is metadata labeling from public catalogs, not
a human-subject rating task. Writes analysis/out/tv_unit_audit.csv and A04_tv_unit_type.json.

Run from the style_evolves root: ``.venv/bin/python pnas-sub/analysis/A04_tv_unit_type.py``.
"""
import os, sys, json, time, urllib.parse, urllib.request
import numpy as np, pandas as pd

ROOT = os.environ.get("NARRATIVE_ATLAS_ROOT", os.path.expanduser("~/uoft/style_evolves"))
os.chdir(ROOT)
OUT = os.path.join(ROOT, "pnas-sub", "analysis", "out")

TV = pd.read_csv("data/corpus/tv_structural_1890_2025.csv", usecols=["tv_idx", "title", "year"])
titles = sorted(set(TV.title.astype(str)))
print(f"[tv] {len(TV)} works, {len(titles)} unique titles to resolve")

# ---------------------------------------------------------------- Wikidata SPARQL (mirrors A14)
ENDPOINT = "https://query.wikidata.org/sparql"
UA = "style-evolves-research/1.0 (samsun.knight@rotman.utoronto.ca) academic-metadata"
HEADERS = {"User-Agent": UA, "Accept": "application/sparql-results+json"}
BATCH = 150

def esc(t):
    return t.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").replace("\t", " ").replace("\r", " ")

def run_query(q, tries=3):
    data = urllib.parse.urlencode({"query": q, "format": "json"}).encode()
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(ENDPOINT, data=data, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except Exception as e:
            last = e
            time.sleep([5, 15, 30][min(k, 2)])
    raise last

# Resumable pull with an on-disk cache, so a killed run loses nothing and the whole corpus resolves
# across several short foreground calls (a long single background pull gets killed). Each run works
# for a wall-clock budget, records every queried title (matched or not) so unmatched titles are not
# re-queried forever, and exits early with a RESUME marker if titles remain.
CACHE = os.path.join(OUT, "A04_p31_cache.json")
RUN_BUDGET_S = float(os.environ.get("A04_BUDGET_S", "80"))
if os.path.exists(CACHE):
    c = json.load(open(CACHE))
    queried = set(c["queried"]); p31 = {k: set(v) for k, v in c["p31"].items()}
else:
    queried, p31 = set(), {}

def save_cache():
    json.dump({"queried": sorted(queried), "p31": {k: sorted(v) for k, v in p31.items()}},
              open(CACHE, "w"))

remaining = [t for t in titles if t not in queried]
print(f"[sparql] {len(queried)} already queried; {len(remaining)} remaining")
n_fail = 0
if remaining:
    start = time.time()
    for bi in range(0, len(remaining), BATCH):
        if time.time() - start > RUN_BUDGET_S:
            break
        chunk = remaining[bi:bi + BATCH]
        values = " ".join(f'"{esc(t)}"@en' for t in chunk)
        q = f"""SELECT ?name ?p31Label WHERE {{
  VALUES ?name {{ {values} }}
  ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> ; schema:name ?name .
  OPTIONAL {{ ?item wdt:P31 ?p31. }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}"""
        try:
            res = run_query(q)
        except Exception as e:
            n_fail += 1
            print(f"[sparql] batch FAILED: {e}", flush=True)
            time.sleep(2.0)
            continue
        for b in res["results"]["bindings"]:
            name = b.get("name", {}).get("value")
            lab = b.get("p31Label", {}).get("value")
            if name is None:
                continue
            s = p31.setdefault(name, set())
            if lab:
                s.add(lab)
        queried |= set(chunk)   # mark queried even if unmatched, so we don't re-query
        save_cache()
        print(f"[sparql] {len(queried)}/{len(titles)} queried; {len(p31)} matched", flush=True)
    save_cache()

still = [t for t in titles if t not in queried]
if still:
    print(f"RESUME: {len(still)} titles still to query -- rerun this script.", flush=True)
    sys.exit(0)
print(f"[sparql] complete: {len(p31)} of {len(titles)} titles matched an enwiki article", flush=True)

# ---------------------------------------------------------------- classification
def classify_p31(labels):
    """Map a work's P31 instance-of labels to a unit type, most specific first. None if no usable P31."""
    if not labels:
        return None
    L = " | ".join(labels).lower()
    if "episode" in L:
        return "episode"
    if "television season" in L or "season of" in L:
        return "season"
    if "miniseries" in L or "limited series" in L:
        return "miniseries"
    if "television film" in L or "television movie" in L or "made-for-television" in L:
        return "tv_film"
    if "anthology" in L:
        return "serial_anthology"
    if "television special" in L or ("special" in L and "television" in L):
        return "special"
    if any(k in L for k in ["television series", "television programme", "television program",
                            "anime television", "animated series", "web series", "soap opera",
                            "sitcom", "drama television", "reality"]):
        return "series"
    return "other"  # matched Wikidata but not a recognized television unit type

def classify_suffix(title):
    """Fallback: the article title's parenthetical marker, used only when Wikidata has no usable P31."""
    t = title.lower()
    if "(tv series)" in t or "(tv programme)" in t or "(tv program)" in t:
        return "series"
    if "miniseries" in t:
        return "miniseries"
    if "(tv film)" in t or "(tv movie)" in t:
        return "tv_film"
    if "(tv special)" in t or "(special)" in t:
        return "special"
    if "season" in t:
        return "season"
    if "episode" in t:
        return "episode"
    return None

rows = []
for r in TV.itertuples():
    labels = sorted(p31.get(str(r.title), set()))
    ut = classify_p31(labels)
    conf = "wikidata"
    if ut is None or ut == "other":
        suf = classify_suffix(str(r.title))
        if suf is not None:
            ut, conf = suf, "title_suffix"
        elif ut is None:
            ut, conf = "unknown", "unknown"
        else:
            conf = "wikidata"  # 'other': matched Wikidata but non-TV-unit type
    rows.append({"tv_idx": r.tv_idx, "title": r.title, "year": r.year,
                 "p31_labels": "; ".join(labels), "unit_type": ut, "confidence": conf})

audit = pd.DataFrame(rows)
audit.to_csv(os.path.join(OUT, "tv_unit_audit.csv"), index=False)

dist = audit.unit_type.value_counts().to_dict()
conf_dist = audit.confidence.value_counts().to_dict()
unknown_share = float((audit.unit_type == "unknown").mean())
summary = {"n_tv": int(len(audit)), "n_titles_matched_wikidata": len(p31),
           "unit_type_counts": dist, "confidence_counts": conf_dist,
           "unknown_share": round(unknown_share, 4), "sparql_batch_failures": n_fail,
           "note": "P31 instance-of via enwiki sitelink (A14 linkage); title-suffix fallback; "
                   "unknown where neither resolves. Metadata labeling, not human rating."}
json.dump(summary, open(os.path.join(OUT, "A04_tv_unit_type.json"), "w"), indent=2)
print("\nunit_type distribution:")
for k, v in sorted(dist.items(), key=lambda x: -x[1]):
    print(f"  {k:16s} {v:6d}  ({v/len(audit)*100:4.1f}%)")
print(f"\nconfidence: {conf_dist}")
print(f"unknown share: {unknown_share:.1%}")
print("wrote analysis/out/tv_unit_audit.csv + A04_tv_unit_type.json")
