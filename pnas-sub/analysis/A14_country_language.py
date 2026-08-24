"""A14 (PNAS Phase 2): does the English-Wikipedia frame's national/linguistic composition
confound the medium-vs-era result? (referee major comments 5 & 13).

We recover country of origin and original language from Wikidata (public SPARQL) for a
stratified random sample of corpus works, then (a) describe the compositional shift over
decades and (b) recompute A2's between/within medium-vs-era variance ratio RESTRICTED to
English-original works, comparing to the same-works full ratio. The question: does the
medium-over-era ordering survive holding language roughly constant?

Wikidata provenance, per enwiki article title (matched on the English Wikipedia sitelink
schema:name):
  - original language: P407 (language of work; used by novels) COALESCED with
    P364 (original language of film or TV show; used by films/TV). Either counts.
  - country of origin: P495.
Labels resolved in English. Batched VALUES queries (~150 titles), polite sleeps, retries.
No LLM. Only public metadata is fetched. Writes out/A14_country_language.json.
Run from ~/uoft/style_evolves/.
"""
import os, sys, json, time, urllib.request, urllib.parse
import numpy as np, pandas as pd

ROOT = os.environ.get("NARRATIVE_ATLAS_ROOT", os.path.expanduser("~/uoft/style_evolves"))
OUT = os.path.join(ROOT, "pnas-sub/analysis/out")
os.makedirs(OUT, exist_ok=True)

# Reproduction is network-free: if the shipped Wikidata snapshot is present, use it and skip the live
# SPARQL query. Delete out/A14_country_language.json to regenerate the country/language pull from Wikidata.
if os.path.exists(os.path.join(OUT, "A14_country_language.json")):
    print("[A14] using shipped A14_country_language.json; skipping live Wikidata query")
    sys.exit(0)
for cd in [os.path.join(os.path.dirname(ROOT), "code"), os.path.join(ROOT, "code")]:
    if os.path.isdir(cd) and cd not in sys.path:
        sys.path.insert(0, cd)
def P(*a): return os.path.join(ROOT, *a)

# ---------------------------------------------------------------- load corpora + A2 basis
F = pd.read_csv(P("data/corpus/film_structural_1890_2025.csv")).rename(columns={"film_idx": "id"})
B = pd.read_csv(P("data/corpus/book_structural_1890_2025.csv")).rename(columns={"book_idx": "id"})
T = pd.read_csv(P("data/corpus/tv_structural_1890_2025.csv")).rename(columns={"tv_idx": "id"})
from _methods import VAL16_KEYS as VALCM_KEYS
VALCM = [c for c in B.columns if any(k in c.lower() for k in VALCM_KEYS) and c in T.columns and c in F.columns]
assert len(VALCM) == 16, f"expected 16-attr basis, got {len(VALCM)}"

# fixed pooled z-transform from the FULL frames (identical to A2), reused for every ratio below
MU = pd.concat([B[VALCM], T[VALCM], F[VALCM]]).mean()
SD = pd.concat([B[VALCM], T[VALCM], F[VALCM]]).std().replace(0, 1)

DECS = list(range(1910, 2021, 10))   # A2's in-window decades (win 1915-2020)
MIN = 30                              # A2's cell threshold

# ---------------------------------------------------------------- stratified sample (seed 0)
SEED = 0
TARGET_PER_MEDIUM = 1500
FRAMES = {"fm": F, "bk": B, "tv": T}

def stratified_sample(df, target, seed):
    d = df[["id", "title", "year"] + VALCM].copy()
    d = d[d["year"].notna()]
    d["dec"] = (d["year"].astype(int) // 10) * 10
    decs = sorted(d["dec"].unique())
    per = int(np.ceil(target / max(len(decs), 1)))
    parts = []
    for dc in decs:
        cell = d[d["dec"] == dc]
        parts.append(cell.sample(min(len(cell), per), random_state=seed))
    return pd.concat(parts, ignore_index=True)

samples = {m: stratified_sample(df, TARGET_PER_MEDIUM, SEED) for m, df in FRAMES.items()}
for m, s in samples.items():
    print(f"[sample] {m}: n={len(s)}  decades={s['dec'].min()}-{s['dec'].max()}")

# unique titles to resolve
all_titles = sorted(set(t for s in samples.values() for t in s["title"].astype(str)))
print(f"[sample] unique titles to resolve: {len(all_titles)}")

# ---------------------------------------------------------------- Wikidata SPARQL resolve
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

# title -> {"langs": set, "countries": set}; only titles with a matched enwiki article appear
resolved = {}
n_batches = int(np.ceil(len(all_titles) / BATCH))
n_fail = 0
for bi in range(n_batches):
    chunk = all_titles[bi * BATCH:(bi + 1) * BATCH]
    values = " ".join(f'"{esc(t)}"@en' for t in chunk)
    q = f"""SELECT ?name ?p407Label ?p364Label ?countryLabel WHERE {{
  VALUES ?name {{ {values} }}
  ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> ; schema:name ?name .
  OPTIONAL {{ ?item wdt:P407 ?p407. }}
  OPTIONAL {{ ?item wdt:P364 ?p364. }}
  OPTIONAL {{ ?item wdt:P495 ?country. }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}"""
    try:
        res = run_query(q)
    except Exception as e:
        n_fail += 1
        print(f"[sparql] batch {bi+1}/{n_batches} FAILED after retries: {e}")
        time.sleep(2.0)
        continue
    for b in res["results"]["bindings"]:
        name = b.get("name", {}).get("value")
        if name is None:
            continue
        rec = resolved.setdefault(name, {"langs": set(), "countries": set()})
        for k in ("p407Label", "p364Label"):
            v = b.get(k, {}).get("value")
            if v:
                rec["langs"].add(v)
        c = b.get("countryLabel", {}).get("value")
        if c:
            rec["countries"].add(c)
    print(f"[sparql] batch {bi+1}/{n_batches}: matched-so-far={len(resolved)}")
    time.sleep(1.5)

def is_english(langs):
    return any("english" in l.lower() for l in langs)   # English / American English / British English

def is_us_uk(countries):
    return any(("united states" in c.lower()) or ("united kingdom" in c.lower()) for c in countries)

# ---------------------------------------------------------------- attach resolution to samples
for m, s in samples.items():
    s["matched"] = s["title"].astype(str).isin(resolved)
    s["langs"] = s["title"].astype(str).map(lambda t: sorted(resolved.get(t, {}).get("langs", set())))
    s["countries"] = s["title"].astype(str).map(lambda t: sorted(resolved.get(t, {}).get("countries", set())))
    s["lang_known"] = s["langs"].map(len) > 0
    s["country_known"] = s["countries"].map(len) > 0
    s["english"] = s["langs"].map(is_english)
    s["us_uk"] = s["countries"].map(is_us_uk)

coverage = {}
for m, s in samples.items():
    coverage[m] = {
        "n_sampled": int(len(s)),
        "n_matched": int(s["matched"].sum()),
        "match_rate": round(float(s["matched"].mean()), 4),
        "n_lang_known": int(s["lang_known"].sum()),
        "lang_known_rate_of_matched": round(float(s.loc[s["matched"], "lang_known"].mean()) if s["matched"].any() else 0.0, 4),
        "n_country_known": int(s["country_known"].sum()),
        "country_known_rate_of_matched": round(float(s.loc[s["matched"], "country_known"].mean()) if s["matched"].any() else 0.0, 4),
    }
    print(f"[coverage] {m}: matched {coverage[m]['n_matched']}/{coverage[m]['n_sampled']} "
          f"({coverage[m]['match_rate']:.1%}); lang-known {coverage[m]['n_lang_known']}, "
          f"country-known {coverage[m]['n_country_known']}")

# ---------------------------------------------------------------- (a) compositional shift by decade
composition = {}
for m, s in samples.items():
    sm = s[s["matched"]]
    comp = {}
    for dc in sorted(sm["dec"].unique()):
        cell = sm[sm["dec"] == dc]
        lk = cell[cell["lang_known"]]
        ck = cell[cell["country_known"]]
        comp[int(dc)] = {
            "n_matched": int(len(cell)),
            "n_lang_known": int(len(lk)),
            "share_english": round(float(lk["english"].mean()), 4) if len(lk) else None,
            "n_country_known": int(len(ck)),
            "share_us_uk": round(float(ck["us_uk"].mean()), 4) if len(ck) else None,
        }
    composition[m] = comp

# early-vs-late trend summary (decade-level Pearson r of share vs decade, in-window decades)
def trend(comp, key):
    xs, ys = [], []
    for dc, v in comp.items():
        if 1910 <= dc <= 2020 and v[key] is not None:
            xs.append(dc); ys.append(v[key])
    if len(xs) < 3:
        return None
    r = float(np.corrcoef(xs, ys)[0, 1])
    return {"r_share_vs_decade": round(r, 3), "n_decades": len(xs),
            "first_dec": int(min(xs)), "first_val": ys[xs.index(min(xs))],
            "last_dec": int(max(xs)), "last_val": ys[xs.index(max(xs))]}

trends = {m: {"english": trend(composition[m], "share_english"),
              "us_uk": trend(composition[m], "share_us_uk")} for m in samples}

# ---------------------------------------------------------------- (b) variance ratio (mirror A2)
def ratio_from(zmat, m_lab, d_lab, media, minrows=MIN):
    C = {m: {} for m in media}
    for m in media:
        for d in DECS:
            sel = (m_lab == m) & (d_lab == d)
            if sel.sum() >= minrows:
                C[m][d] = np.nanmean(zmat[sel], axis=0)
    C = {m: v for m, v in C.items() if v}
    if len(C) < 2:
        return np.nan, 0
    ncells = sum(len(v) for v in C.values())
    medmean = {m: np.mean(list(v.values()), axis=0) for m, v in C.items()}
    grand = np.mean([c for m in C for c in C[m].values()], axis=0)
    between = np.mean([np.mean((medmean[m] - grand) ** 2) for m in C])
    within = np.mean([np.mean([np.mean((c - medmean[m]) ** 2) for c in C[m].values()]) for m in C])
    return (between / within if within > 0 else np.nan), ncells

def build(subset_mask, minrows):
    rows = []
    for m, s in samples.items():
        d = s[subset_mask(s) & (s["year"] >= 1915) & (s["year"] <= 2020)].copy()
        d["m"] = m
        rows.append(d[["m", "dec"] + VALCM])
    D = pd.concat(rows, ignore_index=True)
    Z = ((D[VALCM] - MU) / SD).values
    return ratio_from(Z, D["m"].values, D["dec"].values, ["bk", "fm", "tv"], minrows), \
           ratio_from(Z, D["m"].values, D["dec"].values, ["bk", "fm"], minrows), len(D)

masks = {
    "matched_all":  lambda s: s["matched"],
    "lang_known":   lambda s: s["matched"] & s["lang_known"],
    "english_only": lambda s: s["matched"] & s["lang_known"] & s["english"],
}
ratios = {}
for name, mk in masks.items():
    for minrows in (MIN, 15):
        (r3, c3), (rfn, cfn), nD = build(mk, minrows)
        ratios[f"{name}__min{minrows}"] = {
            "ratio_3media": (round(float(r3), 3) if np.isfinite(r3) else None), "cells_3media": c3,
            "ratio_film_novel": (round(float(rfn), 3) if np.isfinite(rfn) else None), "cells_film_novel": cfn,
            "n_works_in_window": int(nD),
        }
        print(f"[ratio] {name} min{minrows}: 3media={ratios[f'{name}__min{minrows}']['ratio_3media']} "
              f"(cells {c3}), film-novel={ratios[f'{name}__min{minrows}']['ratio_film_novel']} (cells {cfn}), n={nD}")

# ---------------------------------------------------------------- assemble + notes
notes = []
notes.append("Country/language recovered from public Wikidata SPARQL; matched on English Wikipedia "
             "sitelink (schema:name == corpus title). No LLM used.")
notes.append("Original language = P407 (works/novels) coalesced with P364 (original language of film "
             "or TV show); films/TV expose language via P364, not P407, so both are queried. English = "
             "label containing 'English' (English/American English/British English). US/UK = country of "
             "origin (P495) label containing 'United States' or 'United Kingdom'.")
notes.append(f"SPARQL query batches that failed after retries: {n_fail} of {n_batches}.")
notes.append("Ratio pipeline mirrors A2 exactly: fixed pooled z-transform from the FULL corpus frames, "
             "win 1915-2020, decade=(year//10)*10, medium x decade centroids with >=MIN rows, "
             "between/within = mean over attrs of medium-centroid dispersion / mean within-medium "
             "decade-centroid dispersion. min30 mirrors A2's threshold; min15 added because the sampled "
             "+ English-restricted subset thins cells, so min15 keeps more decade cells for a like-for-like "
             "full-vs-English comparison. A2's full-corpus headline is 1.66 (3 media).")
notes.append("'lang_known' baseline (matched works with a known original language) is the like-for-like "
             "comparison for 'english_only'; 'matched_all' additionally includes works whose language is "
             "not recorded on Wikidata.")

# headline comparison: does medium-over-era ordering survive English restriction?
def get(name, key, minrows=MIN):
    return ratios[f"{name}__min{minrows}"][key]

survives_note = {
    "min30": {
        "lang_known_3media": get("lang_known", "ratio_3media", 30),
        "english_only_3media": get("english_only", "ratio_3media", 30),
        "lang_known_film_novel": get("lang_known", "ratio_film_novel", 30),
        "english_only_film_novel": get("english_only", "ratio_film_novel", 30),
    },
    "min15": {
        "lang_known_3media": get("lang_known", "ratio_3media", 15),
        "english_only_3media": get("english_only", "ratio_3media", 15),
        "lang_known_film_novel": get("lang_known", "ratio_film_novel", 15),
        "english_only_film_novel": get("english_only", "ratio_film_novel", 15),
    },
}

out = {
    "basis_n": len(VALCM),
    "seed": SEED,
    "target_per_medium": TARGET_PER_MEDIUM,
    "n_unique_titles_queried": len(all_titles),
    "sparql_endpoint": ENDPOINT,
    "coverage": coverage,
    "composition_by_decade": composition,
    "compositional_trends": trends,
    "variance_ratios": ratios,
    "english_restriction_comparison": survives_note,
    "a2_full_corpus_headline_3media": 1.66,
    "notes": notes,
}
json.dump(out, open(os.path.join(OUT, "A14_country_language.json"), "w"), indent=2)
print("\nwrote out/A14_country_language.json")

# ---------------------------------------------------------------- console summary
print("\n================= SUMMARY =================")
for m in ("fm", "bk", "tv"):
    c = coverage[m]
    print(f"{m}: matched {c['n_matched']}/{c['n_sampled']} ({c['match_rate']:.1%}); "
          f"lang-known {c['n_lang_known']}, country-known {c['n_country_known']}")
print("\nCompositional trend (English-language share and US/UK share vs decade, in-window):")
for m in ("fm", "bk", "tv"):
    te, tu = trends[m]["english"], trends[m]["us_uk"]
    if te:
        print(f"  {m} English: {te['first_dec']}={te['first_val']:.2f} -> {te['last_dec']}={te['last_val']:.2f} (r={te['r_share_vs_decade']})")
    if tu:
        print(f"  {m} US/UK:   {tu['first_dec']}={tu['first_val']:.2f} -> {tu['last_dec']}={tu['last_val']:.2f} (r={tu['r_share_vs_decade']})")
print("\nMedium-vs-era variance ratio (does ordering survive English-only?):")
print(f"  A2 full-corpus headline (3 media): 1.66")
for minr in ("min30", "min15"):
    s = survives_note[minr]
    print(f"  [{minr}] matched+lang-known 3media={s['lang_known_3media']} vs English-only 3media={s['english_only_3media']}")
    print(f"  [{minr}] matched+lang-known film-novel={s['lang_known_film_novel']} vs English-only film-novel={s['english_only_film_novel']}")
print("===========================================")
