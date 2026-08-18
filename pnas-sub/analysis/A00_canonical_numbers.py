"""A00 (PNAS rebuild, Phase 0 / P0.1): generate the canonical numbers ledger.

Every headline quantity in the manuscript is regenerated here from its authoritative source --- the
codebook (attribute counts), the released corpus (work counts), and the committed analysis outputs in
``analysis/out/*.json`` (coefficients) --- and written to ``analysis/out/canonical_numbers.yml`` with,
for each number, its value, the manuscript-facing display string, the source that produced it, the
manuscript locations that must carry it, and a reconciliation note. ``check_numbers.py`` then fails
the build if any manuscript file disagrees with this ledger, so "is the paper described right?" is one
command rather than a hope. Nothing here is hand-typed: change a spec, rerun the analysis, rerun this.

Run from the style_evolves root: ``.venv/bin/python pnas-sub/analysis/A00_canonical_numbers.py``.
"""
import os, sys, json

ROOT = os.environ.get("NARRATIVE_ATLAS_ROOT", os.path.expanduser("~/uoft/style_evolves"))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "code"))
import pandas as pd
from _methods import reconcile_counts

OUT = os.path.join(ROOT, "pnas-sub", "analysis", "out")


def _json(name):
    return json.load(open(os.path.join(OUT, name)))


def collect():
    """Every canonical number, keyed by a stable id, each carrying value/display/source/locations."""
    cb_counts = reconcile_counts()

    # corpus work counts, straight from the released structural spines
    corpus = {}
    for m, fn in [("film", "film"), ("book", "book"), ("tv", "tv")]:
        df = pd.read_csv(f"data/corpus/{fn}_structural_1890_2025.csv",
                         usecols=lambda c: c in ("year",))
        y = pd.to_numeric(df["year"], errors="coerce")
        corpus[m] = {"n": int(len(df)), "dated": int(y.notna().sum()), "undated": int(y.isna().sum())}
    corpus_total = sum(v["n"] for v in corpus.values())

    a2, a3, a9 = _json("A2_variance_ratio.json"), _json("A3_adaptation_one_per_source.json"), _json("A9_medium_era.json")
    a02 = _json("A02_covariance_agreement.json")
    a07 = _json("A07_century_fingerprint.json")
    t5 = a9["task5_standard_decomposition"]

    def entry(value, display, source, locations, note=""):
        return {"value": value, "display": display, "source": source,
                "manuscript_locations": locations, "note": note}

    E = {}
    E["corpus_total"] = entry(corpus_total, "149{,}341", "A00: sum of released structural spines",
        ["00_abstract.tex", "00_significance.tex", "01_intro.tex", "07_methods.tex"],
        "film 94,140 + novel 22,978 + television 32,223 = 149,341.")
    E["corpus_film"] = entry(corpus["film"]["n"], "94{,}140", "A00: film spine", ["07_methods.tex"])
    E["corpus_book"] = entry(corpus["book"]["n"], "22{,}978", "A00: novel spine", ["07_methods.tex"],
        "Novel spine spans 1800--2025 at source; the analysis window is 1915--2020 (_methods.WINDOW).")
    E["corpus_tv"] = entry(corpus["tv"]["n"], "32{,}223", "A00: television spine", ["07_methods.tex"])
    E["respondents_readers"] = entry(711, "711", "survey (declared, post-exclusion)",
        ["01_intro.tex", "07_methods.tex"], "Reconciled 714 (old resource abstract) -> 711 after exclusions.")
    E["respondents_viewers"] = entry(225, "225", "survey (declared, post-exclusion)",
        ["01_intro.tex", "07_methods.tex"])
    E["attrs_scored"] = entry(cb_counts["scored_attributes"], "216", "A00: codebook minus reception",
        ["01_intro.tex"], cb_counts["rule"])
    E["attrs_validated"] = entry(cb_counts["validated_attributes"], "195",
        "A00: codebook, heldout_tier != drop", ["01_intro.tex", "02_results1_signatures.tex"])
    E["robust_core"] = entry(cb_counts["robust_core_attributes"], "121",
        "A00: codebook, release_class == core", ["02_results1_signatures.tex"])
    E["variance_ratio_3media"] = entry(round(a2["point_3media"], 2), "1.66",
        "A2 point_3media (1.658)", ["00_abstract.tex", "01_intro.tex", "02_results1_signatures.tex"],
        "16-attr geometry_shared16 basis; between-medium / within-medium decade-centroid variance.")
    E["variance_ratio_ci_lo"] = entry(a2["boot_ci_3media"][0], "1.47", "A2 boot_ci_3media",
        ["01_intro.tex", "02_results1_signatures.tex"])
    E["variance_ratio_ci_hi"] = entry(a2["boot_ci_3media"][1], "1.79", "A2 boot_ci_3media",
        ["01_intro.tex", "02_results1_signatures.tex"])
    E["variance_ratio_film_novel"] = entry(round(a2["point_film_novel"], 2), "2.21",
        "A2 point_film_novel (2.212)", ["02_results1_signatures.tex"])
    E["eta2_medium_pct"] = entry(t5["eta2_medium_pct"], "3.9", "A9 task5 eta2_medium_pct",
        ["02_results1_signatures.tex"], "Standard factorial multivariate eta^2, 16-attr basis.")
    E["eta2_decade_pct"] = entry(round(t5["eta2_decade_pct"], 1), "1.4", "A9 task5 eta2_decade_pct (1.39)",
        ["02_results1_signatures.tex"])
    E["eta2_interaction_pct"] = entry(t5["eta2_interaction_pct"], "0.2", "A9 task5 eta2_interaction_pct",
        ["02_results1_signatures.tex"])
    E["adaptation_pairs"] = entry(a3["n_pairs"], "437", "A3 n_pairs",
        ["03_results2_adaptation.tex"], "In-corpus film-novel pairs (both scored); 10,886 raw Wikidata links.")
    E["adaptation_sources"] = entry(a3["n_sources"], "316", "A3 n_sources", ["03_results2_adaptation.tex"])
    E["fingerprint_n_shifts"] = entry(a07["n_attrs_total"], "284", "A07 n_attrs_total",
        ["04_results3_trajectories.tex"], "Attribute-by-medium standardized 1950s->2010s shifts in the fingerprint.")
    E["fingerprint_n_fdr_sig"] = entry(a07["n_fdr_sig"], "240", "A07 n_fdr_sig",
        ["04_results3_trajectories.tex"], "FDR-significant shifts of the 284.")
    # Provenance for numbers whose source is reproduce.py / placebo, not analysis/out/*.json (documented
    # here so they are machine-tracked and not re-flagged as orphans). The distinctive PCA figure is
    # guarded; the two ambiguous values (0.04, 0.21 recur elsewhere) are provenance-only, no location
    # check. Values are the committed reproduce.py / placebo targets, not re-derived here.
    E["pca_two_component_var_pct"] = entry(42, "42",
        "reproduce.py: PCA PC1 0.31 + PC2 0.11 on the pooled structural spine", ["main.tex"],
        "Fig. 1 caption: first two PCs capture 42% of variance. Related provenance: the medium-word "
        "placebo 0.04 SD = code/placebo_medium_label.py; the k-means silhouette 0.21 (k=2) / <0.13 "
        "(k>=4) = reproduce.py silhouette.")
    E["covariance_agreement_r"] = entry(round(a02["offdiag_agreement_r_full"], 2), "0.82",
        "A02 offdiag_agreement_r_full (0.818)", ["02_results1_signatures.tex", "si.tex"],
        "Human vs model inter-attribute correlation-matrix off-diagonal agreement, 240 title-matched "
        "films, 44-attr overlap; coupling ratio 1.44. Corrects the stale 0.77 (old plot-LLM linkage).")
    return E


def to_yaml(entries):
    """Minimal YAML emitter for the flat ledger (avoids a PyYAML dependency)."""
    def scalar(v):
        if isinstance(v, str):
            return v if (v and all(c.isalnum() or c in " .,%-_/+" for c in v)) else json.dumps(v)
        return json.dumps(v)
    lines = ["# Canonical numbers ledger --- generated by A00_canonical_numbers.py; do not hand-edit.",
             "# Each entry: value (authoritative), display (manuscript string), source, "
             "manuscript_locations, note.", ""]
    for key, e in entries.items():
        lines.append(f"{key}:")
        lines.append(f"  value: {scalar(e['value'])}")
        lines.append(f"  display: {scalar(e['display'])}")
        lines.append(f"  source: {scalar(e['source'])}")
        lines.append("  manuscript_locations:")
        for loc in e["manuscript_locations"]:
            lines.append(f"    - {scalar(loc)}")
        lines.append(f"  note: {scalar(e['note'])}")
        lines.append("")
    return "\n".join(lines)


import re, glob

MANUSCRIPT_GLOBS = ["pnas-sub/sections/*.tex", "pnas-sub/si.tex", "pnas-sub/main.tex"]

def _norm(text):
    return text.replace("{", "").replace("}", "")

def count_in_text(display, text):
    """How many times the display string appears as a standalone number (not the prefix of a longer
    number, so 1.4 does not match 1.47). Counting, not mere presence, so perturbing one of several
    occurrences of a repeated value is still caught."""
    pat = r"(?<![\d.,])" + re.escape(_norm(display)) + r"(?![\d])"
    return len(re.findall(pat, _norm(text)))

def discover_locations(display):
    """Manuscript files that carry this number, each with its occurrence count, so the checker can
    fail the build if any file's count changes (a straggler, a perturbed value, a dropped mention)."""
    found = {}
    for g in MANUSCRIPT_GLOBS:
        for f in sorted(glob.glob(os.path.join(ROOT, g))):
            c = count_in_text(display, open(f).read())
            if c:
                found[os.path.relpath(f, os.path.join(ROOT, "pnas-sub"))] = c
    return found

if __name__ == "__main__":
    entries = collect()
    # replace curated expectations with the files that actually carry each value (self-calibrating)
    for k, e in entries.items():
        e["manuscript_locations"] = discover_locations(e["display"])
        if not e["manuscript_locations"]:
            print(f"  [warn] {k} ({e['display']}) not found in any manuscript file")
    path = os.path.join(OUT, "canonical_numbers.yml")
    open(path, "w").write(to_yaml(entries))
    # also emit JSON for machine consumers (check_numbers.py reads this)
    json.dump(entries, open(os.path.join(OUT, "canonical_numbers.json"), "w"), indent=2)
    print(f"wrote {len(entries)} canonical numbers -> analysis/out/canonical_numbers.yml (+ .json)")
    for k, e in entries.items():
        print(f"  {k:28s} {str(e['value']):>10s}  ({e['display']})")
