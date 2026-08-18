"""check_numbers.py (PNAS rebuild, Phase 0 / P0.1): the build-time guard.

Fails (exit 1) if the manuscript disagrees with the canonical numbers ledger --- either a tracked
number has drifted out of a file the ledger recorded it in (a straggler / perturbed value), or a
ledger value no longer appears in the manuscript at all. Run after A00_canonical_numbers.py:

    .venv/bin/python pnas-sub/analysis/check_numbers.py            # checks the live manuscript
    .venv/bin/python pnas-sub/analysis/check_numbers.py --texdir DIR   # checks a copy (for testing)

The ledger is authoritative; a disagreement is fixed by correcting the manuscript toward the ledger
(or, if the analysis genuinely changed, by rerunning A00 so the ledger tracks the new value).
"""
import os, sys, json, re, glob

ROOT = os.environ.get("NARRATIVE_ATLAS_ROOT", os.path.expanduser("~/uoft/style_evolves"))
OUT = os.path.join(ROOT, "pnas-sub", "analysis", "out")


def _norm(text):
    return text.replace("{", "").replace("}", "")


def count_in_text(display, text):
    pat = r"(?<![\d.,])" + re.escape(_norm(display)) + r"(?![\d])"
    return len(re.findall(pat, _norm(text)))


def main():
    texdir = None
    if "--texdir" in sys.argv:
        texdir = sys.argv[sys.argv.index("--texdir") + 1]
    base = texdir if texdir else os.path.join(ROOT, "pnas-sub", "sections")

    ledger = json.load(open(os.path.join(OUT, "canonical_numbers.json")))
    failures, checked = [], 0
    for key, e in ledger.items():
        locs = e.get("manuscript_locations") or {}
        for loc, expected in locs.items():
            fname = os.path.basename(loc)
            # resolve against the base dir (sections) or, for si.tex, the pnas-sub root
            cand = os.path.join(base, fname)
            if not os.path.exists(cand):
                cand = os.path.join(ROOT, "pnas-sub", loc)
            if not os.path.exists(cand):
                failures.append(f"{key}: location {loc} not found on disk")
                continue
            checked += 1
            actual = count_in_text(e["display"], open(cand).read())
            if actual != expected:
                failures.append(f"{key}: {e['display']!r} in {loc} appears {actual}x, "
                                f"ledger expects {expected}x (value {e['value']}, source {e['source']})")

    if checked == 0:
        print("check_numbers: no manuscript source present (pnas-sub/*.tex not shipped); the numbers are\n"
              "  recomputed and ledgered above. Place the manuscript here to run the consistency check.")
        sys.exit(0)
    print(f"check_numbers: {checked} (number, location) assertions checked against the ledger.")
    if failures:
        print(f"\nFAIL: {len(failures)} manuscript/ledger disagreement(s):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS: manuscript agrees with the canonical numbers ledger.")


if __name__ == "__main__":
    main()
