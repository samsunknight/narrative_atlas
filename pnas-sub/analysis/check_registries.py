"""check_registries.py (PNAS rebuild): interim drift tripwire for the attribute basis.

Several analysis scripts still carry a private ``VALCM_KEYS`` copy of the sixteen-attribute basis
(a deferral: de-duplicating them must ride with the S2a ROOT unification + mirror deletion, or it
entrenches the stale mirror module). This guard asserts that every such copy is byte-identical to the
canonical ``VAL16_KEYS`` in ``code/_methods.py``, so editing one copy without the others fails the
build during the deferral window. S2a retires this guard when it collapses the duplicates to a single
``from _methods import VAL16_KEYS`` import.

Run from the style_evolves root: ``.venv/bin/python pnas-sub/analysis/check_registries.py``.
"""
import os, sys, ast, glob

ROOT = os.environ.get("NARRATIVE_ATLAS_ROOT", os.path.expanduser("~/uoft/style_evolves"))
sys.path.insert(0, os.path.join(ROOT, "code"))
from _methods import VAL16_KEYS

CANON = list(VAL16_KEYS)
TRACKED = ("VALCM_KEYS", "VAL16_KEYS")


def literal_assignments(path):
    """Every module-level list-literal assigned to a tracked name in a file, without executing it."""
    found = {}
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in TRACKED:
                    try:
                        found[t.id] = ast.literal_eval(node.value)
                    except Exception:
                        pass  # a non-literal (e.g. an import alias) is not a hardcoded copy
    return found


def main():
    failures = []
    for f in sorted(glob.glob(os.path.join(ROOT, "pnas-sub", "analysis", "*.py"))):
        for name, val in literal_assignments(f).items():
            if list(val) != CANON:
                failures.append(f"{os.path.basename(f)}: {name} literal differs from canonical VAL16_KEYS")
    n = len(glob.glob(os.path.join(ROOT, "pnas-sub", "analysis", "*.py")))
    print(f"check_registries: scanned {n} analysis scripts for hardcoded basis copies.")
    if failures:
        print(f"\nFAIL: {len(failures)} attribute-basis copy has drifted from canonical:")
        for x in failures:
            print(f"  - {x}")
        sys.exit(1)
    print("PASS: every hardcoded basis copy is byte-identical to code/_methods.VAL16_KEYS.")


if __name__ == "__main__":
    main()
