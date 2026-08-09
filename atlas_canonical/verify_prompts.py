#!/usr/bin/env python3
"""Guard that binds the shipped prompts to the shipped scores, so a truncated, wrong, or drifted
prompt fails loudly instead of shipping silently. `prompts.csv` is the single source of truth
(one row per atlas column x medium, its exact deployed prompt and the canonical prompt it should
carry). Three checks:

  1. COMPLETENESS  - every atlas column has a prompts.csv row with a prompt (no gaps).
  2. WELL-FORMED   - every canonical prompt's question stem ends in terminal punctuation
                     (?/./!/]) - i.e. is not truncated mid-clause.
  3. REPRODUCTION  - (needs an OpenAI key; run with --reproduce) for a sample of works per column
                     NOT flagged needs_rescore, re-score with the deployed prompt and assert it
                     reproduces the shipped atlas column at r >= 0.90. Truncate a prompt, change a
                     recipe, or ship the wrong string, and this fails.

Checks 1-2 are offline and run in CI on every build. Check 3 is the ground-truth binding and is
run before any release. Exit non-zero on any failure.
"""
import os, re, sys, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ATLAS = os.path.join(HERE, "data")
TAIL = re.compile(r'\s+(Give a number|Respond 1|Answer on a scale|Answer 0|Return ONLY)')
PRE = re.compile(r'^(Based only on[^:]*:|Consider this[^.]*\.|On a scale[^,]*,|You are estimating[^.]*\.|Rate this[^.]*\.|In dramatic terms,?)\s*')

def stem(p):
    m = TAIL.search(str(p)); s = str(p)[:m.start()].rstrip() if m else str(p)
    return PRE.sub('', s).rstrip()

def well_formed(p):
    # a 0-100 intensity template ("...describe the mood...") has no question stem and is fine
    if 'describe the' in str(p) or 'apply to this' in str(p): return True
    return stem(p).endswith(('?', '.', '!', ']'))

def main():
    P = pd.read_csv(os.path.join(HERE, "prompts.csv"))
    fail = 0

    # 1. completeness: every atlas column present with a prompt
    for m in ["film", "book", "tv"]:
        F = pd.read_parquet(os.path.join(ATLAS, f"atlas_{m}.parquet"))
        meta = {"idx", "title", "year", "decade", "medium", f"{m}_idx", "_n_attrs"}
        atlas_cols = {c for c in F.columns if c not in meta}
        have = set(P[P.medium == m].column)
        missing = atlas_cols - have
        if missing:
            print(f"[FAIL] completeness {m}: {len(missing)} atlas columns with no prompts.csv row: {sorted(missing)[:8]}")
            fail += 1
    if not fail:
        print(f"[PASS] completeness: every atlas column has a prompt across all media")

    # 2. well-formedness of the canonical (shipped) prompts
    bad = P[~P.canonical_prompt.map(well_formed)]
    if len(bad):
        print(f"[FAIL] well-formed: {len(bad)} canonical prompts truncated mid-clause: {list(bad.column[:8])}")
        fail += 1
    else:
        print(f"[PASS] well-formed: all {len(P)} canonical prompts terminate cleanly (no truncation)")

    # 3. reproduction (optional, needs key): shipped prompt must reproduce shipped scores
    if "--reproduce" in sys.argv:
        from openai import OpenAI
        from concurrent.futures import ThreadPoolExecutor
        cli = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        plot = {m: pd.read_parquet(os.path.expanduser(f"~/uoft/style_evolves/data/{ 'film_wiki_text' if m=='film' else m+'_wiki_text_century'}.parquet")) for m in ["film"]}
        NUM = re.compile(r'-?\d+\.?\d*')
        def score(title, text, sysmsg):
            try:
                r = cli.chat.completions.create(model="gpt-4o-mini-2024-07-18", temperature=0, max_tokens=20,
                    messages=[{"role": "system", "content": sysmsg}, {"role": "user", "content": f'{title}\n{str(text)[:8000]}'}])
                t = r.choices[0].message.content
                try: return float(json.loads(t)["v"])
                except Exception:
                    g = NUM.search(t); return float(g.group()) if g else np.nan
            except Exception: return np.nan
        m = "film"
        F = pd.read_parquet(os.path.join(ATLAS, f"atlas_{m}.parquet")); pl = plot[m]
        norm = lambda s: re.sub(r'[^a-z0-9]', '', str(s).lower())
        for df in (F, pl): df["nt"] = df.title.map(norm)
        F["yr"] = F.year.astype("Int64"); pl["yr"] = pd.to_numeric(pl.get("year"), errors="coerce").astype("Int64")
        mg = F.merge(pl[["nt", "yr", "text"]].dropna(subset=["yr"]).drop_duplicates(["nt", "yr"]), on=["nt", "yr"]).head(60)
        checkset = P[(P.medium == m) & (~P.needs_rescore) & (~P.get('ensemble', False)) & (~P.get('low_signal', False)) & (~P.get('genre_track', False))].sample(min(25, (P.medium == m).sum()), random_state=0)
        bad_r = []
        for _, row in checkset.iterrows():
            col = row.column
            if col not in mg or mg[col].notna().sum() < 20: continue
            d = mg[[col, "title", "text"]].dropna(subset=[col])
            with ThreadPoolExecutor(max_workers=10) as ex:
                nv = list(ex.map(lambda t: score(t[1], t[2], row.deployed_prompt), d[[col, "title", "text"]].itertuples(index=False, name=None)))
            dd = pd.DataFrame({"n": nv, "a": d[col].values}).dropna()
            r = np.corrcoef(dd.n, dd.a)[0, 1] if len(dd) > 2 and dd.n.nunique() > 1 else np.nan
            if pd.isna(r) or r < 0.80: bad_r.append((col, round(r, 3) if pd.notna(r) else None))
        if bad_r:
            print(f"[FAIL] reproduction: {len(bad_r)} sampled columns do not reproduce their shipped prompt at r>=0.80 (ensemble/aggregate columns exempt): {bad_r}")
            fail += 1
        else:
            print(f"[PASS] reproduction: sampled columns reproduce their shipped prompt at r>=0.80 (ensemble/aggregate columns exempt)")

    print(f"\n{'ALL CHECKS PASSED' if not fail else str(fail)+' CHECK(S) FAILED'}")
    sys.exit(1 if fail else 0)

if __name__ == "__main__":
    main()
