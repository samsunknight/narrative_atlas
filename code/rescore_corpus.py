# ---------------------------------------------------------------------------
# METHOD REFERENCE (not part of replication). Documents how the atlas attribute
# scores were generated with gpt-4o-mini (temperature 0). The scores themselves
# are the released data (data/atlas/, data/corpus/); re-running is optional and
# needs an API key (OPENAI_API_KEY in the environment) plus the plot-text inputs,
# which are NOT redistributed (bulk Wikipedia text). Every deployed per-attribute
# prompt is shipped in data/validation/rescore_manifest.csv (the `prompt` column).
# ---------------------------------------------------------------------------

"""Full-corpus re-score from the manifest. Deployed = best = validated.
Alignment: parquet filtered EXACTLY as score_added.py/rescore_oneat.py (text>300, reset_index) so
position i == corpus {medium}_idx (verified 60/60 title match). Resumable per (medium,idx,attr) cache.
Live-key round-robin. PILOT=N env scores only first N works per medium for verification.
Usage: PILOT=20 WORKERS=8 python3 rescore_corpus.py film book tv     # pilot
       WORKERS=48 python3 rescore_corpus.py film book tv             # full
Out: /tmp/rescore_{medium}/*.json  + data/atlas/{medium}_atlas.csv"""
import pandas as pd, numpy as np, json, os, re, time, itertools, sys
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# API key is read from the environment; no key ships in this package.
CL=[OpenAI()]   # OpenAI() reads OPENAI_API_KEY from the environment
WORKERS=int(os.environ.get('WORKERS','16')); PILOT=int(os.environ.get('PILOT','0'))
_rr=itertools.count()
def call(sysmsg, title, text):
    for a in range(6):
        c=CL[next(_rr)%len(CL)]
        try:
            r=c.chat.completions.create(model='gpt-4o-mini-2024-07-18',temperature=0,response_format={'type':'json_object'},
                max_tokens=20,messages=[{'role':'system','content':sysmsg},{'role':'user','content':f'{title}\n{str(text)[:8000]}'}])
            return json.loads(r.choices[0].message.content).get('v')
        except Exception: time.sleep(1.5*(a+1))
    return None

MW={'film':'movie','book':'book','tv':'TV show'}
# Plot-summary text inputs (NOT redistributed). Point these at your own {medium}
# plot-text tables, each with a title column and a text/plot/summary column.
PARQ={'film':'<film_plot_text.parquet>','book':'<book_plot_text.parquet>','tv':'<tv_plot_text.parquet>'}
MAN=pd.read_csv('data/validation/rescore_manifest.csv')
MAN=MAN[MAN.deploy==True].reset_index(drop=True)

def attrs_for(medium):
    out=[]
    for _,r in MAN.iterrows():
        media=['film','book','tv'] if r['media']=='all' else str(r['media']).split(',')
        if medium in media: out.append(r)
    return out

for medium in (sys.argv[1:] or ['film','book','tv']):
    A=attrs_for(medium); CACHE=f'/tmp/rescore_{medium}'; os.makedirs(CACHE,exist_ok=True)
    d=pd.read_parquet(os.path.expanduser(PARQ[medium])).reset_index(drop=True)
    tc=[c for c in d.columns if c in ('text','plot','summary')][0]
    tt=[c for c in d.columns if c in ('title','name')]
    d['__t']=d[tt[0]] if tt else d.index.astype(str)
    d=d[d[tc].notna() & (d[tc].astype(str).str.len()>300)].reset_index(drop=True)  # == corpus idx convention
    # conscience fix: restrict to works actually in the corpus (idx is a subset of filtered positions) -> saves ~$379
    corp=pd.read_csv(f'data/corpus/{medium}_structural_1890_2025.csv', usecols=[f'{medium}_idx'])
    d=d[d.index.isin(set(corp[f'{medium}_idx']))]
    if PILOT: d=d.head(PILOT)
    mw=MW[medium]
    jobs=[]
    for i in d.index:
        ttl=str(d.loc[i,'__t']); txt=d.loc[i,tc]; umsg=f'{mw}: "{ttl}"'
        for r in A:
            aid=r['attr_id']; f=f'{CACHE}/{i}__{aid}.json'
            if not os.path.exists(f):
                jobs.append((f, r['prompt'].replace('{m}',mw), umsg, txt))
    print(f'[{medium}] works={len(d)} attrs={len(A)} calls_to_make={len(jobs)}',flush=True); t0=time.time()
    def run(a):
        f,sm,title,text=a; v=call(sm,title,text)
        if v is not None: json.dump({'v':v},open(f,'w'))
    with ThreadPoolExecutor(max_workers=WORKERS) as ex: list(ex.map(run,jobs))
    print(f'[{medium}] scored in {(time.time()-t0)/60:.1f} min; assembling...',flush=True)
    rows=[]
    for i in d.index:
        row={f'{medium}_idx':i,'title':d.loc[i,'__t']}
        for r in A:
            f=f"{CACHE}/{i}__{r['attr_id']}.json"
            if os.path.exists(f):
                try: row[r['attr_id']]=json.load(open(f)).get('v')
                except: pass
        rows.append(row)
    os.makedirs('data/atlas',exist_ok=True)
    out=f'data/atlas/{medium}_atlas{"_pilot" if PILOT else ""}.csv'
    pd.DataFrame(rows).to_csv(out,index=False); print(f'[{medium}] -> {out} ({len(rows)} works x {len(A)} attrs)',flush=True)
