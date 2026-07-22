"""GENERATION STEP (not part of reproduce.py). Recomputes the book validation from the
raw reader survey. The raw survey, the review-priming tracking, and the two concept
registries are external inputs NOT included in this release (they carry rater-level
data); place them under inputs/ to re-run. The shipped intermediate this produces,
data/validation/{book_attribute_validation.csv, human_means_book.csv}, is what
reproduce.py reads, so reproduction does not require this script or the raw inputs.

Book attribute validation under the EMNLP convention: drop Good/Bad review-primed readers,
keep No-Review + unmatched (matched at reader level against the reader-survey tracking sheet). Concept-anchored,
title-matched to the book century corpus. Out: data/validation/book_attribute_validation.csv"""
import csv, re, json, collections
import numpy as np, pandas as pd
def norm(s): return re.sub(r"\s+"," ",str(s).strip())
def lead_int(s):
    m=re.match(r"^\s*(\d+)",str(s)); return int(m.group(1)) if m else None
BR="inputs/concept_registry.json"
bk=json.load(open(BR)); BE=[{norm(k):v for k,v in (c.get("option_encoding") or {}).items() if v is not None} for c in bk.values() if c.get("response_type")=="ordinal" and c.get("option_encoding")]
KW=[("not at all|none|never|no\\b",1),("a little|slightly|rarely|some\\b",2),("somewhat|moderately|a few|neither",3),("very|quite|many|often|most\\b",4),("extremely|all\\b|always|entirely|definitely|major",5)]
def venc(L):
    if any(len(l)>30 or"," in l for l in L): return None
    e={}
    for l in L:
        for p,rr in KW:
            if re.search(p,l.lower()): e[l]=rr;break
    return e if len(e)==len(L) and len(set(e.values()))>=3 else None
def benc(L):
    nl=[norm(l) for l in L];best=None;cov=0
    for enc in BE:
        c=sum(1 for l in nl if l in enc)
        if c>cov and c>=max(2,int(0.6*len(nl))):cov=c;best=enc
    return best
def enc_ord(L):
    if sum(lead_int(x) is not None for x in L)/max(1,len(L))>0.8: return {l:lead_int(l) for l in L if lead_int(l) is not None}
    be=benc(L); return {l:be.get(norm(l)) for l in L if norm(l) in be} if be else venc(L)
def clean_pid(p):  # canonical: strip 'Personal ID:' prefix (the dataset paper cleaning script)
    s=str(p).strip(); s=re.sub(r"^Personal\s+ID:\s*","",s,flags=re.I); return s.strip()
trk=pd.read_csv("inputs/review_priming_tracking.csv",dtype=str)
trk["uid"]=trk["User ID"].astype(str).str.strip()
def reduce_review(g):  # a reader primed on ANY book -> primed reader (reader-level drop -> 714)
    vals=[v for v in g["Review"].dropna().tolist()]
    if not vals: return "Unknown"
    if "Good" in vals or "Bad" in vals:
        for v in ("Good","Bad"):
            if v in vals: return v
    return "No Review"
pid_review=trk.groupby("uid").apply(reduce_review).to_dict()
def kept_reader(rid): return rid not in ("","nan","NaN","None") and pid_review.get(rid,"Unknown") not in ("Good","Bad")
bid2t={}
for b,t in zip(trk["Book ID"], trk["Novel"]):
    if pd.notna(t) and str(b).strip() not in ("","nan","NaN","None"):
        try: bid2t[str(int(float(b)))]=t
        except: pass
def nt(s): return re.sub(r'[^a-z0-9]','',str(s).lower())
corp=pd.read_csv("data/corpus/book_structural_1890_2025.csv"); corp["nt"]=corp.title.apply(nt); corp=corp.drop_duplicates("nt").set_index("nt")
reg=json.load(open("inputs/survey_registry.json")); cid2q={c:v['q'] for c,v in reg.items()}
def core(q): return re.sub(r'\b(movie|book|novel|film)s?\b','',re.sub(r'^\s*\d+[a-z]?[\.\)]\s*','',q)).strip().lower()
bp="inputs/book_survey_raw.csv"
r=list(csv.reader(open(bp,encoding='utf-8',errors='replace'))); names=r[0]; data=r[3:]
fin=names.index("Finished"); q2=names.index("Q2"); q7=names.index("Q720"); data=[x for x in data if len(x)>fin and x[fin] in("True","1")]
rows=[]
for cid in [c for c in corp.columns if c not in("title","year","book_idx","nt")]:
    q=cid2q.get(cid)
    if not q: continue
    ck=core(q)[:38]; ci=next((i for i,sq in enumerate(r[1]) if ck and ck in core(sq)),None)
    if ci is None: continue
    tmp=[]; labels=set()
    for x in data:
        if len(x)<=max(ci,q2,q7): continue
        rdr=clean_pid(x[q2])
        if not kept_reader(rdr): continue          # reader-level primed drop (canonical 714)
        try: bkid=str(int(float(x[q7].strip())))
        except: continue
        v=x[ci].strip()
        if v: labels.add(v); tmp.append((bkid,v))
    e=enc_ord(labels)
    if not e: continue
    bm=collections.defaultdict(list)
    for bkid,v in tmp:
        if v in e and e[v] is not None: bm[bkid].append(e[v])
    xs,ys,nread=[],[],0
    for bkid,vs in bm.items():
        t=bid2t.get(bkid); k=nt(t) if t else None
        if k and k in corp.index and pd.notna(corp.loc[k,cid]): xs.append(float(corp.loc[k,cid])); ys.append(np.mean(vs)); nread+=len(vs)
    if len(xs)<25 or np.std(xs)<1e-9: continue
    rows.append({"attribute":cid,"question":core(q)[:50],"n_books":len(xs),"n_ratings":nread,"r2":round(np.corrcoef(xs,ys)[0,1]**2,3)})
out=pd.DataFrame(rows).sort_values("r2",ascending=False)
out.to_csv("data/validation/book_attribute_validation.csv",index=False)
print(f"saved {len(out)} book attrs (EMNLP convention: Good/Bad dropped, unmatched kept)")
print(f"  r2>0.10: {(out.r2>0.10).sum()} | r2>0.05: {(out.r2>0.05).sum()} | median {out.r2.median():.3f}")
print(out.head(10).to_string(index=False))
