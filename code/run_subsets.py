# Robustness of headline results to the attribute subset (appendix exhibit).
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
F=pd.read_csv("data/corpus/film_structural_1890_2025.csv").rename(columns={"film_idx":"id"})
ATTRS=[c for c in F.columns if c not in("id","title","year")]
def col(key): 
    m=[c for c in ATTRS if key in c]; return m[0] if m else None
# attribute groups by KEY
RATCHET={"sci-fi":"science_fictional","#settings":"how_many_major_settings","#protag":"how_many_protagonists",
         "#sidechar":"named_side","world-building":"world_building","immersive":"immersive"}
FASHION={"surprise":"unsurprising","proactive":"proactive","competence":"competent","plot-vs-char":"character_driven","moved":"moved"}
RATCHET={k:col(v) for k,v in RATCHET.items() if col(v)}; FASHION={k:col(v) for k,v in FASHION.items() if col(v)}
# subsets
R_full=list(RATCHET.values())
R_val=[RATCHET[k] for k in ["sci-fi","#protag","world-building"] if k in RATCHET]      # cross-medium-validated
R_read=[RATCHET[k] for k in ["sci-fi","#settings","#protag","#sidechar","world-building"] if k in RATCHET]  # readable-structural (drop immersive)
Ff_full=list(FASHION.values())
Ff_struct=[FASHION[k] for k in ["surprise","proactive","plot-vs-char"] if k in FASHION] # structural-only (drop evaluative competence/moved)
def zmean(df,cols): 
    Z=pd.DataFrame({c:(df[c].astype(float)-df[c].astype(float).mean())/df[c].astype(float).std() for c in cols}); return Z.mean(axis=1)
def trend(df,idx):  # per-century slope of decade means
    d=df.dropna(subset=[idx]).copy(); d["dec"]=(d.year//10)*10
    g=d.groupby("dec")[idx].mean()
    return round(np.polyfit(g.index.values.astype(float),g.values,1)[0]*100,3)
S=pd.read_csv("results/tables/imdb_success.csv")[["id","rating","votes"]]
M=S.merge(F,on="id",how="left"); M["lvotes"]=np.log1p(M.votes); M["dec"]=(M.year//10)*10
def partial(idxcol,out):
    d=M.dropna(subset=[idxcol,out,"dec"]); rx=d[idxcol]-d.groupby("dec")[idxcol].transform("mean"); ry=d[out]-d.groupby("dec")[out].transform("mean")
    return round(np.corrcoef(rx,ry)[0,1],3),len(d)
rows=[]
for label,cols,kind in [("ratchet: full (6)",R_full,"r"),("ratchet: cross-medium-validated (3)",R_val,"r"),("ratchet: readable-structural (5)",R_read,"r"),
                         ("fashion: full (5)",Ff_full,"f"),("fashion: structural-only (3)",Ff_struct,"f")]:
    Fi=F.copy(); Fi["idx"]=zmean(Fi,cols); tr=trend(Fi,"idx")
    M["idx"]=zmean(M,cols); rv,n=partial("idx","lvotes"); rr,_=partial("idx","rating")
    rows.append({"index":label,"n_attr":len(cols),"trend/century":tr,"reach (votes)":rv,"rating":rr,"n":n})
out=pd.DataFrame(rows); out.to_csv("results/tables/S_subsets.csv",index=False)
print(out.to_string(index=False))
