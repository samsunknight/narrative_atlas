"""Composition-constant crystallization robustness (SI).
Shows the rise in mean |pairwise correlation| among spine attributes survives
holding the film corpus's genre mix fixed three independent ways. Run from repo root."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False})

F=pd.read_csv("data/corpus/film_structural_1890_2025.csv").rename(columns={"film_idx":"id"})
ATTRS=[c for c in F.columns if c not in("id","title","year")]
pf=pd.read_parquet("data/atlas/century_frame_film.parquet")
GEN=[c for c in pf.columns if c.startswith("genre_")]
g=pf[["idx"]+GEN].copy()
for gc in GEN: g[gc]=(g[gc]>=70).astype(int)
D=F.merge(g,left_on="id",right_on="idx",how="left"); D["dec"]=(D.year//10)*10
DECS=list(range(1910,2011,10))
pw={c:D[c].fillna(0).sum() for c in GEN}; tot=sum(pw.values()); pw={k:v/tot for k,v in pw.items()}
MAJ=[c for c in GEN if pw[c]>0.03]

def mabs(cm): return np.nanmean(np.abs(cm)[np.triu_indices_from(cm,1)])
def crys(sub):
    s=sub[ATTRS].dropna(axis=1,how="all"); return mabs(s.corr().values) if len(s)>=40 else np.nan
base={d:crys(D[D.dec==d]) for d in DECS}
def resid(d):
    sub=D[D.dec==d]
    if len(sub)<40: return np.nan
    X=np.column_stack([np.ones(len(sub)),sub[MAJ].fillna(0).values]); R={}
    for a in ATTRS:
        y=sub[a].values.astype(float); m=~np.isnan(y)
        if m.sum()<40: continue
        b=np.linalg.lstsq(X[m],y[m],rcond=None)[0]
        r=np.full(len(sub),np.nan); r[m]=y[m]-X[m]@b; R[a]=r
    return mabs(pd.DataFrame(R).corr().values)
B={d:resid(d) for d in DECS}
C={d:crys(D[(D.dec==d)&(D.genre_Drama==1)]) for d in DECS}
def prim(row):
    tags=[c for c in MAJ if row.get(c)==1]; return min(tags,key=lambda c:pw[c]) if tags else None
D["pg"]=D.apply(prim,axis=1)
tgt={c:pw[c] for c in MAJ}; s=sum(tgt.values()); tgt={k:v/s for k,v in tgt.items()}
def wcorr(sub,w):
    Xv=sub[ATTRS].values.astype(float); n=Xv.shape[1]; cm=np.full((n,n),np.nan)
    for i in range(n):
        for j in range(i+1,n):
            a=Xv[:,i]; b=Xv[:,j]; m=~(np.isnan(a)|np.isnan(b))
            if m.sum()<40: continue
            wi=w[m]; am=a[m]-np.average(a[m],weights=wi); bm=b[m]-np.average(b[m],weights=wi)
            cov=np.average(am*bm,weights=wi); va=np.average(am*am,weights=wi); vb=np.average(bm*bm,weights=wi)
            cm[i,j]=cov/np.sqrt(va*vb) if va>0 and vb>0 else np.nan
    return mabs(cm)
def rew(d):
    sub=D[(D.dec==d)&(D.pg.notna())]
    if len(sub)<40: return np.nan
    obs=sub.pg.value_counts(normalize=True).to_dict()
    w=sub.pg.map(lambda p:tgt.get(p,0)/obs.get(p,1)).values.astype(float)
    return wcorr(sub,w)
Dr={d:rew(d) for d in DECS}

fig,ax=plt.subplots(figsize=(7,5))
for lab,dic,c,ls,mk in [("baseline",base,"#1a1a1a","-","o"),
                        ("reweight to fixed genre mix",Dr,"#c0392b","--","s"),
                        ("genre-residualized",B,"#2471a3","--","^"),
                        ("within Drama only",C,"#27ae60",":","D")]:
    xs=[d for d in DECS if not np.isnan(dic[d])]; ys=[dic[d] for d in xs]
    ax.plot(xs,ys,color=c,ls=ls,marker=mk,ms=5,lw=2,label=lab)
ax.set_xlabel("decade"); ax.set_ylabel("mean |pairwise correlation| among spine attributes")
ax.set_title("Crystallization is not a composition artifact",loc="left",fontweight="bold",fontsize=12)
ax.legend(frameon=False,fontsize=9,loc="lower right"); ax.set_ylim(0.18,0.40)
plt.tight_layout(); plt.savefig("results/figures/FIG_crys_composition.png",dpi=150,bbox_inches="tight")
print("FIG_crys_composition built")
