# 95% CIs (film-level nonparametric bootstrap) on the reception dissociation partials of Supplementary Table S4.
# Within-decade partial correlation of each attribute index with IMDb log-votes (reach) and average rating.
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
rng=np.random.RandomState(0)
F=pd.read_csv("data/corpus/film_structural_1890_2025.csv").rename(columns={"film_idx":"id"})
ATTRS=[c for c in F.columns if c not in ("id","title","year")]
def col(key):
    m=[c for c in ATTRS if key in c]; return m[0] if m else None
RAT={k:col(v) for k,v in {"sci-fi":"science_fictional","#settings":"how_many_major_settings","#protag":"how_many_protagonists","#sidechar":"named_side","world-building":"world_building","immersive":"immersive"}.items() if col(v)}
FAS={k:col(v) for k,v in {"surprise":"unsurprising","proactive":"proactive","competence":"competent","plot-vs-char":"character_driven","moved":"moved"}.items() if col(v)}
subs=[("spectacle full",list(RAT.values())),
      ("spectacle cross-medium (3)",[RAT[k] for k in ["sci-fi","#protag","world-building"]]),
      ("spectacle readable (5)",[RAT[k] for k in ["sci-fi","#settings","#protag","#sidechar","world-building"]]),
      ("fashion full",list(FAS.values())),
      ("fashion structural (3)",[FAS[k] for k in ["surprise","proactive","plot-vs-char"]])]
S=pd.read_csv("results/tables/imdb_success.csv")[["id","rating","votes"]]
M=S.merge(F,on="id",how="left").dropna(subset=["rating","votes","year"]).reset_index(drop=True)
M["lvotes"]=np.log1p(M.votes); M["dec"]=(M.year//10)*10
def zmean(df,cols):
    return np.column_stack([(df[c].astype(float)-df[c].astype(float).mean())/df[c].astype(float).std() for c in cols]).mean(1)
def wdpartial(d,idx,out):
    rx=idx-pd.Series(idx).groupby(d["dec"].values).transform("mean").values
    ry=d[out].values-d.groupby("dec")[out].transform("mean").values
    return np.corrcoef(rx,ry)[0,1]
B=600; n=len(M); print(f"n={n}, bootstraps={B}")
for lab,cols in subs:
    idx0=zmean(M,cols); pr,pg=wdpartial(M,idx0,"lvotes"),wdpartial(M,idx0,"rating")
    br,bg=[],[]
    for _ in range(B):
        d=M.iloc[rng.randint(0,n,n)].reset_index(drop=True); idx=zmean(d,cols)
        br.append(wdpartial(d,idx,"lvotes")); bg.append(wdpartial(d,idx,"rating"))
    rlo,rhi=np.percentile(br,[2.5,97.5]); glo,ghi=np.percentile(bg,[2.5,97.5])
    print(f"{lab:28s} reach {pr:+.2f} [{rlo:+.2f},{rhi:+.2f}]  rating {pg:+.2f} [{glo:+.2f},{ghi:+.2f}]")
