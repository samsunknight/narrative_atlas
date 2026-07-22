import pandas as pd, numpy as np, json, warnings, os
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler; from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression; from sklearn.model_selection import cross_val_score, cross_val_predict; from sklearn.metrics import silhouette_score, balanced_accuracy_score
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10.5,"axes.spines.top":False,"axes.spines.right":False})
def tlen(path):
    d=pd.read_parquet(path).reset_index(drop=True); tc=[c for c in d.columns if c in("text","plot","summary")][0]
    d=d[d[tc].notna()&(d[tc].astype(str).str.len()>300)].reset_index(drop=True); return d[tc].astype(str).str.len()
def load(f,idx,path=None):
    d=pd.read_csv(f"data/corpus/{f}_structural_century.csv").rename(columns={idx:"id"}); d["medium"]=f
    d=d[(d.year>=1890)&(d.year<=2025)]
    if path: L=tlen(path); d["loglen"]=np.log(d["id"].map(L).fillna(L.median()))
    return d
F=load("film","film_idx","/tmp/dbx_dl/film_wiki_text.parquet")
B=load("book","book_idx","/Users/samsunknight/uoft/cultural_physics/data/book_wiki_text_century.parquet")
T=load("tv","tv_idx","/tmp/dbx_dl/tv_wiki_text_century.parquet")
ATTRS=[c for c in F.columns if c not in("id","title","year","medium","loglen")]
def sn(c):
    for k,n in [("science_fictional","sci-fi"),("fantastical","fantastical"),("realistic_was_the_world","realistic-world"),("world_building","world-building"),("how_many_protagonists","#protagonists"),("named_side","#side-chars"),("how_many_major_settings","#settings"),("competent","competence"),("proactive","proactiveness"),("likable","likability"),("relatable","relatability"),("real_did_this","feels-real"),("emotionally_invested","emot-investment"),("pace_of_the","pace"),("unresolved","resolution"),("unsurprising","surprise"),("confusing","clarity"),("convincing","plot-convincing"),("evocative","visual-evocative"),("interesting_did_you_find_the_visual","visual-interest"),("engaging_did_you_find_the_dial","dialogue-engaging"),("realistic_did_you_find_the_dial","dialogue-realism"),("showing_and","showing/telling"),("immersive","immersive"),("score","score"),("plot_driven","plot-vs-character"),("relevant_are_these","identity-relevance"),("quality_of_the_character","character-quality"),("quality_of_the_setting","setting-quality"),("moved","moved-emotionally")]:
        if k in c: return n
    return c[:16]
C={"film":"#2471a3","book":"#c0392b","tv":"#27ae60"}; built=[]
F15,B15,T15=[d[(d.year>=1915)&(d.year<=2020)] for d in (F,B,T)]; ALL15=pd.concat([F15,B15,T15],ignore_index=True)
def z(s): return (s-s.mean())/s.std()

# ===== T1 corpus table =====
try:
    rows=[]
    for d,nm in [(F,"film"),(B,"book"),(T,"tv")]:
        e=lambda lo,hi: ((d.year>=lo)&(d.year<hi)).sum()
        rows.append({"medium":nm,"N":len(d),"span":f"{int(d.year.min())}-{int(d.year.max())}","pre-1930":e(1890,1930),"1930-1960":e(1930,1960),"1960-1990":e(1960,1990),"1990-2020":e(1990,2025)})
    pd.DataFrame(rows).to_csv("results/tables/T1_corpus.csv",index=False); built.append("T1_corpus")
except Exception as ex: print("T1 err",ex)

# ===== T2 validation table =====
try:
    mv=pd.read_csv("data/validation/movie_attribute_validation.csv"); bv=pd.read_csv("data/validation/book_attribute_validation.csv")
    mr={r.attribute:r.r2 for r in mv.itertuples()}; br={r.attribute:r.r2 for r in bv.itertuples()}
    rows=[{"attribute":sn(a),"film_r2":mr.get(a),"book_r2":br.get(a),"both_media":(mr.get(a,0)>0.05 and br.get(a,0)>0.05)} for a in ATTRS]
    pd.DataFrame(rows).sort_values("film_r2",ascending=False,na_position="last").to_csv("results/tables/T2_validation.csv",index=False); built.append("T2_validation")
except Exception as ex: print("T2 err",ex)

# ===== S1 attribute dictionary =====
try:
    reg=json.load(open("/Users/samsunknight/uoft/movienet/data/canonical_full/registry.json"))
    cid2q={c:v['q'] for c,v in reg.items()}
    rows=[{"attribute":sn(a),"id":a,"survey_question":cid2q.get(a,"")[:90]} for a in ATTRS]
    pd.DataFrame(rows).to_csv("results/tables/S1_attribute_dictionary.csv",index=False); built.append("S1_attribute_dictionary")
except Exception as ex: print("S1 err",ex)

# ===== ED1 silent->sound / changepoints =====
try:
    yc=ALL15.groupby("year")[ATTRS].mean(); yc=yc[pd.concat([F15,B15,T15]).groupby("year").size()>=30]
    ycf=F[F.year.between(1905,1960)].groupby("year")[ATTRS].mean(); ycf=ycf[F.groupby("year").size()>=20]
    shift=np.sqrt(((ycf-ycf.mean())/ycf.std()).diff().pow(2).sum(axis=1))
    fig,ax=plt.subplots(figsize=(8,4.2)); ax.plot(shift.index,shift.values,color="#34495e",lw=2)
    ax.axvline(1927,color="#c0392b",ls="--",lw=2); ax.annotate("sound (1927)",(1927,shift.max()*.9),color="#c0392b",fontsize=10,fontweight="bold")
    ax.set_xlabel("year"); ax.set_ylabel("year-over-year style shift"); ax.set_title("ED1  The coming of sound as a structural break (film)",loc="left",fontweight="bold")
    plt.tight_layout(); plt.savefig("results/figures/ED1_silent_sound.png",dpi=140,bbox_inches="tight"); built.append("ED1_silent_sound")
except Exception as ex: print("ED1 err",ex)

# ===== ED2 per-medium ratchet/fashion consistency =====
try:
    RAT=[a for a in ATTRS if sn(a) in("sci-fi","#settings","#protagonists","#side-chars","immersive","world-building")]
    FAS=[a for a in ATTRS if sn(a) in("surprise","proactiveness","competence","plot-vs-character")]
    fig,ax=plt.subplots(figsize=(7,4.5)); w=.35; x=np.arange(3)
    def acm(d,grp):
        dec=d.assign(dec=(d.year//10)*10).groupby("dec")[grp].mean(); dec=dec[d.assign(dec=(d.year//10)*10).groupby("dec").size()>=25]
        return np.mean([np.corrcoef(z(dec[a]).dropna().values[:-1],z(dec[a]).dropna().values[1:])[0,1] for a in grp if len(dec[a].dropna())>5])
    rr=[acm(d,RAT) for d in (F15,B15,T15)]; ff=[acm(d,FAS) for d in (F15,B15,T15)]
    ax.bar(x-w/2,rr,w,label="ratchet attrs",color="#2471a3"); ax.bar(x+w/2,ff,w,label="fashion attrs",color="#c0392b")
    ax.set_xticks(x); ax.set_xticklabels(["film","book","tv"]); ax.set_ylabel("mean temporal autocorr"); ax.legend()
    ax.set_title("ED2  The ratchet/fashion split holds in every medium",loc="left",fontweight="bold")
    plt.tight_layout(); plt.savefig("results/figures/ED2_per_medium_split.png",dpi=140,bbox_inches="tight"); built.append("ED2_per_medium_split")
except Exception as ex: print("ED2 err",ex)

# ===== ED3 era-predictability (raw vs balanced accuracy: raw is inflated by class imbalance) =====
try:
    PAL={"film":"#c0392b","book":"#2c3e50","tv":"#27ae60"}
    fig,ax=plt.subplots(figsize=(7.2,4.6)); res=[]
    for d,nm in [(F15,"film"),(B15,"book"),(T15,"tv")]:
        dd=d.dropna(subset=ATTRS).copy(); dd["dec"]=(dd.year//10)*10; vc=dd.dec.value_counts(); dd=dd[dd.dec.isin(vc[vc>=200].index)]
        if dd.dec.nunique()<4: continue
        yp=cross_val_predict(LogisticRegression(max_iter=300),StandardScaler().fit_transform(dd[ATTRS]),dd.dec.values,cv=3)
        raw=(yp==dd.dec.values).mean(); bal=balanced_accuracy_score(dd.dec.values,yp)
        res.append((nm,raw,bal,1/dd.dec.nunique(),dd.dec.value_counts(normalize=True).max()))
    x=np.arange(len(res)); w=0.36
    ax.bar(x-w/2,[r[1] for r in res],w,color=[PAL[r[0]] for r in res],alpha=0.45,label="raw accuracy")
    ax.bar(x+w/2,[r[2] for r in res],w,color=[PAL[r[0]] for r in res],label="balanced accuracy")
    for i,r in enumerate(res):
        ax.plot([i-w,i+w],[r[3],r[3]],color="0.35",ls=":",lw=1.5,label=("uniform chance" if i==0 else None))
        ax.plot([i-w,i+w],[r[4],r[4]],color="0.6",ls="--",lw=1.2,label=("majority class" if i==0 else None))
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in res])
    ax.spines[["top","right"]].set_visible(False); ax.set_axisbelow(True); ax.grid(axis="y",alpha=0.15)
    ax.legend(frameon=False,loc="upper left",fontsize=9,ncol=2)
    ax.set_ylabel("decade-prediction accuracy"); ax.set_ylim(0,max(r[1] for r in res)*1.35)
    ax.set_title("How era-stamped is each medium's style?",loc="left",fontweight="bold")
    plt.tight_layout(); plt.savefig("results/figures/ED3_era_predictability.png",dpi=180,bbox_inches="tight"); built.append("ED3_era_predictability")
except Exception as ex: print("ED3 err",ex)

# ===== ED4 rarefied variety =====
try:
    va=[a for a in ATTRS if sn(a) in("sci-fi","fantastical","#protagonists","competence","proactiveness")]
    g=F15.dropna(subset=va).assign(dec=(F15.year//10)*10); xs,ys=[],[]
    for dec in sorted(g.dec.unique()):
        sub=g[g.dec==dec]
        if len(sub)<400: continue
        s=sub.sample(400,random_state=0); cells={tuple(min(3,int(v*4//7)) for v in row[va].values) for _,row in s.iterrows()}; xs.append(dec); ys.append(len(cells))
    fig,ax=plt.subplots(figsize=(7,4.2)); ax.plot(xs,ys,color="#c0392b",lw=2.5,marker="o",markersize=6)
    ax.spines[["top","right"]].set_visible(False); ax.set_axisbelow(True); ax.grid(axis="y",alpha=0.15)
    ax.set_xlabel("decade"); ax.set_ylabel("distinct regions of style space"); ax.set_title("Stylistic variety rises, then plateaus",loc="left",fontweight="bold")
    plt.tight_layout(); plt.savefig("results/figures/ED4_rarefied_variety.png",dpi=180,bbox_inches="tight"); built.append("ED4_rarefied_variety")
except Exception as ex: print("ED4 err",ex)

# ===== ED5 clustering silhouette (continuum) =====
try:
    Xs=StandardScaler().fit_transform(ALL15[ATTRS].fillna(ALL15[ATTRS].mean())); ks=[2,3,4,5,6,8,10,12]; sils=[]
    for k in ks:
        km=KMeans(k,n_init=5,random_state=0).fit(Xs); sils.append(silhouette_score(Xs[::25],km.labels_[::25]))
    fig,ax=plt.subplots(figsize=(7,4.2)); ax.plot(ks,sils,color="#2c3e50",lw=2.5,marker="o",markersize=6)
    ax.axhspan(0,0.25,color="grey",alpha=.06)
    ax.spines[["top","right"]].set_visible(False); ax.set_axisbelow(True); ax.grid(axis="y",alpha=0.15)
    ax.set_xlabel(r"number of clusters $k$"); ax.set_ylabel("silhouette score"); ax.set_ylim(0,0.35); ax.set_title("Style space is a filled continuum, not discrete types",loc="left",fontweight="bold")
    plt.tight_layout(); plt.savefig("results/figures/ED5_continuum.png",dpi=180,bbox_inches="tight"); built.append("ED5_continuum")
except Exception as ex: print("ED5 err",ex)

# ===== S2 full lead-lag matrix =====
try:
    def ys_(d,a,mn=15):
        gg=d.dropna(subset=[a]).groupby("year")[a].agg(["mean","count"]); return gg[gg["count"]>=mn]["mean"]
    def ll(sx,sy):
        yrs=sorted(set(sx.index)&set(sy.index))
        if len(yrs)<25: return None
        x=sx.reindex(yrs).rolling(3,center=True).mean().diff().dropna(); yv=sy.reindex(yrs).rolling(3,center=True).mean().diff().dropna()
        yy=sorted(set(x.index)&set(yv.index)); x,yv=x.reindex(yy).values,yv.reindex(yy).values; best=(0,0)
        for lag in range(-6,7):
            a2,b2=(x[lag:],yv[:len(yv)-lag]) if lag>=0 else (x[:lag],yv[-lag:]); n=min(len(a2),len(b2))
            if n>15 and abs(np.corrcoef(a2[:n],b2[:n])[0,1])>abs(best[1]): best=(lag,np.corrcoef(a2[:n],b2[:n])[0,1])
        rng=np.random.RandomState(0); null=[abs(np.corrcoef(x,rng.permutation(yv))[0,1]) for _ in range(400)]
        return best[0],round(best[1],3),round(np.mean([nn>=abs(best[1]) for nn in null]),3)
    rows=[]
    for a in ATTRS:
        for (dx,nx),(dy,ny) in [((B15,"book"),(F15,"film")),((F15,"film"),(T15,"tv")),((B15,"book"),(T15,"tv"))]:
            r=ll(ys_(dx,a),ys_(dy,a))
            if r: rows.append({"attribute":sn(a),"pair":f"{nx}-{ny}","lag_yrs":r[0],"peak_r":r[1],"perm_p":r[2],"leader":(nx if r[0]>0 else ny)})
    pd.DataFrame(rows).sort_values("perm_p").to_csv("results/tables/S2_leadlag_matrix.csv",index=False); built.append("S2_leadlag_matrix")
except Exception as ex: print("S2 err",ex)

# ===== S3 window sensitivity =====
try:
    RAT=[a for a in ATTRS if sn(a) in("sci-fi","#settings","#protagonists","immersive","world-building")]
    rows=[]
    for lo in [1915,1930,1950]:
        Fw=F[F.year>=lo]; dec=Fw.assign(dec=(Fw.year//10)*10).groupby("dec")[RAT].mean(); dec=dec[Fw.assign(dec=(Fw.year//10)*10).groupby("dec").size()>=25]
        ac=np.mean([np.corrcoef(z(dec[a]).dropna().values[:-1],z(dec[a]).dropna().values[1:])[0,1] for a in RAT if len(dec[a].dropna())>5])
        rows.append({"window":f"{lo}+","ratchet_autocorr":round(ac,3),"n_films":len(Fw)})
    pd.DataFrame(rows).to_csv("results/tables/S3_window_sensitivity.csv",index=False); built.append("S3_window_sensitivity")
except Exception as ex: print("S3 err",ex)

print("BUILT:", built)
print("figures:", sorted([f for f in os.listdir("results/figures") if f.endswith(".png")]))
print("tables:", sorted(os.listdir("results/tables")))
