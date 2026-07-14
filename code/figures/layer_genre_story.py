#!/usr/bin/env python3
"""GENRE LAYER of the Narrative Atlas — full analysis suite + the complete story.
18 genre attributes (0-100 intensities: Action..Western), scored across film/book/tv.
Genres are SHARED across media, so this layer tests whether the same genre rises/falls
on the same clock everywhere. Builds: (1) genre lifecycles small-multiples + summary CSV;
(2) cross-medium co-movement; (3) per-work genre entropy (omnivore test) w/ coverage-confound
check; (4) genre x era fingerprints; (5) reception dissociation (rating vs reach, film);
(6) validation vs IMDb tags (AUC). Saves to results/layers/genre/.
Style mirrors code/cool_descriptives.py. Discipline: floor film/book 1930, tv 1950;
CAP trend characterization at 2010s (2020s = coverage artifact); percentile-standardize
within medium for cross-medium comparisons.
"""
import pandas as pd, numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

BR="data/atlas"
OUT="results/layers/genre"; os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({"font.size":8,"axes.spines.top":False,"axes.spines.right":False})

GENRES=["Action","Adventure","Animation","Comedy","Crime","Documentary","Drama","Family",
        "Fantasy","Historical","Horror","Musical","Mystery","Romance","Science_Fiction",
        "Thriller","War","Western"]
GC=[f"genre_{g}" for g in GENRES]
FLOOR={"film":1930,"book":1930,"tv":1950}     # medium-specific coverage floor
CAP=2010                                       # last decade used to characterize trends
def nice(g): return g.replace("_"," ").replace("Science Fiction","Sci-Fi")

def load(m):
    df=pd.read_parquet(f"{BR}/century_frame_{m}.parquet")
    df=df[df.year>=FLOOR[m]].copy(); df=df[df.year<=2025]
    df["dec"]=(df.year//10*10).astype(int); df["medium"]=m
    return df
FR={m:load(m) for m in ["film","book","tv"]}
for m,df in FR.items():
    print(f"{m}: n={len(df):,}  decades {df.dec.min()}s-{df.dec.max()}s")

# ============================================================================
# (1) GENRE LIFECYCLES  — the headline
# ============================================================================
# per medium x genre x decade mean intensity
traj={}   # traj[m] = DataFrame decade x genre
for m,df in FR.items():
    traj[m]=df.groupby("dec")[GC].mean()
    traj[m].columns=GENRES

def classify(s):
    """s: pandas Series decade->intensity, capped at CAP for characterization."""
    sc=s[s.index<=CAP]
    if len(sc)<3: return "n/a",None,np.nan,np.nan
    first,last=sc.iloc[0],sc.iloc[-1]; peak_dec=int(sc.idxmax()); peak_val=sc.max()
    lo,hi=sc.index.min(),sc.index.max()
    delta=last-first
    frac=(peak_dec-lo)/(hi-lo)                       # where in the century the peak sits
    # revival = late uptick after a trough below both ends
    trough=sc.min(); trough_dec=sc.idxmin()
    rng=sc.max()-sc.min()
    late_rise = (last-trough)>0.35*rng and trough_dec<peak_dec and last>first
    if late_rise and frac>0.6 and abs(delta)<0.6*rng:
        shape="revival"
    elif frac<0.25 and delta<-0.25*rng: shape="decline"
    elif frac>0.75 and delta> 0.25*rng: shape="rise"
    elif 0.25<=frac<=0.75 and (peak_val-max(first,last))>0.25*rng: shape="hump"
    elif delta> 0.25*rng: shape="rise"
    elif delta<-0.25*rng: shape="decline"
    else: shape="flat"
    return shape,peak_dec,round(delta,2),round(100*delta/(first+1e-9),1)

rows=[]
for m in ["film","book","tv"]:
    T=traj[m]
    for g in GENRES:
        s=T[g].dropna()
        shape,peak,delta,pct=classify(s)
        first_dec=int(s.index.min()); last_use=s[s.index<=CAP]
        rows.append({"medium":m,"genre":nice(g),"shape":shape,"peak_decade":peak,
                     "mean_first_dec":round(s.iloc[0],2),
                     f"mean_{CAP}s":round(last_use.iloc[-1],2) if len(last_use) else np.nan,
                     "delta_first_to_2010s":delta,"pct_change":pct,
                     "mean_2020s":round(s.loc[2020],2) if 2020 in s.index else np.nan})
life=pd.DataFrame(rows)
life.to_csv(f"{OUT}/genre_lifecycles.csv",index=False)
print("\n=== GENRE LIFECYCLES (film) ===")
print(life[life.medium=="film"][["genre","shape","peak_decade","mean_first_dec","mean_2010s","pct_change"]].to_string(index=False))

# small-multiples: 18 panels, one per genre, 3 media lines
COL={"film":"#1f77b4","book":"#d62728","tv":"#2ca02c"}
fig,axs=plt.subplots(3,6,figsize=(17,9),sharex=True)
for ax,g in zip(axs.flat,GENRES):
    for m in ["film","book","tv"]:
        s=traj[m][g]
        ax.plot(s.index,s.values,color=COL[m],lw=1.6,label=m)
        # mark 2020s as hollow (coverage artifact)
        if 2020 in s.index: ax.plot(2020,s.loc[2020],'o',ms=3,mfc="white",mec=COL[m])
    ax.set_title(nice(g),fontsize=9); ax.set_xlim(1930,2025)
    ax.axvspan(2015,2025,color="grey",alpha=0.06)
axs.flat[0].legend(fontsize=7,loc="upper left")
fig.suptitle("Genre lifecycles across the century — mean 0-100 intensity by decade "
             "(film blue, the novel red, tv green; shaded = 2020s coverage artifact)",fontsize=12)
fig.supylabel("mean genre intensity (0-100)")
fig.tight_layout(); fig.savefig(f"{OUT}/FIG_genre_lifecycles.png",dpi=140); plt.close(fig)
print("saved FIG_genre_lifecycles.png")

# ============================================================================
# (2) CROSS-MEDIUM CO-MOVEMENT
# ============================================================================
# align on common decades (>=1950 so tv is present), percentile-standardize within
# medium isn't needed for correlation of shape; use raw decade means on shared window.
common=sorted(set(traj["film"].index)&set(traj["book"].index)&set(traj["tv"].index))
common=[d for d in common if d<=CAP]
xm=[]
for g in GENRES:
    fb=spearmanr(traj["film"].loc[common,g],traj["book"].loc[common,g])[0]
    ft=spearmanr(traj["film"].loc[common,g],traj["tv"].loc[common,g])[0]
    bt=spearmanr(traj["book"].loc[common,g],traj["tv"].loc[common,g])[0]
    xm.append({"genre":nice(g),"rho_film_book":round(fb,2),"rho_film_tv":round(ft,2),
               "rho_book_tv":round(bt,2),"mean_rho":round(np.nanmean([fb,ft,bt]),2)})
xmdf=pd.DataFrame(xm).sort_values("mean_rho",ascending=False)
xmdf["verdict"]=np.where(xmdf.mean_rho>=0.5,"shared clock",
                 np.where(xmdf.mean_rho<=0.0,"medium-specific","mixed"))
xmdf.to_csv(f"{OUT}/genre_crossmedium.csv",index=False)
print("\n=== CROSS-MEDIUM CO-MOVEMENT (Spearman of decade trajectories, shared window "
      f"{common[0]}s-{common[-1]}s) ===")
print(xmdf.to_string(index=False))

# ============================================================================
# (3) GENRE ENTROPY  — the omnivore test  (+coverage confound check)
# ============================================================================
def work_entropy(df):
    G=df[GC].clip(lower=0).values.astype(float)
    s=G.sum(1,keepdims=True); s[s==0]=1
    P=G/s
    with np.errstate(divide="ignore",invalid="ignore"):
        H=-(P*np.log(P+1e-12)).sum(1)
    return H/np.log(len(GC))    # normalized 0-1
ent_rows=[]; ent_by_dec={}
for m,df in FR.items():
    df=df.copy(); df["H"]=work_entropy(df)
    ent_by_dec[m]=df.groupby("dec").agg(H=("H","mean"),n=("H","size"))
for m in ["film","book","tv"]:
    e=ent_by_dec[m]; ec=e[e.index<=CAP]
    # trend: Spearman of entropy vs decade; coverage confound: entropy vs log(n)
    tr=spearmanr(ec.index,ec.H)[0]
    conf=spearmanr(ec.H,np.log(ec.n))[0]
    ent_rows.append({"medium":m,"H_first_dec":round(ec.H.iloc[0],3),"H_2010s":round(ec.H.iloc[-1],3),
                     "delta_H":round(ec.H.iloc[-1]-ec.H.iloc[0],3),
                     "rho_H_vs_decade":round(tr,2),"rho_H_vs_logN_coverage":round(conf,2)})
entdf=pd.DataFrame(ent_rows); entdf.to_csv(f"{OUT}/genre_entropy.csv",index=False)
print("\n=== GENRE ENTROPY (omnivore test; blending across the 18 genres) ===")
print(entdf.to_string(index=False))

fig,axs=plt.subplots(1,2,figsize=(13,4.5))
for m in ["film","book","tv"]:
    e=ent_by_dec[m]
    axs[0].plot(e.index,e.H,color=COL[m],lw=1.8,marker="o",ms=3,label=m)
axs[0].axvspan(2015,2025,color="grey",alpha=0.08)
axs[0].set_title("Per-work genre entropy by decade\n(normalized Shannon over 18 genres)")
axs[0].set_xlabel("decade"); axs[0].set_ylabel("mean normalized entropy"); axs[0].legend(fontsize=7)
for m in ["film","book","tv"]:
    e=ent_by_dec[m][ent_by_dec[m].index<=CAP]
    axs[1].scatter(np.log(e.n),e.H,color=COL[m],s=18,label=m)
axs[1].set_title("Coverage confound: entropy vs sample size\n(each point = a decade)")
axs[1].set_xlabel("log(n works in decade)"); axs[1].set_ylabel("mean entropy"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/FIG_genre_entropy.png",dpi=140); plt.close(fig)
print("saved FIG_genre_entropy.png")

# ============================================================================
# (4) GENRE x ERA FINGERPRINTS  — which genres define each decade
# ============================================================================
# z-score each genre across decades (within medium), capped at CAP; heatmap ordered by peak.
def fingerprint(m):
    T=traj[m]; T=T[T.index<=CAP]
    Z=T.sub(T.mean(0),axis=1).div(T.std(0)+1e-9,axis=1)   # decade x genre, z within genre
    return Z
Zfilm=fingerprint("film")
peak=Zfilm.idxmax(axis=0)                       # decade each genre peaks
order=peak.sort_values().index
Zo=Zfilm[order].T                               # genre x decade, ordered by peak decade
fig,ax=plt.subplots(figsize=(9,7))
im=ax.imshow(Zo.values,aspect="auto",cmap="RdBu_r",vmin=-1.8,vmax=1.8)
ax.set_xticks(range(len(Zo.columns))); ax.set_xticklabels([f"{d}s" for d in Zo.columns],rotation=45,ha="right")
ax.set_yticks(range(len(Zo.index))); ax.set_yticklabels([nice(g) for g in Zo.index])
ax.set_title("Genre x era fingerprint (film): z within genre across decades,\nordered by peak decade — the diagonal wave of genre",fontsize=10)
cb=fig.colorbar(im,ax=ax,fraction=0.03,pad=0.02); cb.set_label("z (dark red = high for this genre's history)")
fig.tight_layout(); fig.savefig(f"{OUT}/FIG_genre_fingerprint.png",dpi=140); plt.close(fig)
print("saved FIG_genre_fingerprint.png")

print("\n=== DECADE FINGERPRINTS (film): top 3 genres most distinctive per decade ===")
fp_rows=[]
for d in Zfilm.index:
    top=Zfilm.loc[d].sort_values(ascending=False).head(3)
    fp_rows.append({"decade":f"{d}s","defining_genres":", ".join(f"{nice(g)} (+{v:.1f})" for g,v in top.items())})
    print(f"  {d}s: "+", ".join(f"{nice(g)} (z={v:+.1f})" for g,v in top.items()))
pd.DataFrame(fp_rows).to_csv(f"{OUT}/genre_decade_fingerprints.csv",index=False)

# ============================================================================
# (5) RECEPTION (film): which genres rewarded on RATING vs REACH
# ============================================================================
R=pd.read_parquet("results/reception/atlas_reception_matched.parquet")
R=R[R.medium=="film"].copy()
R=R[R.n_ratings>=5].copy()                       # drop 1-rating noise
R["logreach"]=np.log(R.n_ratings)
# residualize outcomes on decade (era controls) then correlate w/ genre intensity
R["dec"]=(R.year//10*10)
def resid(y,by):
    mu=R.groupby(by)[y].transform("mean"); return R[y]-mu
R["rating_r"]=resid("mean_rating","dec"); R["reach_r"]=resid("logreach","dec")
rec_rows=[]
for g in GC:
    d=R[[g,"rating_r","reach_r"]].dropna()
    if len(d)<200: continue
    rr=spearmanr(d[g],d.rating_r)[0]; ch=spearmanr(d[g],d.reach_r)[0]
    rec_rows.append({"genre":nice(g.replace("genre_","")),"rho_rating":round(rr,3),
                     "rho_reach":round(ch,3),"n":len(d),"reach_minus_rating":round(ch-rr,3)})
rec=pd.DataFrame(rec_rows).sort_values("reach_minus_rating",ascending=False)
rec.to_csv(f"{OUT}/genre_reception.csv",index=False)
print(f"\n=== RECEPTION (film, n~{len(R):,}; decade-residualized Spearman) ===")
print("Rewarded on CRITICAL RATING (top):")
print(rec.sort_values("rho_rating",ascending=False).head(5)[["genre","rho_rating","rho_reach"]].to_string(index=False))
print("Rewarded on REACH / popularity (top):")
print(rec.sort_values("rho_reach",ascending=False).head(5)[["genre","rho_rating","rho_reach"]].to_string(index=False))

fig,ax=plt.subplots(figsize=(7.5,6))
ax.axhline(0,color="grey",lw=.5); ax.axvline(0,color="grey",lw=.5)
ax.scatter(rec.rho_rating,rec.rho_reach,s=30,color="#8e44ad")
for _,r in rec.iterrows():
    ax.annotate(r.genre,(r.rho_rating,r.rho_reach),fontsize=7,xytext=(3,2),textcoords="offset points")
lim=max(rec.rho_rating.abs().max(),rec.rho_reach.abs().max())*1.15
ax.plot([-lim,lim],[-lim,lim],ls="--",color="grey",lw=.6)
ax.set_xlabel("assoc. with critical RATING (decade-resid. Spearman)")
ax.set_ylabel("assoc. with REACH / popularity")
ax.set_title("Film reception: what genres buy — rating vs reach")
fig.tight_layout(); fig.savefig(f"{OUT}/FIG_genre_reception.png",dpi=140); plt.close(fig)
print("saved FIG_genre_reception.png")

# ============================================================================
# (6) VALIDATION vs IMDb TAGS (AUC)
# ============================================================================
gv=pd.read_csv("results/tables/genre_validation.csv")
print("\n=== EXISTING genre_validation.csv (scalar constructs vs IMDb tags) ===")
print(gv.to_string(index=False))
# recompute AUC for the genre_ LAYER attributes directly, film idx <-> imdb_film_genres
img=pd.read_csv("data/matched/imdb_film_genres.csv")
Ffull=pd.read_parquet(f"{BR}/century_frame_film.parquet")
J=Ffull.merge(img,left_on="idx",right_on="id",how="inner")
IMDB_MAP={"Action":"Action","Adventure":"Adventure","Animation":"Animation","Comedy":"Comedy",
    "Crime":"Crime","Documentary":"Documentary","Drama":"Drama","Family":"Family","Fantasy":"Fantasy",
    "Historical":"History","Horror":"Horror","Musical":"Musical","Mystery":"Mystery","Romance":"Romance",
    "Science_Fiction":"Sci-Fi","Thriller":"Thriller","War":"War","Western":"Western"}
vrows=[]
for g in GENRES:
    tag=IMDB_MAP[g]
    y=J.imdb_genres.fillna("").str.contains(rf"\b{tag}\b",regex=True).astype(int)
    x=J[f"genre_{g}"]
    d=pd.DataFrame({"x":x,"y":y}).dropna()
    if d.y.sum()<40:
        vrows.append({"genre":nice(g),"imdb_tag":tag,"auc":np.nan,"n_pos":int(d.y.sum())}); continue
    vrows.append({"genre":nice(g),"imdb_tag":tag,"auc":round(roc_auc_score(d.y,d.x),3),"n_pos":int(d.y.sum())})
vg=pd.DataFrame(vrows).sort_values("auc",ascending=False)
vg.to_csv(f"{OUT}/genre_validation_layer.csv",index=False)
print("\n=== RECOMPUTED AUC for the 18 genre-LAYER attrs vs IMDb tags (film, idx-matched) ===")
print(vg.to_string(index=False))
print(f"\nAUC summary: median={vg.auc.median():.3f}  min={vg.auc.min():.3f}  max={vg.auc.max():.3f}  "
      f"n>=0.75: {(vg.auc>=0.75).sum()}/{vg.auc.notna().sum()}")
print("GENRE_LAYER_DONE")
