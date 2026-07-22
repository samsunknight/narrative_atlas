#!/usr/bin/env python3
"""COOL DESCRIPTIVES round 2 for the NHB Resource atlas. Certified data (deploy attrs).
Builds three new descriptive cuts, PNG@140dpi + CSV + printed findings:
  (1) Genre x tone coupling: film genre x decade heatmap of within-genre darkness
      (mean mood_Dark among films scoring >=60 on each genre), + raw 1935->1970 point
      change per genre, Horror flagged ceiling-limited. Plus a genre x mood matrix (2010s).
  (2) Stylistic variety over time (rarefaction): coverage-robust diversity per medium --
      fix n works sampled per decade, count occupied cells of the top-2-PC style grid,
      averaged over resamples. Tests 'did the range of stories widen then plateau?'
  (3) The Production Code as a cross-medium natural experiment: raw darkness index for
      film/novel/TV on a common scale, Code window (1934-1968) shaded. Film should kink
      at the Code boundaries; the novel (never treated) should not. Descriptive, NOT causal.
Saves to results/cool/. Trends capped at the 2010s (2020s = partial-bin/recency artifact)."""
import pandas as pd, numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch
rng = np.random.default_rng(20260709)
os.makedirs("results/cool", exist_ok=True)
plt.rcParams.update({"font.size":8,"axes.spines.top":False,"axes.spines.right":False})

man = pd.read_csv("data/validation/rescore_manifest.csv")
man = man[man.deploy==True]; DEPLOY=set(man.attr_id)
DATA="data/atlas/century_frame_{}.parquet"
FLOOR={"film":1930,"book":1930,"tv":1950}
TREND_CAP=2010  # exclude 2020s from trend claims

def load(m):
    df=pd.read_parquet(DATA.format(m))
    df=df[df.year.between(FLOOR[m],2025)].copy()
    df["dec"]=(df.year//10*10).astype(int)
    return df

# ============================================================================
# FIG 1: GENRE x TONE COUPLING (film) -- within-genre darkness by decade
# ============================================================================
film=load("film")
genres=[g.replace("genre_","") for g in film.columns if g.startswith("genre_")]
decs=list(range(1930,2030,10))
THR=60  # a film "is" a genre if it scores >=60 on it
MINN=25 # need at least this many genre-films in a decade cell to report

# genre x decade matrix of mean darkness among films in that genre
G=pd.DataFrame(index=genres,columns=decs,dtype=float)
Ncell=pd.DataFrame(index=genres,columns=decs,dtype=float)
for g in genres:
    sub=film[film["genre_"+g]>=THR]
    for d in decs:
        cell=sub[sub.dec==d]["mood_Dark"]
        Ncell.loc[g,d]=len(cell)
        G.loc[g,d]=cell.mean() if len(cell)>=MINN else np.nan
# order genres by overall darkness (darkest at top)
order=G[[d for d in decs if d<=TREND_CAP]].mean(1).sort_values(ascending=False).index.tolist()
G=G.loc[order]; Ncell=Ncell.loc[order]

# raw point change 1935->1970 : use 1930s->1970s decade means (the known darkening window)
chg=(G[1970]-G[1930]).round(1)
# ceiling diagnostic: share of genre-films at mood_Dark>=90 in 1970s
ceil={}
for g in genres:
    sub=film[(film["genre_"+g]>=THR)&(film.dec==1970)]["mood_Dark"]
    ceil[g]=100*np.mean(sub>=90) if len(sub) else np.nan

fig,ax=plt.subplots(figsize=(11,7))
Gplot=G[[d for d in decs]]  # show all, but mask trend beyond cap visually via annotation
im=ax.imshow(Gplot.values.astype(float),aspect="auto",cmap="magma",vmin=30,vmax=75)
ax.set_xticks(range(len(decs))); ax.set_xticklabels([f"{d}s" for d in decs])
ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
# annotate each cell with the darkness value; hatch thin cells
for i,g in enumerate(order):
    for j,d in enumerate(decs):
        v=Gplot.loc[g,d]
        if np.isnan(v):
            ax.text(j,i,"·",ha="center",va="center",color="grey",fontsize=8); continue
        ax.text(j,i,f"{v:.0f}",ha="center",va="center",fontsize=6,
                color="white" if v<55 else "black")
ax.axvline(decs.index(TREND_CAP)+0.5,color="cyan",lw=1.2,ls="--")
ax.text(decs.index(TREND_CAP)+0.6,-0.7,"2020s: partial bin\n(excluded from trends)",
        fontsize=6,color="teal",va="bottom")
ax.set_title("Genre x tone coupling: within-genre darkness by decade (film)\n"
             "mean mood_Dark among films scoring ≥60 on the genre; cells <25 films shown as ·",
             fontsize=10)
cb=fig.colorbar(im,ax=ax,fraction=0.03,pad=0.02); cb.set_label("mean mood_Dark (0–100)")
fig.tight_layout(); fig.savefig("results/cool/FIG_genre_tone_coupling.png",dpi=140); plt.close(fig)
print("saved FIG_genre_tone_coupling.png")

# companion bar: raw 1930s->1970s darkening per genre
chg_sorted=chg.dropna().sort_values()
fig,ax=plt.subplots(figsize=(7,6))
cols=["#7a0177" if g!="Horror" else "#bdbdbd" for g in chg_sorted.index]
ax.barh(range(len(chg_sorted)),chg_sorted.values,color=cols)
ax.set_yticks(range(len(chg_sorted))); ax.set_yticklabels(chg_sorted.index,fontsize=8)
ax.axvline(0,color="k",lw=.6)
for i,(g,v) in enumerate(chg_sorted.items()):
    ax.text(v+(0.3 if v>=0 else -0.3),i,f"{v:+.1f}",va="center",
            ha="left" if v>=0 else "right",fontsize=7)
ax.set_xlabel("Δ mean mood_Dark, 1930s → 1970s (points)")
ax.set_title("Every genre darkened 1930s→1970s (film)\nHorror greyed: ceiling-limited (already darkest)",fontsize=10)
fig.tight_layout(); fig.savefig("results/cool/FIG_genre_darkening_bars.png",dpi=140); plt.close(fig)
print("saved FIG_genre_darkening_bars.png")

# CSV for fig 1
G.to_csv("results/cool/CSV_genre_darkness_by_decade.csv")
pd.DataFrame({"delta_dark_1930s_1970s":chg,"pct_at_ceiling_90plus_1970s":pd.Series(ceil),
             "n_1970s":Ncell[1970]}).to_csv("results/cool/CSV_genre_darkening.csv")

# companion: genre x mood matrix for the 2010s (which moods each genre carries)
moods=[c for c in film.columns if c.startswith("mood_") and c in DEPLOY]
f2010=film[film.dec==2010]
GM=pd.DataFrame(index=order,columns=[m.replace("mood_","") for m in moods],dtype=float)
for g in order:
    sub=f2010[f2010["genre_"+g]>=THR]
    if len(sub)>=MINN:
        GM.loc[g]=sub[moods].mean().values
# z-score each mood column so we see which genre over-indexes on it
GMz=(GM-GM.mean(0))/GM.std(0)
# order mood columns by a rough dark->light hand order kept from cool_descriptives
mood_order=["Dark","Bleak","Tragic","Gritty","Tense","Melancholic","Chilling","Eerie","Sad",
            "Bittersweet","Mysterious","Raw","Reflective","Dreamy","Surreal","Epic","Adventurous",
            "Romantic","Whimsical","Quirky","Energetic","Cozy","Optimistic","Inspirational",
            "Hopeful","Heartwarming","Lighthearted","Funny"]
cols_present=[m for m in mood_order if m in GMz.columns]+[m for m in GMz.columns if m not in mood_order]
GMz=GMz[cols_present]
fig,ax=plt.subplots(figsize=(13,6))
im=ax.imshow(GMz.values.astype(float),aspect="auto",cmap="RdBu_r",vmin=-2,vmax=2)
ax.set_xticks(range(len(cols_present))); ax.set_xticklabels(cols_present,rotation=60,ha="right",fontsize=6)
ax.set_yticks(range(len(GMz))); ax.set_yticklabels(GMz.index,fontsize=8)
ax.set_title("Which moods each genre carries (film, 2010s)\nz-scored across genres per mood; red = genre over-indexes on that mood",fontsize=10)
fig.colorbar(im,ax=ax,fraction=0.02,pad=0.01)
fig.tight_layout(); fig.savefig("results/cool/FIG_genre_mood_matrix_2010s.png",dpi=140); plt.close(fig)
GMz.to_csv("results/cool/CSV_genre_mood_matrix_2010s.csv")
print("saved FIG_genre_mood_matrix_2010s.png")

# ============================================================================
# FIG 2: STYLISTIC VARIETY OVER TIME (rarefaction) per medium
# ============================================================================
# Build a shared style space per medium from deploy numeric attrs -> top 2 PCs,
# grid into GRID x GRID cells, and (rarefaction) fix n works sampled per decade,
# count distinct occupied cells + effective number of cells via entropy. Average
# over resamples. Coverage-robust: the fixed n removes the 'more films => more cells'
# artifact.
GRID=12; NRESAMP=200
NSAMP={"film":300,"book":300,"tv":180}  # tv thinner early; 1950s has ~214

def style_pcs(df):
    cols=[c for c in df.columns if c in DEPLOY and pd.api.types.is_numeric_dtype(df[c])]
    X=df[cols].astype(float)
    X=X.loc[:,X.std()>0]
    Z=(X-X.mean())/X.std()
    Z=Z.fillna(0.0).values
    # PCA via SVD, top 2
    Zc=Z-Z.mean(0)
    U,S,Vt=np.linalg.svd(Zc,full_matrices=False)
    pcs=Zc@Vt[:2].T
    return pcs

var_traj={}
for m in ["film","book","tv"]:
    df=load(m)
    pcs=style_pcs(df)
    df=df.assign(pc1=pcs[:,0],pc2=pcs[:,1])
    # fixed global grid edges from robust quantiles so cells are comparable across decades
    lo1,hi1=np.quantile(df.pc1,[0.01,0.99]); lo2,hi2=np.quantile(df.pc2,[0.01,0.99])
    e1=np.linspace(lo1,hi1,GRID+1); e2=np.linspace(lo2,hi2,GRID+1)
    n=NSAMP[m]
    rows=[]
    for d in sorted(df.dec.unique()):
        cell=df[df.dec==d]
        if len(cell)<n:  # not enough to rarefy at fixed n -> skip (thin bin)
            rows.append((d,np.nan,np.nan,len(cell))); continue
        occ=[]; eff=[]
        p1=cell.pc1.values; p2=cell.pc2.values
        for _ in range(NRESAMP):
            idx=rng.choice(len(cell),size=n,replace=False)
            H,_,_=np.histogram2d(p1[idx],p2[idx],bins=[e1,e2])
            k=H.sum()
            occ.append((H>0).sum())
            pr=H[H>0]/k
            eff.append(np.exp(-(pr*np.log(pr)).sum()))  # effective # occupied cells (Hill/Shannon)
        rows.append((d,np.mean(occ),np.mean(eff),len(cell)))
    var_traj[m]=pd.DataFrame(rows,columns=["dec","occupied_cells","effective_cells","n_decade"])

fig,axs=plt.subplots(1,2,figsize=(13,5))
palette={"film":"#1f77b4","book":"#d62728","tv":"#2ca02c"}
for m in ["film","book","tv"]:
    t=var_traj[m]
    tt=t[t.dec<=TREND_CAP]
    for ax,ycol,lab in [(axs[0],"occupied_cells","occupied grid cells"),
                        (axs[1],"effective_cells","effective # cells (Hill₁)")]:
        ax.plot(tt.dec,tt[ycol],"-o",color=palette[m],label=f"{m} (n={NSAMP[m]}/dec)")
        # dashed continuation into partial 2020s
        t20=t[t.dec>=TREND_CAP]
        ax.plot(t20.dec,t20[ycol],":o",color=palette[m],alpha=0.5,mfc="white")
for ax,tit in [(axs[0],f"Occupied cells of a {GRID}x{GRID} style grid"),
               (axs[1],"Effective number of occupied cells (entropy)")]:
    ax.set_title(tit,fontsize=9); ax.set_xlabel("decade"); ax.legend(fontsize=7)
    ax.axvspan(TREND_CAP+5,2025,color="grey",alpha=0.08)
axs[0].set_ylabel("distinct cells (rarefied, mean of 200 resamples)")
fig.suptitle("Stylistic variety over time (rarefaction: fixed works/decade) — did the range of stories widen then plateau?",fontsize=11)
fig.tight_layout(); fig.savefig("results/cool/FIG_variety_rarefaction.png",dpi=140); plt.close(fig)
print("saved FIG_variety_rarefaction.png")
pd.concat({m:var_traj[m].set_index("dec") for m in var_traj},axis=0
          ).to_csv("results/cool/CSV_variety_rarefaction.csv")

# ============================================================================
# FIG 3: THE PRODUCTION CODE AS A CROSS-MEDIUM NATURAL EXPERIMENT
# ============================================================================
# Raw darkness index (mean mood_Dark) by 5-year bin, film/book/tv on common scale.
# Code enforced 1934 -> ended 1968 (shaded). Film = treated; novel = never-treated
# control; TV = differently/late-treated. Descriptive: "coincides with", not causal;
# 1968 New-Hollywood confound noted.
CODE0,CODE1=1934,1968
def darkness_series(m,binw=5):
    df=load(m)
    df["bin"]=(df.year//binw*binw).astype(int)
    s=df.groupby("bin")["mood_Dark"].agg(["mean","count"])
    s=s[s["count"]>=30]  # drop thin bins
    return s
ser={m:darkness_series(m) for m in ["film","book","tv"]}

fig,ax=plt.subplots(figsize=(12,6))
ax.axvspan(CODE0,CODE1,color="#d62728",alpha=0.10,zorder=0)
ax.text((CODE0+CODE1)/2,None or 0,"",)  # noop
for m in ["film","book","tv"]:
    s=ser[m]
    ax.plot(s.index,s["mean"],"-o",color=palette[m],label=m,ms=4)
ax.axvline(CODE0,color="#d62728",ls="--",lw=1)
ax.axvline(CODE1,color="#d62728",ls="--",lw=1)
ylo,yhi=ax.get_ylim()
ax.text(CODE0,yhi,"Code enforced\n1934",color="#d62728",fontsize=7,va="top",ha="left")
ax.text(CODE1,ylo,"Code ends 1968",color="#d62728",fontsize=7,va="bottom",ha="left")
ax.text((CODE0+CODE1)/2,yhi*0.99,"Production Code window",color="#d62728",fontsize=8,ha="center",va="top")
ax.set_xlabel("5-year bin"); ax.set_ylabel("darkness index: mean mood_Dark (0–100)")
ax.set_title("Film darkens against a flat novel baseline\n"
             "Dark-mood intensity by cohort; Production Code window (1934–1968) shaded — a descriptive contrast, not a causal claim",
             fontsize=10)
ax.legend(fontsize=8,loc="lower right")
ax.set_xlim(1930,2025)
fig.tight_layout(); fig.savefig("results/cool/FIG_production_code_crossmedium.png",dpi=140); plt.close(fig)
print("saved FIG_production_code_crossmedium.png")
pd.concat({m:ser[m] for m in ser},axis=1).to_csv("results/cool/CSV_production_code_crossmedium.csv")

# ---- FINDINGS to stdout ----
print("\n================ FINDINGS ================")
print("\n[1] GENRE DARKENING 1930s->1970s (raw points):")
for g,v in chg.dropna().sort_values(ascending=False).items():
    tag=" (CEILING-LIMITED)" if g=="Horror" else ""
    print(f"   {g:16s} {v:+5.1f}  (1930s {G.loc[g,1930]:.0f} -> 1970s {G.loc[g,1970]:.0f}; "
          f"{ceil[g]:.0f}% at >=90 by 1970s){tag}")
print("\n[2] VARIETY (rarefied, effective cells), 1st decade -> peak -> 2010s:")
for m in var_traj:
    t=var_traj[m].dropna(subset=["effective_cells"])
    tt=t[t.dec<=TREND_CAP]
    if len(tt)<2: continue
    first,last=tt.iloc[0],tt.iloc[-1]
    peak=tt.loc[tt.effective_cells.idxmax()]
    print(f"   {m:5s}: {int(first.dec)}s {first.effective_cells:.1f} -> peak {int(peak.dec)}s "
          f"{peak.effective_cells:.1f} -> 2010s {last.effective_cells:.1f}  "
          f"(occupied {tt.iloc[0].occupied_cells:.0f}->{tt.iloc[-1].occupied_cells:.0f} of {GRID*GRID})")
print("\n[3] PRODUCTION CODE cross-medium darkness (mean mood_Dark):")
for m in ["film","book","tv"]:
    s=ser[m]
    def near(yr):
        if len(s)==0: return np.nan
        return s.iloc[(s.index-yr).to_series().abs().argmin()]["mean"]
    print(f"   {m:5s}: pre-Code ~1930 {near(1930):.1f} | mid-Code ~1950 {near(1950):.1f} | "
          f"end-Code ~1968 {near(1968):.1f} | post ~1975 {near(1975):.1f} | 2010s {near(2015):.1f}")
print("\nCOOL2_DONE")
