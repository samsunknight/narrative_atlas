# Regenerates the five figures whose original generators were lost:
#   FIG_adaptation, FIG_crystallization, SUPP_1d_shifts, SUPP_crystallization_heatmap, SUPP_highstat_planes
# All derive from the released structural corpus + adaptation_deltas.csv. Run from project root.
import pandas as pd, numpy as np, re, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
COL={"film":"#1f77b4","book":"#d62728","tv":"#2ca02c"}
def load(m,idx):  # released 1890-2025 corpus (matches reproduce.py); natural decade bins
    d=pd.read_csv(f"data/corpus/{m}_structural_1890_2025.csv").rename(columns={idx:"id"})
    return d[d.year<=2025].assign(dec=lambda x:(x.year//10*10).astype(int),medium=m)
F,B,T=load("film","film_idx"),load("book","book_idx"),load("tv","tv_idx")
ATTRS=[c for c in F.columns if c not in ("id","title","year","dec","medium")]
# clean short display name per raw survey column
def nm(c):
    c=c.lower()
    for key,lab in [("science_fictional","sci-fi"),("fantastical","fantastical"),("realistic_did_you_find_the_dialogue","dlg-real"),
        ("realistic_was_the_world","realistic"),("world_building","world-build"),("major_settings","# settings"),
        ("many_protagonists","# protag"),("named_side","# side chars"),("plot_driven","plot-driven"),("character_driven","char.-driven"),("immersive","immersion"),
        ("unresolved","resolution"),("unsurprising","surprise"),("confusing","clarity"),("convincing","plot conv."),
        ("8a_how_likable","likability"),("competent","competence"),("proactive","proactiveness"),("real_did_this_protagonist","protag. realness"),
        ("emotionally_invested","emotional investment"),("relatable","relatability"),("quality_of_the_character","char.-writing quality"),
        ("quality_of_the_setting","setting quality"),("evocative","visual evocativeness"),("interesting_did_you_find_the_visual","visual interest"),
        ("engaging_did_you_find_the_dialogue","dialogue engagement"),("now_let_s_talk","dialogue show-vs-tell"),("moved_by_this","emotional impact"),
        ("like_the_score","score"),("relevant_are_these_aspect","identity relevance"),("pace","pace")]:
        if key in c: return lab
    return re.sub(r'^\d+[a-z]?_','',c).replace('_',' ')[:18]
NAMES={c:nm(c) for c in ATTRS}
win=lambda df: df[(df.year>=1915)&(df.year<=2020)]   # standardize on the analysis window
allc=pd.concat([win(F),win(B),win(T)]); am={a:allc[a].mean() for a in ATTRS}; asd={a:(allc[a].std() or 1) for a in ATTRS}
def z(df,a): return (df[a]-am[a])/asd[a]

# ---------------- FIG_crystallization (line) + SUPP_crystallization_heatmap ----------------
def mean_abs_corr(sub):  # matches reproduce.py crys(): drop all-NaN cols, mean |r| off-diagonal
    C=sub[ATTRS].dropna(axis=1,how="all").astype(float).corr().abs().values
    iu=np.triu_indices(C.shape[0],1); return np.nanmean(C[iu])
rows={m:[] for m in ["film","book","tv"]}
for m,df in [("film",F),("book",B),("tv",T)]:
    for dec,g in df.groupby("dec"):
        if dec>=1910 and len(g)>=40: rows[m].append((dec,mean_abs_corr(g)))
# combined MAIN exhibit: line (a) + the two correlation matrices themselves (b, c)
def _corr_ordered(sub):
    C=sub[ATTRS].astype(float).corr(); order=C.abs().mean().sort_values(ascending=False).index.tolist(); return C.loc[order,order],order
_early=F[(F.year>=1915)&(F.year<=1945)]; _recent=F[(F.year>=1980)&(F.year<=2010)]
_Ce,_order=_corr_ordered(_early); _Cr=_recent[ATTRS].astype(float).corr().loc[_order,_order]
_labs=[NAMES.get(c,c) for c in _order]
_iu=np.triu_indices(len(_order),1)
_me=np.nanmean(np.abs(_Ce.values[_iu])); _mr=np.nanmean(np.abs(_Cr.values[_iu]))
fig=plt.figure(figsize=(12,11.6))
gs=fig.add_gridspec(2,2,height_ratios=[0.66,1.0],hspace=0.34,wspace=0.10)
axl=fig.add_subplot(gs[0,:])
for m in ["film","book","tv"]:
    xy=np.array(rows[m]); axl.plot(xy[:,0],xy[:,1],"-o",color=COL[m],lw=2.0,ms=5,label={"film":"film","book":"the novel","tv":"television"}[m])
axl.set_xlabel("decade"); axl.set_ylabel("mean |correlation|\namong attributes")
axl.legend(frameon=False,loc="upper left"); axl.spines[["top","right"]].set_visible(False)
axl.set_title("a   Film's attributes grow more tightly correlated over the century",fontsize=13,fontweight="bold",loc="left")
axb=fig.add_subplot(gs[1,0]); axc=fig.add_subplot(gs[1,1])
for ax,Cm,ttl,mv,ylab in [(axb,_Ce,"b   Film 1915–1945",_me,True),(axc,_Cr,"c   Film 1980–2010",_mr,False)]:
    im=ax.imshow(Cm.values,cmap="RdBu_r",vmin=-0.6,vmax=0.6,aspect="equal")
    ax.set_xticks(range(len(_order))); ax.set_xticklabels(_labs,rotation=90,fontsize=6.5)
    ax.set_yticks(range(len(_order)))
    ax.set_yticklabels(_labs if ylab else [],fontsize=6.5)
    ax.set_title(f"{ttl}   (mean |r| = {mv:.2f})",fontsize=12,fontweight="bold",loc="left")
cb=fig.colorbar(im,ax=[axb,axc],fraction=0.022,pad=0.02); cb.set_label("attribute correlation")
fig.suptitle("The crystallization of film",fontsize=15,fontweight="bold",y=0.955)
fig.savefig("results/figures/FIG_crystallization.png",dpi=150,bbox_inches="tight"); plt.close(fig)
fcr=dict(rows["film"]); print(f"crystallization film: {min(fcr):.0f}s={fcr[min(fcr)]:.2f} -> 1980s={fcr.get(1980,np.nan):.2f}")

# heatmap: film early vs recent, attrs ordered by mean|r| (descending), with clean names
def corr_ordered(sub):
    C=sub[ATTRS].astype(float).corr(); order=C.abs().mean().sort_values(ascending=False).index.tolist(); return C.loc[order,order],order
early=F[(F.year>=1915)&(F.year<=1945)]; recent=F[(F.year>=1980)&(F.year<=2010)]
Ce,order=corr_ordered(early); Cr=recent[ATTRS].astype(float).corr().loc[order,order]
labs=[NAMES[c] for c in order]
fig,axs=plt.subplots(1,2,figsize=(15,7.2))
for ax,Cm,ttl in [(axs[0],Ce,f"Film 1915–1945  (mean |r| = {np.nanmean(np.abs(Ce.values[np.triu_indices(len(order),1)])):.2f})"),
                   (axs[1],Cr,f"Film 1980–2010  (mean |r| = {np.nanmean(np.abs(Cr.values[np.triu_indices(len(order),1)])):.2f})")]:
    im=ax.imshow(Cm.values,cmap="RdBu_r",vmin=-0.6,vmax=0.6)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labs,rotation=90,fontsize=6.5)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(labs,fontsize=6.5); ax.set_title(ttl,fontsize=12)
fig.suptitle("Crystallization, shown as the correlation structure itself",fontsize=14,y=1.0)
cb=fig.colorbar(im,ax=axs,fraction=0.025,pad=0.02); cb.set_label("attribute correlation")
fig.savefig("results/figures/SUPP_crystallization_heatmap.png",dpi=150,bbox_inches="tight"); plt.close(fig)

# ---------------- SUPP_1d_shifts ----------------
def shift(df,a):
    e=df[df.dec==1950][a].mean(); l=df[df.dec==2010][a].mean(); return (l-e)/asd[a]
S=pd.DataFrame({m:{a:shift(df,a) for a in ATTRS} for m,df in [("film",F),("book",B),("tv",T)]})
S=S.reindex(S["film"].sort_values().index); yb=np.arange(len(S))
fig,ax=plt.subplots(figsize=(9,10))
for m in ["film","book","tv"]:
    ax.scatter(S[m].values,yb,color=COL[m],s=34,label={"film":"Film","book":"Novel","tv":"Television"}[m],zorder=3)
ax.axvline(0,color="#888",lw=1); ax.set_yticks(yb); ax.set_yticklabels([NAMES[a] for a in S.index],fontsize=8)
ax.set_xlabel("standardized shift, 1950s → 2010s (SD units)"); ax.set_ylim(-1,len(S))
ax.set_title("Per-attribute shift over the second half of the century"); ax.legend(frameon=False,loc="lower right")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig("results/figures/SUPP_1d_shifts.png",dpi=150,bbox_inches="tight"); plt.close(fig)

# ---------------- SUPP_highstat_planes ----------------
def comp(df,keys): cs=[c for c in ATTRS if any(k in c.lower() for k in keys)]; return np.mean([z(df,c) for c in cs],axis=0)
PLANES=[("immersion","dialogue engagement",["immersive"],["engaging_did_you_find_the_dialogue"],"most separates the media"),
        ("plot resolution","surprise",["unresolved"],["unsurprising"],"moves most over the century")]
DECS=[1930,1970,2010]; HR=3.6; DR=2.8
def H(sub,xc,yc):
    rj=np.random.RandomState(0); jx=xc+rj.uniform(-.4,.4,len(sub)); jy=yc+rj.uniform(-.4,.4,len(sub))
    h,_,_=np.histogram2d(jx,jy,bins=36,range=[[-HR,HR],[-HR,HR]]); return gaussian_filter(h.T,1.3)
fig,axes=plt.subplots(2,3,figsize=(9,6.4))
for r,(xl,yl,xk,yk,why) in enumerate(PLANES):
    for c,dec in enumerate(DECS):
        ax=axes[r,c]
        for m,df in [("film",F),("book",B),("tv",T)]:
            sub=df[df.dec==dec]
            if len(sub)>=40:
                h=H(sub,comp(sub,xk),comp(sub,yk)); mx=h.max()
                if mx>0: ax.contour(h,levels=np.linspace(0.3*mx,0.9*mx,3),extent=[-HR,HR,-HR,HR],origin="lower",colors=COL[m],linewidths=1.1,alpha=.9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-DR,DR); ax.set_ylim(-DR,DR)
        if r==0: ax.set_title(f"{dec}s",fontweight="bold")
        if c==0: ax.set_ylabel(f"{xl} × {yl}\n({why})",fontsize=8)
from matplotlib.lines import Line2D
fig.legend([Line2D([0],[0],color=COL[m],lw=2) for m in ["film","book","tv"]],["film","the novel","television"],
           loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(0.5,1.04))
fig.suptitle("Density on the two empirically-strongest planes",y=1.08,fontweight="bold")
fig.tight_layout(); fig.savefig("results/figures/SUPP_highstat_planes.png",dpi=150,bbox_inches="tight"); plt.close(fig)

# ---------------- FIG_adaptation (from adaptation_deltas.csv) ----------------
AD=pd.read_csv("results/layers/structure/adaptation_deltas.csv").sort_values("film_minus_novel_SD")
agree=AD.agrees_sign.values
fig,ax=plt.subplots(figsize=(9,5.6)); yb=np.arange(len(AD))
ax.barh(yb,AD.film_minus_novel_SD,color=["#1f77b4" if a else "#9aa0a6" for a in agree],alpha=.9,zorder=2)
ax.errorbar(AD.film_minus_novel_SD,yb,xerr=AD.se,fmt="none",ecolor="k",elinewidth=.8,capsize=2,zorder=3)
ax.scatter(AD.cross_section_SD,yb,marker="D",s=48,color="#e6820a",zorder=4,label="cross-sectional")
ax.axvline(0,color="k",lw=.6); ax.set_yticks(yb); ax.set_yticklabels(AD.attribute,fontsize=9)
ax.set_xlabel("film $-$ novel (standardized)"); ax.set_title(f"The same story, told as a novel and as a film ({int(AD.n_pairs.iloc[0])} adaptations)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#1f77b4",label="within-pair, agrees in sign"),Patch(color="#9aa0a6",label="within-pair, disagrees"),
                   Line2D([0],[0],marker="D",color="w",markerfacecolor="#e6820a",markersize=8,label="cross-sectional")],
          fontsize=8,loc="lower right")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig("results/figures/FIG_adaptation.png",dpi=150,bbox_inches="tight"); plt.close(fig)
print("saved: FIG_crystallization, SUPP_crystallization_heatmap, SUPP_1d_shifts, SUPP_highstat_planes, FIG_adaptation")
