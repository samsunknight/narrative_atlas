import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
def load(f,idx):
    d=pd.read_csv(f"data/corpus/{f}_structural_1890_2025.csv").rename(columns={idx:"id"}); d["medium"]=f
    return d[(d.year>=1915)&(d.year<=2020)]
F,B,T=load("film","film_idx"),load("book","book_idx"),load("tv","tv_idx")
ATTRS=[c for c in F.columns if c not in("id","title","year","medium")]
ALLc=pd.concat([F,B,T]); am={a:ALLc[a].mean() for a in ATTRS}; asd={a:ALLc[a].std() for a in ATTRS}
def comp(d,keys): cs=[c for c in ATTRS if any(k in c for k in keys)]; return np.mean([(d[c]-am[c])/asd[c] for c in cs],axis=0)
PAIRS=[("relatability_resolution",["relatable"],["unresolved"],"distant → relatable","unresolved → resolved"),
 ("fantastical_proactiveness",["fantastical"],["proactive"],"grounded → fantastical","passive → proactive"),
 ("protagonists_surprise",["how_many_protagonists"],["unsurprising"],"single lead → large ensemble","predictable → surprising"),
 ("emotional_visual",["emotionally_invested","moved"],["interesting_did_you_find_the_visual","evocative"],"detached → emotionally involving","plain → visually rich"),
 ("warmth_scale",["likable","relatable","real_did_this"],["how_many_major_settings","how_many_protagonists","named_side"],"cold → warm characters","intimate → epic"),
 ("speculative_agentic",["science_fictional","fantastical","world_building"],["competent","proactive"],"grounded → speculative","passive → agentic")]
# the two empirically-strongest planes (shown in the main grid only, not the SI density family)
EXTRA=[("immersion_dialogue",["immersive"],["how_engaging"],"shallow → immersive","flat → engaging dialogue"),
       ("resolution_surprise",["unresolved"],["unsurprising"],"unresolved → resolved","predictable → surprising")]
for d in (F,B,T):
    for name,xk,yk,xl,yl in PAIRS+EXTRA:
        d[name+"_x"]=comp(d,xk); d[name+"_y"]=comp(d,yk)
DRNG=[-2.8,2.8]   # display range: gives each medium's cloud margin so contours are not clipped at the frame
HRNG=[-3.6,3.6]   # histogram/contour grid, padded beyond the display so contours close inside the grid
RNG=HRNG          # all contour extents reference the padded grid
def Hraw(sub,xc,yc):
    rj=np.random.RandomState(0)  # jitter dissolves the discrete 1-7 lattice on single-attribute planes
    jx=sub[xc].values+rj.uniform(-0.4,0.4,len(sub)); jy=sub[yc].values+rj.uniform(-0.4,0.4,len(sub))
    H,_,_=np.histogram2d(jx,jy,bins=36,range=[HRNG,HRNG]); return gaussian_filter(H.T,1.3)
def Hnorm(sub,xc,yc):
    H=Hraw(sub,xc,yc); return H/H.sum() if H.sum()>0 else H
mediaC=[("film",F,"Blues"),("book",B,"Reds"),("tv",T,"Greens")]
def early(d):
    g=d.groupby((d.year//10)*10).size(); return int(min(dec for dec,n in g.items() if n>=40))
DECS=[1930,1960,1990,2010]
for name,xk,yk,xl,yl in PAIRS:
    xc,yc=name+"_x",name+"_y"
    # ---- LEVELS grid: decade rows x medium cols ----
    fig,axes=plt.subplots(len(DECS),3,figsize=(8.8,2.5*len(DECS)))
    for r,dec in enumerate(DECS):
        for c,(mn,d,cm) in enumerate(mediaC):
            ax=axes[r,c]; sub=d[(d.year//10)*10==dec]
            if len(sub)>=10:
                H=Hraw(sub,xc,yc); ax.contourf(H,levels=10,extent=[*RNG,*RNG],origin="lower",cmap=cm,vmin=0,vmax=H.max())
            else:
                ax.text(0.5,0.5,"sparse",ha="center",va="center",transform=ax.transAxes,color="gray")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(DRNG); ax.set_ylim(DRNG)
            if r==0: ax.set_title(mn,fontweight="bold")
            if c==0: ax.set_ylabel(f"{dec}s",fontweight="bold",fontsize=11)
    fig.suptitle(f"Style-space density by decade and medium — {xl} (x), {yl} (y)",fontweight="bold",y=1.0)
    plt.tight_layout(); plt.savefig(f"outputs/figures/LEVELS_{name}.png",dpi=140,bbox_inches="tight"); plt.close()
    # ---- DIFF map: early -> 2010s, per medium ----
    fig,axes=plt.subplots(1,3,figsize=(15.5,5.4))
    for ax,(mn,d,cm) in zip(axes,mediaC):
        e=1950; se=d[(d.year//10)*10==e]; sl=d[(d.year//10)*10==2010]
        diff=Hnorm(sl,xc,yc)-Hnorm(se,xc,yc); v=np.abs(diff).max()
        ax.contourf(diff,levels=np.linspace(-v,v,12),extent=[*RNG,*RNG],origin="lower",cmap="RdBu_r",extend="both")
        ex,ey=se[xc].mean(),se[yc].mean(); lx,ly=sl[xc].mean(),sl[yc].mean()
        ax.scatter([ex],[ey],marker="o",s=55,facecolors="none",edgecolors="k",linewidths=1.8,zorder=5)
        ax.scatter([lx],[ly],marker="o",s=55,c="k",zorder=5)
        ax.annotate("",xy=(lx,ly),xytext=(ex,ey),arrowprops=dict(arrowstyle="-|>",lw=2.2,color="k"),zorder=4)
        ax.set_title(f"{mn}  ({e}s → 2010s)",fontweight="bold"); ax.set_xlabel(xl)
        if mn=="film": ax.set_ylabel(yl)
        ax.axhline(0,color="gray",lw=.5,ls=":"); ax.axvline(0,color="gray",lw=.5,ls=":"); ax.set_xlim(DRNG); ax.set_ylim(DRNG)
    fig.suptitle(f"Where works moved: density change, 1950s → 2010s — {xl} (x) vs {yl} (y)\n(red = gained, blue = lost; hollow = early centroid, filled = 2010s, arrow = drift)",fontweight="bold",y=1.03)
    plt.tight_layout(); plt.savefig(f"outputs/figures/DIFF_{name}.png",dpi=140,bbox_inches="tight"); plt.close()
# ---- MAIN lite figure: multi-plane contour grid, fewer decades, media overlaid ----
LBL={name:(xl,yl) for name,xk,yk,xl,yl in PAIRS+EXTRA}
PTITLE={"speculative_agentic":"speculative × agentic","emotional_visual":"emotional × visual","warmth_scale":"character warmth × scale","immersion_dialogue":"immersion × dialogue","resolution_surprise":"plot resolution × surprise"}
MP=["speculative_agentic","emotional_visual","warmth_scale","immersion_dialogue","resolution_surprise"]; MD=[1930,1970,2010]
# short axis end-labels per row so the plane is readable without the caption
AXLBL={"speculative_agentic":("grounded → science-fictional/fantastical","passive → proactive"),
       "emotional_visual":("detached → involving","plain → lavish"),
       "warmth_scale":("cold → warm","intimate → epic"),
       "immersion_dialogue":("shallow → immersive","flat → engaging dialogue"),
       "resolution_surprise":("unresolved → resolved","predictable → surprising")}
mc={"film":"#1f77b4","book":"#d62728","tv":"#2ca02c"}
fillcm={"film":"Blues","book":"Reds","tv":"Greens"}
figM,axesM=plt.subplots(len(MP),len(MD),figsize=(2.9*len(MD),2.9*len(MP)))
for r,name in enumerate(MP):
    xc,yc=name+"_x",name+"_y"; xl,yl=LBL[name]
    for c,dec in enumerate(MD):
        ax=axesM[r,c]
        for mn,d,_ in mediaC:
            sub=d[(d.year//10)*10==dec]
            if len(sub)>=40:
                H=Hraw(sub,xc,yc); mx=H.max()
                if mx>0:
                    ax.contourf(H,levels=np.linspace(0.5*mx,mx,4),extent=[*RNG,*RNG],origin="lower",cmap=fillcm[mn],alpha=0.22,zorder=1)
                    ax.contour(H,levels=np.linspace(0.3*mx,0.9*mx,3),extent=[*RNG,*RNG],origin="lower",colors=mc[mn],linewidths=1.2,alpha=0.9,zorder=3)
                ax.scatter([sub[xc].mean()],[sub[yc].mean()],s=28,c=mc[mn],edgecolor="white",linewidths=0.7,zorder=6)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(DRNG); ax.set_ylim(DRNG)
        ax.axhline(0,color="0.85",lw=.5); ax.axvline(0,color="0.85",lw=.5)
        if r==0: ax.set_title(f"{dec}s",fontweight="bold",fontsize=12)
        if c==0:
            ax.set_ylabel(f"{PTITLE[name]}",fontsize=9,fontweight="bold")
            axx,axy=AXLBL[name]
            ax.set_xlabel(axx,fontsize=7.5,color="0.45",style="italic")
            ax.text(-0.12,0.5,axy,transform=ax.transAxes,rotation=90,ha="center",va="center",fontsize=7.5,color="0.45",style="italic")
from matplotlib.lines import Line2D
figM.legend([Line2D([0],[0],color=mc[m],lw=2) for m in ["film","book","tv"]],["film","novel","television"],loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(0.5,1.03),fontsize=11)
figM.suptitle("Narrative style by decade: where each medium sits, and how it drifts",fontweight="bold",y=1.07)
plt.tight_layout(); plt.savefig("outputs/figures/MAIN_density_grid.png",dpi=150,bbox_inches="tight"); plt.close()
print("saved MAIN_density_grid.png")
print("saved LEVELS_* and DIFF_* for:", [p[0] for p in PAIRS])
