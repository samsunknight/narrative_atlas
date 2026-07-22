import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.decomposition import PCA; from sklearn.preprocessing import StandardScaler
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.8})
def load(f,idx):
    d=pd.read_csv(f"data/corpus/{f}_structural_1890_2025.csv").rename(columns={idx:"id"})
    return d[(d.year>=1915)&(d.year<=2020)]
F,B,T=load("film","film_idx"),load("book","book_idx"),load("tv","tv_idx")
ATTRS=[c for c in F.columns if c not in("id","title","year")]
def sn(c):
    for k,n in [("science_fictional","sci-fi"),("fantastical","fantastical"),("realistic_was_the_world","realistic world"),("world_building","world-building"),("how_many_protagonists","# protagonists"),("named_side","# side chars"),("how_many_major_settings","# settings"),("competent","competence"),("proactive","proactiveness"),("likable","likability"),("relatable","relatability"),("real_did_this","feels real"),("emotionally_invested","emot. investment"),("pace_of_the","pace"),("unresolved","resolution"),("unsurprising","surprise"),("confusing","clarity"),("convincing","plot convincing"),("evocative","visual evocative"),("interesting_did_you_find_the_visual","visual interest"),("engaging_did_you_find_the_dial","dialogue engaging"),("realistic_did_you_find_the_dial","dialogue realism"),("now_let_s_talk","show vs tell"),("showing_and","showing/telling"),("immersive","immersive"),("score","score"),("plot_driven","plot vs character"),("relevant_are_these","identity relev."),("quality_of_the_character","character quality"),("quality_of_the_setting","setting quality"),("moved","moved")]:
        if k in c: return n
    return c[:14]
C={"film":"#2471a3","book":"#c0392b","tv":"#27ae60","rat":"#2471a3","fash":"#c0392b"}

# ===== FIG1 validation =====
mv=pd.read_csv("data/validation/movie_attribute_validation.csv"); bv=pd.read_csv("data/validation/book_attribute_validation.csv")
mr={r.attribute:r.r2**0.5 for r in mv.itertuples()}; br={r.attribute:r.r2**0.5 for r in bv.itertuples()}
shared=[a for a in ATTRS if a in mr and a in br]
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5.5))
top=sorted([(sn(a),mr[a]) for a in mr if a in ATTRS],key=lambda z:z[1])[-14:]
a1.barh([n for n,_ in top],[v for _,v in top],color="#2471a3",alpha=.85)
a1.set_xlabel("validation r  (LLM vs human raters)"); a1.set_title("a   Film attributes, validated against 225 viewers",loc="left",fontweight="bold",fontsize=12)
a2.scatter([mr[a] for a in shared],[br[a] for a in shared],s=55,color="#7f8c8d",edgecolor="k",lw=.4)
for a in shared:
    if mr[a]>0.4 or br[a]>0.3: a2.annotate(sn(a),(mr[a],br[a]),fontsize=7.5,xytext=(3,2),textcoords="offset points")
a2.axhline(0.22,color="#c0392b",ls=":",lw=1); a2.axvline(0.22,color="#c0392b",ls=":",lw=1)
a2.set_xlabel("film validation r"); a2.set_ylabel("book validation r"); a2.set_title("b   Both-media validation (dotted = 0.22 bar)",loc="left",fontweight="bold",fontsize=12)
plt.tight_layout(); plt.savefig("results/figures/FIG1_validation.png",dpi=150,bbox_inches="tight"); print("FIG1")

# ===== FIG2 style-space =====
ALL=pd.concat([F.assign(medium="film"),B.assign(medium="book"),T.assign(medium="tv")],ignore_index=True).dropna(subset=ATTRS).reset_index(drop=True)  # complete-case, matches reproduce.py
Xs=((ALL[ATTRS]-ALL[ATTRS].mean())/ALL[ATTRS].std()).values; p=PCA(4).fit(Xs); pc=p.transform(Xs)
fig,(a1,a2,a3)=plt.subplots(1,3,figsize=(18.5,5.5))
for m in ["film","book","tv"]:
    mask=ALL.medium==m; a1.scatter(pc[mask,0][::8],pc[mask,1][::8],s=4,alpha=.15,color=C[m],label=m)
a1.set_xlabel(f"PC1 ({p.explained_variance_ratio_[0]:.0%}) — craft/quality"); a1.set_ylabel(f"PC2 ({p.explained_variance_ratio_[1]:.0%}) — speculative")
a1.legend(markerscale=3,fontsize=9); a1.set_title("a   The space of narrative style",loc="left",fontweight="bold",fontsize=12)
ld=sorted(zip([sn(a) for a in ATTRS],p.components_[1]),key=lambda z:z[1])
a2.barh([n for n,_ in ld[-6:]]+["…"]+[n for n,_ in ld[:6]],[v for _,v in ld[-6:]]+[0]+[v for _,v in ld[:6]],color=["#8e44ad"]*6+["w"]+["#16a085"]*6)
a2.set_xlabel("loading on PC2"); a2.set_title("b   The speculative axis (PC2 loadings)",loc="left",fontweight="bold",fontsize=12); a2.axvline(0,color="k",lw=.5)
# --- panel c: adaptation contrast (same source story, novel vs film) ---
ADLAB={"sci-fi":"sci-fi","fantastical":"fantastical","realistic":"realistic world","world-building":"world-building","#protagonists":"# protagonists","competence":"competence","proactiveness":"proactiveness","relatability":"relatability"}
ad=pd.read_csv("results/tables/adaptation_deltas.csv")
ad=ad[ad.attribute.isin(ADLAB)].copy(); ad["lab"]=ad.attribute.map(ADLAB)
ad=ad.sort_values("within_pair_delta").reset_index(drop=True); yc=np.arange(len(ad))
a3.barh(yc,ad.within_pair_delta,color=[C["film"] if v>0 else C["book"] for v in ad.within_pair_delta],alpha=.9)
a3.errorbar(ad.within_pair_delta,yc,xerr=1.96*ad.se,fmt="none",ecolor="#2c3e50",elinewidth=1.1,capsize=3,zorder=4)
a3.set_yticks(yc); a3.set_yticklabels(ad.lab,fontsize=9.5)
a3.axvline(0,color="k",lw=.6)
a3.set_xlabel("film − novel (SD)   ◄ lower in film | higher in film ►")
a3.set_title("c   The same source story, novel vs film",loc="left",fontweight="bold",fontsize=12)
plt.tight_layout(); plt.savefig("results/figures/FIG2_stylespace.png",dpi=150,bbox_inches="tight"); print("FIG2")

# ===== FIG3 FIXED (labels) =====
fdec=F.assign(dec=(F.year//10)*10).groupby("dec")[ATTRS].mean(); fdec=fdec[F.assign(dec=(F.year//10)*10).groupby("dec").size()>=30]
def z(s): return (s-s.mean())/s.std()
ac={a:np.corrcoef(z(fdec[a]).dropna().values[:-1],z(fdec[a]).dropna().values[1:])[0,1] for a in ATTRS if len(fdec[a].dropna())>6}
items=sorted(ac.items(),key=lambda kv:kv[1]); names=[sn(a) for a,_ in items]; vals=[v for _,v in items]
def cat(v): return C["fash"] if v<0.6 else ("#7f8c8d" if v<0.82 else C["rat"])
fig=plt.figure(figsize=(13,7)); gs=fig.add_gridspec(1,2,width_ratios=[1.25,1],wspace=0.5)
ax=fig.add_subplot(gs[0]); y=np.arange(len(names))
ax.hlines(y,0,vals,color=[cat(v) for v in vals],lw=2,alpha=.8); ax.scatter(vals,y,c=[cat(v) for v in vals],s=42,zorder=3)
ax.set_yticks(y); ax.set_yticklabels(names,fontsize=8.5)
ax.axvspan(-0.2,0.6,color=C["fash"],alpha=.05); ax.axvspan(0.82,1.02,color=C["rat"],alpha=.05)
ax.set_xlabel("temporal autocorrelation"); ax.set_title("a   The two clocks of style",loc="left",fontweight="bold",fontsize=13)
ax.text(0.91,len(names)-2,"RATCHET\n(one-way)",color=C["rat"],fontsize=10.5,ha="center",fontweight="bold")
ax.text(0.30,2,"FASHION\n(returns)",color=C["fash"],fontsize=10.5,ha="center",fontweight="bold"); ax.set_xlim(-0.05,1.02)
ax2=fig.add_subplot(gs[1]); yrs=fdec.index
for a,c,lab in [("science_fictional","#2471a3","sci-fi (ratchet)"),("immersive","#5dade2","immersive (ratchet)"),("unsurprising","#c0392b","surprise (fashion)"),("proactive","#e67e22","proactiveness (fashion)")]:
    col=next(x for x in ATTRS if a in x); ax2.plot(yrs,z(fdec[col]),color=c,lw=2.2,label=lab,marker="o",ms=3)
ax2.axhline(0,color="k",lw=.5,alpha=.3); ax2.legend(fontsize=8.5,frameon=False,loc="upper left")
ax2.set_xlabel("decade"); ax2.set_ylabel("standardized level"); ax2.set_title("b   Two shapes of change (film)",loc="left",fontweight="bold",fontsize=13)
plt.savefig("results/figures/FIG3_ratchet_fashion.png",dpi=150,bbox_inches="tight"); print("FIG3 fixed")

# ===== FIG6 media personalities + divergence =====
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5.5))
VAL=["sci-fi","fantastical","realistic world","world-building","# protagonists","competence","proactiveness"]
off=[]
for a in ATTRS:
    if sn(a) in VAL: off.append((sn(a),(F[a].mean()-B[a].mean())/pd.concat([F,B])[a].std()))
off.sort(key=lambda z:z[1])
a1.barh([n for n,_ in off],[v for _,v in off],color=["#c0392b" if v<0 else "#2471a3" for _,v in off])
a1.axvline(0,color="k",lw=.6); a1.set_xlabel("film − book (z)   ◄ books higher | films higher ►"); a1.set_title("a   Media personalities (divergence atlas)",loc="left",fontweight="bold",fontsize=12)
va=[a for a in ATTRS if sn(a) in VAL]; dists=[]
for dd in range(1920,2015,10):
    fc=F[(F.year>=dd)&(F.year<dd+20)][va].mean(); bc=B[(B.year>=dd)&(B.year<dd+20)][va].mean(); dists.append((dd,np.sqrt(((fc-bc)**2).sum())))
a2.plot([d for d,_ in dists],[v for _,v in dists],color="#34495e",lw=2.6,marker="o")
a2.set_xlabel("decade"); a2.set_ylabel("film–book stylistic distance"); a2.set_title("b   The media are diverging",loc="left",fontweight="bold",fontsize=12)
plt.tight_layout(); plt.savefig("results/figures/FIG6_media.png",dpi=150,bbox_inches="tight"); print("FIG6")
print("\nall main figures built:", __import__("os").listdir("results/figures"))
