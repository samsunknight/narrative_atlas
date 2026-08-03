# Rebuilds certified FIG_F2_validation.png. Counts match Table 1 (161 scored / 150 validated; structure 52/48).
# Panel a is the build-and-release schematic (benchmark -> per-work ruler -> validated instrument -> atlas);
# panels b-d are the human validation itself, so the one figure carries both the pipeline and the evidence for it.
import pandas as pd, numpy as np, re, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False})
NAVY="#1f3b57"; GREY="#c7ccd1"; ORANGE="#c0603a"; TEAL="#3f7d74"; BAR_BG="#eceef0"; THR="#c0603a"
C_HUMAN="#c0392b"; C_INSTR="#7d3c98"; C_ATLAS="#2471a3"; C_BG="#f4f1ea"
R="."
import os
# Read the frozen 161-attribute instrument. In the working tree data/validation/ holds a 195-row
# exploratory superset, so prefer the canonical package copy when it is present; inside the shipped
# package that path does not exist and the local data/validation/ copy is already the 161-row instrument.
_DICT=f"{R}/data/validation/attribute_dictionary.csv"
_CANON=f"{R}/rep_build/narrative_atlas/data/validation/attribute_dictionary.csv"
if os.path.exists(_CANON): _DICT=_CANON
d=pd.read_csv(_DICT)
d['fr']=pd.to_numeric(d['film_r'],errors='coerce'); d['brk']=pd.to_numeric(d['book_r'],errors='coerce')
def nice(a): return re.sub(r'^\d+[a-z]?_','',str(a)).replace('_',' ').strip()

fig=plt.figure(figsize=(15.5,18))
gs=fig.add_gridspec(3,3,width_ratios=[0.72,0.72,1.05],height_ratios=[0.42,1,1],wspace=0.5,hspace=0.30)
fig.suptitle("A human-validated measurement system for narrative form",fontsize=18,fontweight="bold",y=0.995)

# ---- panel a: the build-and-release pipeline ----
axp=fig.add_subplot(gs[0,:]); axp.set_xlim(0,13); axp.set_ylim(0.55,3.85); axp.axis("off")
stages=[(0.2,C_HUMAN,"HUMAN BENCHMARK","714 readers + 225 viewers\nassigned complete novels\nand films, rated on\n>140 narrative attributes","HumanReader + HumanViewer\n(aggregate public)"),
        (3.4,C_HUMAN,"PER-WORK JUDGMENTS","work-mean rating +\nrater agreement +\nreliability ceiling\nfor every attribute","the human ruler"),
        (6.6,C_INSTR,"VALIDATED INSTRUMENT","LLM answers the same\nquestions from a plot\nsummary; kept only where\nit matches human means","150 of >160 attributes\nvalidate, tiered"),
        (9.8,C_ATLAS,"NARRATIVE ATLAS","the instrument applied\nat scale across film,\nthe novel, and television,\n1890–2025","149,341 works on one\ncommon scale")]
bw,bh,by=2.7,2.2,1.3
for i,(x,c,title,body,foot) in enumerate(stages):
    axp.add_patch(FancyBboxPatch((x,by),bw,bh,boxstyle="round,pad=0.05,rounding_size=0.12",lw=1.6,edgecolor=c,facecolor=C_BG,mutation_aspect=1))
    axp.text(x+bw/2,by+bh-0.30,title,ha="center",va="center",fontsize=11.5,fontweight="bold",color=c)
    axp.text(x+bw/2,by+bh/2-0.15,body,ha="center",va="center",fontsize=10,color="#222")
    axp.text(x+bw/2,by-0.40,foot,ha="center",va="center",fontsize=9.6,style="italic",color=c)
    if i<len(stages)-1:
        axp.add_patch(FancyArrowPatch((x+bw+0.04,by+bh/2),(x+bw+0.46,by+bh/2),arrowstyle="-|>",mutation_scale=22,lw=2,color="#555"))
axp.set_title("a   From human benchmark to atlas, built as one stack",fontsize=14.5,fontweight="bold",loc="left")

# ---- panel b: structure per-attribute film validation, split into two columns (validated = tier A/B, matches Table 1's 42/46) ----
axa1=fig.add_subplot(gs[1:,0]); axa2=fig.add_subplot(gs[1:,1])
st=d[d['layer']=='structure'].dropna(subset=['fr']).sort_values('fr',ascending=False).copy()
st['val']=st['tier'].isin(['A','B'])
half=(len(st)+1)//2
for ax,sub in [(axa1,st.iloc[:half]),(axa2,st.iloc[half:])]:
    yy=np.arange(len(sub))[::-1]   # highest r at top of each column
    for yi,fr,val in zip(yy,sub['fr'],sub['val']):
        c=NAVY if val else GREY
        ax.plot([0,fr],[yi,yi],color=c,lw=1.05,alpha=.5,zorder=1); ax.scatter(fr,yi,s=28,color=c,zorder=3)
    ax.axvline(0.22,color=THR,ls="--",lw=1.4)
    ax.set_yticks(yy); ax.set_yticklabels(sub['attribute'].map(nice),fontsize=11.5)
    ax.set_ylim(-1,len(sub)); ax.set_xlim(0,0.75)
    ax.set_xlabel("Validation $r$",fontsize=11)
axa1.text(0.235,half-1.2,"Validated\n$r\\geq0.22$",color=THR,fontsize=10,va="top")
axa1.set_title("b  Structure layer — per-attribute film validation",fontsize=14.5,fontweight="bold",loc="left")
fig.text(0.295,0.045,f"{int(st['val'].sum())} of {len(st)} structural attributes validate against human ratings"
         "   ·   validation $r$ = LLM score vs. 225-viewer human mean, zero-shot",ha="center",fontsize=11,color="#555")

# ---- panel c: per-layer validate bars (Table 1 deployed counts) ----
axb=fig.add_subplot(gs[1,2])
LAY=[("Structure & plot",48,52,NAVY,"median $r$ 0.35  (top 0.78)"),
     ("Setting",2,2,TEAL,"$r$ 0.71 / 0.57"),
     ("Story shape",5,5,NAVY,"Vonnegut arcs 0.23–0.34"),
     ("Conflict type",4,6,ORANGE,"vs. nature 0.61"),
     ("Character arc",9,9,TEAL,"arc-change $r$ 0.45–0.54"),
     ("Narration",1,1,TEAL,"# narrators 0.26 (book)"),
     ("Mood",28,31,NAVY,"median $r$ 0.41"),
     ("Genre",18,18,ORANGE,"median AUC 0.91"),
     ("Texture",35,37,NAVY,"median $r$ 0.40  (visual 0.44)")]
yb=np.arange(len(LAY))[::-1]
for yi,(lab,v,tot,col,note) in zip(yb,LAY):
    f=100*v/tot
    axb.barh(yi,100,color=BAR_BG,height=0.62,zorder=1); axb.barh(yi,f,color=col,height=0.62,zorder=2)
    axb.text(f-2,yi,f"{v}/{tot}",color="white",ha="right",va="center",fontweight="bold",fontsize=12.5,zorder=3)
    axb.text(103,yi,note,va="center",fontsize=11,color="#333")
axb.set_yticks(yb); axb.set_yticklabels([l for l,*_ in LAY],fontsize=11.5)
axb.set_xlim(0,100); axb.set_xticks([0,25,50,75,100]); axb.set_xticklabels(["0","25","50","75","100%"])
axb.set_xlabel("Attributes that validate against human ground truth",fontsize=11.5)
axb.set_title("c   Every construct validates",fontsize=14.5,fontweight="bold",loc="left"); axb.set_ylim(-0.6,len(LAY)-0.4)

# ---- panel d: cross-medium scatter. The 8 come from the dictionary (guaranteed); greys from the film/book validation join ----
axc=fig.add_subplot(gs[2,2])
def nq(q): q=re.sub(r'\b(movie|movies|film|book|books|novel|story)\b','',str(q).lower()); return re.sub(r'[^a-z]','',q)[:40]
mv=pd.read_csv(f"{R}/data/validation/movie_attribute_validation.csv"); bk=pd.read_csv(f"{R}/data/validation/book_attribute_validation.csv")
mv['k']=mv['question'].map(nq); mv['fr']=np.sqrt(mv['r2'].clip(lower=0)); bk['k']=bk['question'].map(nq); bk['brk']=np.sqrt(bk['r2'].clip(lower=0))
J=mv.merge(bk[['k','brk']],on='k',how='inner').drop_duplicates('k')
eight=d[d['cross_medium']==True][['attribute','fr','brk']].copy()
LABMAP={'sci-fi world':'sci-fi','fantastical world':'fantastical','realistic world':'realistic world',
  '# protagonists':'# protagonists','world-building relevance':'world-building','protagonist competent':'competence',
  'protagonist relatable':'relatability','protagonist proactive':'proactiveness',
  'plot-driven':'plot-driven','character-driven':'character-driven',
  'Character development':'internal change','Opening hook':'opening hook','Time linearity':'time linearity',
  'Plot linearity':'plot linearity','Ending reversal (peripeteia)':'peripeteia'}
eight['lab']=eight['attribute'].map(nice).map(LABMAP)
eight=eight[eight["lab"].notna()].copy()  # keep the labelled structural cross-medium set for the scatter
# offsets so labels never touch a marker (esp. the relatab/compet/proact cluster)
OFF={'sci-fi':(8,-1,'left','center'),'fantastical':(8,4,'left','bottom'),'realistic world':(8,7,'left','bottom'),
     '# protagonists':(9,7,'left','bottom'),'world-building':(2,-13,'center','top'),'competence':(-10,3,'right','center'),
     'relatability':(-9,-9,'right','top'),'proactiveness':(-11,9,'right','bottom'),
     'plot-driven':(-8,-2,'right','top'),'character-driven':(9,-8,'left','top'),
     'internal change':(9,-7,'left','top'),'opening hook':(0,9,'center','bottom'),'time linearity':(-9,8,'right','bottom'),
     'plot linearity':(2,9,'center','bottom'),'peripeteia':(9,-6,'left','top')}
axc.axhline(0.22,color=THR,ls="--",lw=1.0,alpha=.8); axc.axvline(0.22,color=THR,ls="--",lw=1.0,alpha=.8)
axc.plot([0,0.85],[0,0.85],color="#bbb",ls="--",lw=0.8,zorder=0)
eightk=set(nq(x) for x in ['science fictional','fantastical','realistic','protagonists','world building','competent','relatable','proactive'])
greys=J[~J['k'].apply(lambda k:any(e in k or k in e for e in eightk))]
axc.scatter(greys['fr'],greys['brk'],s=42,color=GREY,zorder=2,label="film-only")
axc.scatter(eight['fr'],eight['brk'],s=72,color=NAVY,zorder=3,label="validate in both media")
for _,r in eight.iterrows():
    lab=r['lab']; dx,dy,ha,va=OFF[lab]
    axc.annotate(lab,(r['fr'],r['brk']),textcoords="offset points",xytext=(dx,dy),ha=ha,va=va,fontsize=9.5,color=NAVY)
axc.set_xlim(0,0.85); axc.set_ylim(0,0.85)
axc.set_xlabel("Film validation $r$",fontsize=11.5); axc.set_ylabel("Book validation $r$",fontsize=11.5)
axc.set_title("d   Cross-medium replication (core structure)",fontsize=14.5,fontweight="bold",loc="left")
axc.legend(loc="lower right",fontsize=10.5,frameon=False)

fig.savefig(f"{R}/results/figures_certified/FIG_F2_validation.png",dpi=200,bbox_inches="tight"); plt.close(fig)
print(f"F2 saved | panel a {int(st['val'].sum())}/{len(st)} | panel c {len(eight)} cross-medium + {len(greys)} film-only")
