# Rebuilds certified FIG_F2_validation.png (original generator lost). All counts match Table 1 (162/148).
import pandas as pd, numpy as np, re, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False})
NAVY="#1f3b57"; GREY="#c7ccd1"; ORANGE="#c0603a"; TEAL="#3f7d74"; BAR_BG="#eceef0"; THR="#c0603a"
R="."
d=pd.read_csv(f"{R}/data/validation/attribute_dictionary.csv")
d['fr']=pd.to_numeric(d['film_r'],errors='coerce'); d['brk']=pd.to_numeric(d['book_r'],errors='coerce')
def nice(a): return re.sub(r'^\d+[a-z]?_','',str(a)).replace('_',' ').strip()

fig=plt.figure(figsize=(19,12))
gs=fig.add_gridspec(2,2,width_ratios=[1.05,1.0],height_ratios=[1,1],wspace=0.30,hspace=0.34)
fig.suptitle("Human validation of the Narrative Atlas instrument across all five layers",fontsize=17,fontweight="bold",y=0.985)

# ---- panel a: structure per-attribute film validation (validated = tier A/B, matches Table 1's 59/67) ----
axa=fig.add_subplot(gs[:,0])
st=d[d['layer']=='structure'].dropna(subset=['fr']).sort_values('fr').copy()
st['val']=st['tier'].isin(['A','B'])
y=np.arange(len(st))
for yi,fr,val in zip(y,st['fr'],st['val']):
    c=NAVY if val else GREY
    axa.plot([0,fr],[yi,yi],color=c,lw=1.05,alpha=.5,zorder=1); axa.scatter(fr,yi,s=32,color=c,zorder=3)
axa.axvline(0.22,color=THR,ls="--",lw=1.4); axa.text(0.235,len(st)-1.5,"Validated\n$r\\geq0.22$",color=THR,fontsize=10,va="top")
axa.set_yticks(y); axa.set_yticklabels(st['attribute'].map(nice),fontsize=7.2)
axa.set_ylim(-1,len(st)); axa.set_xlim(0,0.75)
axa.set_xlabel("Validation $r$  (LLM score vs.\\ 225-viewer human mean, zero-shot)",fontsize=10)
axa.set_title("a   Structure layer — per-attribute film validation",fontsize=13,fontweight="bold",loc="left")
axa.text(0.5,-0.075,f"{int(st['val'].sum())} of {len(st)} structural attributes validate against human ratings",
         transform=axa.transAxes,ha="center",fontsize=9.5,color="#555")

# ---- panel b: per-layer validate bars (Table 1 deployed counts) ----
axb=fig.add_subplot(gs[0,1])
LAY=[("Structure\n(scalar attrs)",59,67,NAVY,"median $r$ 0.28  (top 0.70)"),
     ("Mood\n(31 tags)",28,31,NAVY,"median $r$ 0.39"),
     ("Genre\n(18 labels)",18,18,ORANGE,"median AUC 0.91"),
     ("Arc\n(9 arc cells)",9,9,TEAL,"arc-change $r$ 0.37–0.48"),
     ("Texture\n(descriptors)",34,37,NAVY,"median $r$ 0.35  (visual 0.42)")]
yb=np.arange(len(LAY))[::-1]
for yi,(lab,v,tot,col,note) in zip(yb,LAY):
    f=100*v/tot
    axb.barh(yi,100,color=BAR_BG,height=0.62,zorder=1); axb.barh(yi,f,color=col,height=0.62,zorder=2)
    axb.text(f-2,yi,f"{v}/{tot}",color="white",ha="right",va="center",fontweight="bold",fontsize=11,zorder=3)
    axb.text(103,yi,note,va="center",fontsize=9.5,color="#333")
axb.set_yticks(yb); axb.set_yticklabels([l for l,*_ in LAY],fontsize=10)
axb.set_xlim(0,100); axb.set_xticks([0,25,50,75,100]); axb.set_xticklabels(["0","25","50","75","100%"])
axb.set_xlabel("Attributes that validate against human ground truth",fontsize=10)
axb.set_title("b   All five layers validate",fontsize=13,fontweight="bold",loc="left"); axb.set_ylim(-0.6,len(LAY)-0.4)

# ---- panel c: cross-medium scatter. The 8 come from the dictionary (guaranteed); greys from the film/book validation join ----
axc=fig.add_subplot(gs[1,1])
def nq(q): q=re.sub(r'\b(movie|movies|film|book|books|novel|story)\b','',str(q).lower()); return re.sub(r'[^a-z]','',q)[:40]
mv=pd.read_csv(f"{R}/data/validation/movie_attribute_validation.csv"); bk=pd.read_csv(f"{R}/data/validation/book_attribute_validation.csv")
mv['k']=mv['question'].map(nq); mv['fr']=np.sqrt(mv['r2'].clip(lower=0)); bk['k']=bk['question'].map(nq); bk['brk']=np.sqrt(bk['r2'].clip(lower=0))
J=mv.merge(bk[['k','brk']],on='k',how='inner').drop_duplicates('k')
eight=d[d['cross_medium']==True][['attribute','fr','brk']].copy()
LABMAP={'sci-fi world':'sci-fi','fantastical world':'fantastical','realistic world':'realistic world',
  '# protagonists':'# protagonists','world-building relevance':'world-building','protagonist competent':'competence',
  'protagonist relatable':'relatability','protagonist proactive':'proactiveness'}
eight['lab']=eight['attribute'].map(nice).map(LABMAP)
assert eight['lab'].notna().all(), f"unmapped: {eight[eight['lab'].isna()]['attribute'].tolist()}"
# offsets so labels never touch a marker (esp. the relatab/compet/proact cluster)
OFF={'sci-fi':(8,-1,'left','center'),'fantastical':(8,4,'left','bottom'),'realistic world':(6,-11,'left','top'),
     '# protagonists':(8,2,'left','center'),'world-building':(6,-12,'left','top'),'competence':(10,9,'left','bottom'),
     'relatability':(-10,-10,'right','top'),'proactiveness':(-10,9,'right','bottom')}
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
axc.set_xlabel("Film validation $r$",fontsize=10); axc.set_ylabel("Book validation $r$",fontsize=10)
axc.set_title("c   Cross-medium replication (structure)",fontsize=13,fontweight="bold",loc="left")
axc.legend(loc="lower right",fontsize=9,frameon=False)

fig.savefig(f"{R}/outputs/figures/FIG_F2_validation.png",dpi=200,bbox_inches="tight"); plt.close(fig)
print(f"F2 saved | panel a {int(st['val'].sum())}/{len(st)} | panel c 8 cross-medium + {len(greys)} film-only")
