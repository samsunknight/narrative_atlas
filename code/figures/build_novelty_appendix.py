"""SI section S1.6, 'Narrative-form novelty and semantic novelty are distinct': the figure and
every number the subsection reports. Panel a plots each film's narrative-form novelty (from its
standardized atlas attributes) against its semantic novelty (from a plot-summary embedding), each
defined as one minus the mean similarity to the twenty nearest predecessors of the previous ten
release years. Panel b shows how much of each atlas layer a ridge probe recovers from the embedding.

The script also emits, to results/novelty_appendix_numbers.json, the full set of reported figures:
the primary correlation between the two novelties, its descriptive-basis variant (evaluative
attributes set aside), its calendar-decade-pool variant, the split-half reliability of the
narrative-form measure, and the per-layer decodability. Because it reads the raw Wikipedia plot text
and an external embedding model, this analysis is not part of the turnkey reproduction package (see
the SI footnote); this script is its committed source, so no reported number is an orphan.
"""
import os, re, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.stats import pearsonr; from sklearn.linear_model import Ridge; from sklearn.model_selection import KFold
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False})
BASE="/Users/samsunknight/Library/CloudStorage/Dropbox/University of Toronto/"; SE=BASE+"style_evolves/"; CP=BASE+"cultural_physics/"; NN=BASE+"narrative_novelty/"
SCR="/private/tmp/claude-501/-Users-samsunknight-Library-CloudStorage-Dropbox-University-of-Toronto-movies-taste-over-time/5f9ad73a-5a7c-4bea-ae04-478faf6b572e/scratchpad/"
PKG=os.path.dirname(os.path.abspath(__file__))                    # walk up to the package root
while os.path.dirname(PKG)!=PKG and not os.path.isdir(os.path.join(PKG,"data","atlas")): PKG=os.path.dirname(PKG)
def norm(s): return re.sub(r'[^a-z0-9]','',str(s).lower())

# The plot-summary embedding is the OpenAI text-embedding-3-small vector of each film's raw Wikipedia
# plot text for the 9,999-film sample. It is now SHIPPED with the package (data/novelty/), sourced from
# the sibling narrative_novelty project's cache (emb_film_sample_9999.npy) and row-aligned to `samp`
# below: reproducing the frozen SI numbers -- primary r=0.19 and the per-layer decodability (genre 0.67,
# mood 0.63, structure 0.47, texture 0.47, character-arc 0.30) -- to the reported precision confirms the
# alignment (a permuted embedding would drive decodability to ~0). No re-embedding / API call is made.
# The one remaining external input is the raw Wikipedia plot text (CP film_wiki_text.parquet), used only
# to reconstruct `samp`; it is not redistributed (see the SI footnote), so this script skips gracefully
# rather than hard-failing when that text is absent.
EMB_CANDIDATES=[os.path.join(PKG,"data","novelty","emb_plot_summary_9999.npy"),   # shipped with package
                NN+"data/emb_film_sample_9999.npy",                               # sibling-project cache
                SCR+"emb_9999.npy"]                                               # legacy scratch (wiped)
EMB_PATH=next((p for p in EMB_CANDIDATES if os.path.exists(p)), None)
_ext_ok = (os.path.exists(CP+"data/film_wiki_text.parquet")
           and os.path.exists(SE+"data/atlas/century_frame_film.parquet")
           and EMB_PATH is not None)
if not _ext_ok:
    print("SKIP novelty appendix (SI Fig 15 / §S1.6): the raw Wikipedia plot text needed to rebuild the "
          "film sample is not shipped (see the SI footnote). The plot-summary embedding itself IS shipped "
          "(data/novelty/); supply CP film_wiki_text.parquet to run this end to end.")
    raise SystemExit(0)

AF=pd.read_parquet(SE+"data/atlas/century_frame_film.parquet")
cb=pd.read_csv(SE+"data/validation/attribute_dictionary.csv"); LAY={r['column']:r['layer'] for _,r in cb.iterrows()}
attrs=[c for c in AF.columns if c not in ("idx","title","year","decade","medium") and AF[c].dtype!=object]
# The descriptive basis sets aside attributes that record an evaluative judgment of the work (its
# felt quality, how engaging or enjoyable it is) rather than a descriptive property of its form.
EVAL_KEYS=["quality","liking","liked","enjoy","moved","satisf","engaging","interesting","overall",
           "glad","compelling","evocative","feeling","relatab","convincing"]
desc_attrs=[a for a in attrs if not any(e in a.lower() for e in EVAL_KEYS)]

AF=AF.dropna(subset=["title","year"]); AF["k"]=AF.title.map(norm)+"|"+AF.year.astype(int).astype(str)
TX=pd.read_parquet(CP+"data/film_wiki_text.parquet").dropna(subset=["title","year","text"]); TX["k"]=TX.title.map(norm)+"|"+TX.year.astype(int).astype(str); TX=TX.drop_duplicates("k")
M=AF.merge(TX[["k","text"]],on="k",how="inner").dropna(subset=attrs+["text"]); M=M[(M.year>=1930)&(M.year<=2015)].copy(); M["dec"]=(M.year//10*10)
samp=M.groupby("dec",group_keys=False).apply(lambda g:g.sample(min(len(g),max(20,10000//M.dec.nunique())),random_state=0)).reset_index(drop=True)
yrs=samp.year.astype(int).values
E=np.load(EMB_PATH)
assert E.shape[0]==len(samp), f"embedding rows {E.shape[0]} != samp {len(samp)} (row alignment broken)"
print(f"loaded plot-summary embedding {E.shape} from {EMB_PATH}")
def std(cols): X=samp[cols].values.astype(float); return np.nan_to_num((X-np.nanmean(X,0))/(np.nanstd(X,0)+1e-9))
A_all, A_desc = std(attrs), std(desc_attrs)

def novelty(X, pool_fn, k=20):   # 1 - mean top-k cosine similarity to a film's predecessor pool
    Xn=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9); out=np.full(len(X),np.nan)
    for i in range(len(X)):
        cand=pool_fn(i)
        if len(cand)<k+1: continue
        out[i]=1-np.sort((Xn[i:i+1]@Xn[cand].T)[0])[-k:].mean()
    return out
roll=lambda i: np.where((yrs>=yrs[i]-10)&(yrs<yrs[i]))[0]           # trailing ten years
dec=(yrs//10*10); cal=lambda i: np.where(dec==dec[i]-10)[0]         # coarser: the prior calendar decade
def corr(nX,nY): m=~(np.isnan(nX)|np.isnan(nY)); return pearsonr(nX[m],nY[m])[0], int(m.sum())

nE_roll, nE_cal = novelty(E,roll), novelty(E,cal)
r_primary, n = corr(novelty(A_all,roll), nE_roll)                  # headline
r_desc, _    = corr(novelty(A_desc,roll), nE_roll)                 # evaluative attributes set aside
r_cal, _     = corr(novelty(A_all,cal),  nE_cal)                   # calendar-decade predecessor pool

# split-half reliability of the narrative-form measure: halve each film's predecessor pool at random,
# score it twice, correlate the halves, and correct to the full pool with Spearman-Brown.
def novelty_half(X, which, k=10):
    Xn=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9); out=np.full(len(X),np.nan); rng=np.random.RandomState(0)
    for i in range(len(X)):
        cand=roll(i)
        if len(cand)<2*k+2: continue
        perm=rng.permutation(cand); h=perm[:len(perm)//2] if which==0 else perm[len(perm)//2:]
        if len(h)<k+1: continue
        out[i]=1-np.sort((Xn[i:i+1]@Xn[h].T)[0])[-k:].mean()
    return out
r_half, _ = corr(novelty_half(A_all,0), novelty_half(A_all,1)); reliability = 2*r_half/(1+r_half)

# decodability: recover each atlas attribute from the embedding (5-fold CV), median R^2 per layer
Y=samp[attrs].values.astype(float); pred=np.zeros_like(Y)
for tr,te in KFold(5,shuffle=True,random_state=0).split(E): pred[te]=Ridge(10.0).fit(E[tr],Y[tr]).predict(E[te])
rows=[(a,LAY.get(a),np.corrcoef(pred[:,i],Y[:,i])[0,1]**2) for i,a in enumerate(attrs) if a in LAY and np.std(Y[:,i])>1e-9]
dl=pd.DataFrame(rows,columns=["a","layer","r2"]).groupby("layer")["r2"].median()
order=["genre","mood","structure","texture","character-arc"]; disp={"character-arc":"character\narc"}
dvals=[dl.get(l,np.nan) for l in order]

fig,(ax,axb)=plt.subplots(1,2,figsize=(11,4.4),gridspec_kw={"width_ratios":[1.3,1]})
ax.scatter(novelty(A_all,roll)[~np.isnan(nE_roll)],nE_roll[~np.isnan(nE_roll)],s=5,alpha=0.16,color="#1f3b57",edgecolors="none")
ax.set_xlabel("narrative-form novelty\n(atlas attributes)"); ax.set_ylabel("semantic novelty\n(plot-summary embedding)")
ax.set_title("a   The two novelties overlap only modestly",loc="left",fontweight="bold",fontsize=12.5)
ax.text(0.04,0.96,f"$r={r_primary:.2f}$  (n={n:,})",transform=ax.transAxes,va="top",fontsize=11,color="#333")
NAVY="#1f3b57"
axb.barh(range(len(order))[::-1],dvals,color=NAVY,height=0.62)
for yi,v in zip(range(len(order))[::-1],dvals): axb.text(v+0.015,yi,f"{v:.2f}",va="center",fontweight="bold",fontsize=10.5)
axb.set_yticks(range(len(order))[::-1]); axb.set_yticklabels([disp.get(l,l) for l in order],fontsize=10.5)
axb.set_xlim(0,0.8); axb.set_xlabel("variance recovered ($R^2$)")
axb.set_title("b   What the embedding recovers of each layer",loc="left",fontweight="bold",fontsize=12.5)
import shutil
FIG="SUPP_novelty_axes.png"; MAIN=SE+"results/figures/"+FIG
plt.tight_layout(); fig.savefig(MAIN,dpi=190,bbox_inches="tight")
# mirror the certified PNG to the package outputs/ and the Overleaf figures/ dir (skip any absent)
for m in [os.path.join(PKG,"outputs","figures",FIG),
          str(os.path.expanduser("~/Library/CloudStorage/Dropbox/Apps/Overleaf/narrative_atlas_resource/figures/"+FIG))]:
    if os.path.isdir(os.path.dirname(m)): shutil.copyfile(MAIN,m); print(f"  mirrored -> {m}")

nums={"primary_r":round(r_primary,3),"descriptive_basis_r":round(r_desc,3),"calendar_pool_r":round(r_cal,3),
      "split_half_reliability":round(reliability,3),"n":n,
      "decodability_medianR2":{l:round(float(dl.get(l,np.nan)),3) for l in order}}
json.dump(nums,open(SE+"results/novelty_appendix_numbers.json","w"),indent=2)
print("SI S1.6 numbers:",json.dumps(nums)); print("saved SUPP_novelty_axes.png")
