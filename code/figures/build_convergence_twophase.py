# Extended Data: two-phase book<->television convergence.
# Left: book-TV centroid distance by decade. Right: each medium's movement along the
# book-TV axis (TV drifts toward the novel throughout; the novel turns toward TV only after ~1990).
# Method matches replication/reproduce.py (VAL8 cross-medium-validated attributes, z-scored decade centroids).
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def load(f,idx): return pd.read_csv(f"data/corpus/{f}_structural_1890_2025.csv").rename(columns={idx:"id"})
F,B,T=load("film","film_idx"),load("book","book_idx"),load("tv","tv_idx")
VAL8_KEYS=["science_fictional","fantastical","realistic_was_the_world","world_building",
           "relatable_did_you_find","competent_was_this_protagonist","how_many_protagonists","proactiv"]
VAL8=[c for c in F.columns if any(k in c.lower() for k in VAL8_KEYS) and c in T.columns and c in B.columns]
pool=pd.concat([B[VAL8],T[VAL8],F[VAL8]]); mu,sd=pool.mean(),pool.std().replace(0,1)
def z(df): return (df[VAL8]-mu)/sd
decs=list(range(1950,2011,10))
def cent(df,d):
    s=df[(df.year>=d)&(df.year<d+10)]; return z(s).mean().values if len(s)>=30 else None
cz={m:{d:cent(df,d) for d in decs} for m,df in [("bk",B),("fm",F),("tv",T)]}
d0=decs[0]
dist=lambda a,b,d: np.linalg.norm(cz[a][d]-cz[b][d]) if cz[a][d] is not None and cz[b][d] is not None else np.nan
# book-TV axis, fixed at the first shared decade; unit vector from TV toward book(novel)
u=cz["bk"][d0]-cz["tv"][d0]; u=u/np.linalg.norm(u)
def move_toward(m, sign):  # projection of a medium's displacement (since d0) onto the axis
    return [ (sign*np.dot(cz[m][d]-cz[m][d0], u)) if cz[m][d] is not None else np.nan for d in decs]
bk_tv=[dist("bk","tv",d) for d in decs]
tv_to_novel=move_toward("tv", +1)   # TV moving toward book(novel) is +u
novel_to_tv=move_toward("bk", -1)   # book moving toward TV is -u
fig,(axL,axR)=plt.subplots(1,2,figsize=(11,4.4))
xs=[d for d in decs]
axL.plot(xs,bk_tv,color="#333",lw=2.4,marker="o"); axL.set_title("Book–television distance narrows",fontweight="bold")
axL.set_xlabel("decade"); axL.set_ylabel("centroid distance (SD units)"); axL.set_ylim(0,1.8)
axL.spines[["top","right"]].set_visible(False); axL.grid(axis="y",alpha=.15)
axR.axhline(0,color="grey",lw=.6); axR.axvline(1990,color="grey",lw=.8,ls="--",alpha=.7)
axR.plot(xs,tv_to_novel,color="#2ca02c",lw=2.4,marker="o",label="television → novel")
axR.plot(xs,novel_to_tv,color="#d62728",lw=2.4,marker="s",label="novel → television")
axR.set_title("Who moves toward whom",fontweight="bold"); axR.set_xlabel("decade")
axR.set_ylabel("movement along book–TV axis (SD)"); axR.legend(frameon=False,loc="upper left")
axR.spines[["top","right"]].set_visible(False); axR.grid(axis="y",alpha=.15)
plt.tight_layout(); plt.savefig("results/figures/fig_convergence_twophase.png",dpi=150,bbox_inches="tight"); plt.close()
print("book-TV dist:", {d:round(v,2) for d,v in zip(decs,bk_tv)})
print("tv->novel:", [round(v,2) for v in tv_to_novel]); print("novel->tv:", [round(v,2) for v in novel_to_tv])
print("saved fig_convergence_twophase.png")
