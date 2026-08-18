import pandas as pd, numpy as np, re, warnings
warnings.filterwarnings("ignore")
def textlen(path):
    d=pd.read_parquet(path).reset_index(drop=True)
    tc=[c for c in d.columns if c in("text","plot","summary")][0]
    d=d[d[tc].notna() & (d[tc].astype(str).str.len()>300)].reset_index(drop=True)
    return d[tc].astype(str).str.len()
def loadc(f,idx,path):
    d=pd.read_csv(f"data/corpus/{f}_structural_century.csv").rename(columns={idx:"id"})
    d=d[(d.year>=1910)&(d.year<=2020)]; d["year"]=d.year.astype(int)
    L=textlen(path); d["loglen"]=np.log(d["id"].map(L).fillna(L.median())); return d
F=loadc("film","film_idx","/tmp/dbx_dl/film_wiki_text.parquet")
B=loadc("book","book_idx","/Users/samsunknight/uoft/style_evolves/data/book_wiki_text_century.parquet")
T=loadc("tv","tv_idx","/tmp/dbx_dl/tv_wiki_text_century.parquet")
ALL=pd.concat([F,B,T],ignore_index=True)
ATTRS=[c for c in F.columns if c not in("id","title","year","medium","loglen")]
def short(c):
    for k,n in [("fantastical","fantastical"),("science_fictional","scifi"),("realistic_was_the_world","realistic"),("world_building_to","worldbuild"),("how_many_protagonists","#protag"),("named_side","#sidechar"),("competent","competence"),("proactive","proactive"),("likable","likability"),("relatable","relatable"),("real_did_this","feelsreal"),("emotionally_invested","emotinvest"),("pace_of_the","pace"),("unsurprising","surprise"),("convincing","plotconvince"),("interesting_did_you_find_the_visual","visinterest"),("immersive","immersive"),("how_many_major_settings","#settings"),("plot_driven","plotvschar")]:
        if k in c: return n
    return c[:10]
SN={c:short(c) for c in ATTRS}
print("="*64)
# 1. did summaries get longer over time / differ by medium?
print("PLOT-SUMMARY LENGTH — corr with year (the artifact driver):")
for d,nm in [(F,"film"),(B,"book"),(T,"tv")]:
    print(f"  {nm}: r(year,loglen)={np.corrcoef(d.year,d.loglen)[0,1]:+.2f} | median chars: 1950s={int(np.exp(d[d.year<1960].loglen.median()))} 2010s={int(np.exp(d[d.year>=2010].loglen.median()))}")
# 2. which attributes are length-driven (suspects)?
print("\nATTRIBUTE x LENGTH correlation (film; high = length-inflated suspect):")
ls=sorted([(SN[a],np.corrcoef(F[a].fillna(F[a].mean()),F.loglen)[0,1]) for a in ATTRS],key=lambda z:-z[1])
print("  most length-driven:", ", ".join(f"{n}{v:+.2f}" for n,v in ls[:5]))
print("  least length-driven:", ", ".join(f"{n}{v:+.2f}" for n,v in ls[-4:]))
# 3. residualize each attr on loglen (within medium) and re-run the headline trends
def resid(d):
    r=d.copy()
    for a in ATTRS:
        x=d.loglen.values; y=d[a].fillna(d[a].mean()).values
        b=np.polyfit(x,y,1); r[a]=y-np.polyval(b,x)
    return r
Fr,Br,Tr=resid(F),resid(B),resid(T); ALLr=pd.concat([Fr,Br,Tr],ignore_index=True)
RATCHET=["scifi","#settings","#protag","#sidechar","immersive","pace","visinterest","worldbuild"]
rc=[a for a in ATTRS if SN[a] in RATCHET]
print("\nSPECTACLE-ESCALATION: raw vs length-residualized (film, z-mean ratchet attrs by decade):")
for label,d in [("RAW",F),("RESID",Fr)]:
    z=(d[rc]-ALL[rc].mean())/ALL[rc].std() if label=="RAW" else (d[rc]-ALLr[rc].mean())/ALLr[rc].std()
    idx=z.assign(dec=(d.year//10)*10).groupby("dec").mean().mean(axis=1)
    idx=idx[d.assign(dec=(d.year//10)*10).groupby("dec").size()>=30]
    chg=idx[idx.index>=2000].mean()-idx[idx.index<1950].mean()
    print(f"  {label}: 1920s={idx[idx.index<1930].mean():+.2f} 2010s={idx[idx.index>=2010].mean():+.2f}  Δ(post2000 - pre1950)={chg:+.2f}")
# 4. ratchet/fashion autocorr survives residualization?
print("\nRATCHET/FASHION autocorr: raw vs residualized (film):")
for grp,nm in [(rc,"RATCHET"),([a for a in ATTRS if SN[a] in ["surprise","proactive","competence","plotvschar"]],"FASHION")]:
    for label,d in [("raw",F),("resid",Fr)]:
        dec=d.assign(dec=(d.year//10)*10).groupby("dec")[grp].mean(); dec=dec[d.assign(dec=(d.year//10)*10).groupby("dec").size()>=30]
        acs=[np.corrcoef(((dec[a]-dec[a].mean())/dec[a].std()).dropna().values[:-1],((dec[a]-dec[a].mean())/dec[a].std()).dropna().values[1:])[0,1] for a in grp if len(dec[a].dropna())>5]
        print(f"  {nm} {label}: mean autocorr={np.mean(acs):.2f}", end="  " if label=="raw" else "\n")
# 5. sci-fi book->film lead-lag survives residualization?
print("\nSCI-FI book->film lead-lag: raw vs residualized:")
def ys_(d,a,mn=15):
    g=d.dropna(subset=[a]).groupby("year")[a].agg(["mean","count"]); return g[g["count"]>=mn]["mean"]
sci=[a for a in ATTRS if SN[a]=="scifi"][0]
for label,(bb,ff) in [("raw",(B,F)),("resid",(Br,Fr))]:
    sx,sy=ys_(bb,sci),ys_(ff,sci); yrs=sorted(set(sx.index)&set(sy.index))
    x=sx.reindex(yrs).rolling(3,center=True).mean().diff().dropna(); y=sy.reindex(yrs).rolling(3,center=True).mean().diff().dropna()
    yy=sorted(set(x.index)&set(y.index)); x,y=x.reindex(yy).values,y.reindex(yy).values
    best=(0,0)
    for lag in range(0,7):
        a2,b2=(x[lag:],y[:len(y)-lag]) if lag>0 else (x,y); n=min(len(a2),len(b2))
        if n>15 and np.corrcoef(a2[:n],b2[:n])[0,1]>best[1]: best=(lag,np.corrcoef(a2[:n],b2[:n])[0,1])
    print(f"  {label}: book leads film {best[0]}yr r={best[1]:.2f}")
