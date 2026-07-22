# GENERATION STEP: within-decade partial correlation of each genre's intensity with IMDb
# log-votes (reach) and average rating (acclaim). Same within-decade design as the attribute-index
# reception analysis (reception_bootstrap.py / Supplementary Table S4). Reads only shipped files.
# Out: data/validation/genre_reception.csv  (the shipped copy is what reproduce.py reads)
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
G=['Documentary','Musical','Western','Comedy','Fantasy','Family','Thriller','Action','Drama',
   'Animation','Mystery','Historical','War','Crime','Science_Fiction','Horror','Romance','Adventure']
gcols=['genre_'+x for x in G]
A=pd.read_parquet("data/atlas/century_frame_film.parquet", columns=['idx','year']+gcols).rename(columns={'idx':'id'})
S=pd.read_csv("data/matched/imdb_film_ratings.csv")[["id","rating","votes"]]
M=S.merge(A,on="id").dropna(subset=["rating","votes","year"]+gcols).reset_index(drop=True)
M["lvotes"]=np.log1p(M.votes); M["dec"]=(M.year//10)*10
def wdp(x,out):
    rx=M[x].values-M.groupby("dec")[x].transform("mean").values
    ry=M[out].values-M.groupby("dec")[out].transform("mean").values
    return np.corrcoef(rx,ry)[0,1]
rows=[{"genre":g.replace('_',' '),"reach":round(wdp('genre_'+g,'lvotes'),3),
       "acclaim":round(wdp('genre_'+g,'rating'),3)} for g in G]
out=pd.DataFrame(rows); out["tilt"]=(out.acclaim-out.reach).round(3)
out=out.sort_values("tilt").reset_index(drop=True)
out.to_csv("data/validation/genre_reception.csv",index=False)
print(f"n={len(M)} films"); print(out.to_string(index=False))
