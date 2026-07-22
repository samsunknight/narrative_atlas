#!/usr/bin/env python3
"""
Rebuild the two IMDb-derived match files that are NOT redistributed in this package
(IMDb data is under a non-commercial license). Running this regenerates:
    data/matched/imdb_film_ratings.csv   (id, rating, votes)
    data/matched/imdb_film_genres.csv    (id, imdb_genres)
keyed by the film corpus id (film_idx), which re-enables the 13 IMDb-dependent checks
in reproduce.py (genre structural->IMDb AUC, reception partials, genre decomposition).

The join key is the film's title + release year, which IS shipped (data/corpus/
film_structural_1890_2025.csv). We match it to IMDb's public dataset the same way the
paper did: normalized title + year against movie-type IMDb titles.

USAGE
-----
1. Download IMDb's public (non-commercial) dataset files from
   https://developer.imdb.com/non-commercial-datasets/ :
       title.basics.tsv.gz     (primaryTitle, originalTitle, startYear, titleType, genres)
       title.ratings.tsv.gz    (tconst, averageRating, numVotes)
2. Run:  python3 code/rebuild_imdb_match.py /path/to/title.basics.tsv.gz /path/to/title.ratings.tsv.gz
   (accepts .tsv or .tsv.gz)
"""
import sys, os, re
import pandas as pd

def norm(s): return re.sub(r'[^a-z0-9]', '', str(s).lower())

def main(basics_path, ratings_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus = pd.read_csv(os.path.join(root, "data/corpus/film_structural_1890_2025.csv")).rename(columns={"film_idx": "id"})
    corpus = corpus[["id", "title", "year"]].dropna(subset=["title", "year"]).copy()
    corpus["k"] = corpus.title.map(norm); corpus["y"] = corpus.year.astype(int)

    b = pd.read_csv(basics_path, sep="\t", na_values="\\N",
                    usecols=["tconst", "titleType", "primaryTitle", "originalTitle", "startYear", "genres"],
                    dtype=str)
    b = b[b.titleType.isin(["movie", "tvMovie"])].copy()
    b["y"] = pd.to_numeric(b.startYear, errors="coerce")
    b = b.dropna(subset=["y"]); b["y"] = b.y.astype(int)
    # one IMDb row per (normalized title, year); primary title preferred, original title as fallback
    rows = []
    for tcol in ("primaryTitle", "originalTitle"):
        t = b.dropna(subset=[tcol]).copy(); t["k"] = t[tcol].map(norm)
        rows.append(t[["tconst", "k", "y", "genres"]])
    bt = pd.concat(rows).drop_duplicates(["k", "y"])

    r = pd.read_csv(ratings_path, sep="\t", na_values="\\N", dtype={"tconst": str})
    r["averageRating"] = pd.to_numeric(r.averageRating, errors="coerce")
    r["numVotes"] = pd.to_numeric(r.numVotes, errors="coerce")

    m = corpus.merge(bt, on=["k", "y"], how="inner").merge(r, on="tconst", how="left")
    m = m.drop_duplicates("id")

    out = os.path.join(root, "data/matched"); os.makedirs(out, exist_ok=True)
    rat = m.dropna(subset=["averageRating", "numVotes"])[["id"]].copy()
    rat["rating"] = m.loc[rat.index, "averageRating"].values
    rat["votes"] = m.loc[rat.index, "numVotes"].values
    rat.to_csv(os.path.join(out, "imdb_film_ratings.csv"), index=False)

    gen = m.dropna(subset=["genres"])[["id"]].copy()
    gen["imdb_genres"] = m.loc[gen.index, "genres"].values
    gen.to_csv(os.path.join(out, "imdb_film_genres.csv"), index=False)

    print(f"wrote data/matched/imdb_film_ratings.csv ({len(rat):,} films) and "
          f"imdb_film_genres.csv ({len(gen):,} films). reproduce.py will now run the 13 IMDb-dependent checks.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python3 code/rebuild_imdb_match.py <title.basics.tsv[.gz]> <title.ratings.tsv[.gz]>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
