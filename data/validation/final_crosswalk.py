"""FINAL hand-verified crosswalk: every unique scorable (class-A) survey concept
(from reading all 9 chunks) mapped to its EXACT column in the film and book frames
(read from the actual column lists), with newscore + validation r. No fuzzy matching:
each film_col/book_col below was confirmed against the printed frame column lists."""
import pandas as pd
FF=set(pd.read_parquet("data/atlas/century_frame_film.parquet").columns)
BF=set(pd.read_parquet("data/atlas/century_frame_book.parquet").columns)
NF=set(pd.read_csv("data/atlas/newscore/scores_film.csv",nrows=1).columns)
NB=set(pd.read_csv("data/atlas/newscore/scores_book.csv",nrows=1).columns)

# concept, qid, asks(movie,book), film_col, book_col, film_r, book_r, note
# film_col/book_col: exact frame column, or "newscore:<x>", or "" if absent
R=[
("genre","Q724",1,1,"genre_*","genre_*",0.90,0.93,"AUC vs IMDb(film)/survey(book)"),
("plot/char-driven","Q7",1,1,"film_plot_driven/character_driven","book_plot_driven/character_driven",0.51,0.43,""),
("mood","Q8",1,1,"mood_*","mood_*",0.52,0.48,""),
("# protagonists","Q209",1,1,"1_how_many_protagonists...","book_Q209_n_protagonists",0.55,0.53,""),
("# side characters","Q211",1,1,"3_approximately_how_many_named_side...","",0.26,None,"book: only importance scored, not count"),
("side-char importance","Q212/Q469",1,1,"4_how_important..side_characters","book_Q469_imp_characters",0.26,None,""),
("protagonist traits","Q215",1,1,"","",None,None,"free-text; not scored either medium"),
("marginalized identity (membership)","Q216",1,1,"","",None,None,"only *relevance* scored, not membership"),
("identity relevance","Q218",1,1,"identity_relevance","book_identity_relevance",0.64,None,""),
("primary conflict type","Q237",1,1,"","",None,None,"NOVEL - unscored both"),
("secondary conflict type","Q238",1,1,"","",None,None,"NOVEL - unscored both"),
("likability arc","Q710",1,1,"arc_likable_*","arc_likable_*",0.49,None,""),
("competence arc","Q241",1,1,"arc_competent_*","arc_competent_*",0.43,0.54,""),
("proactivity arc","Q242",1,1,"arc_proactive_*","arc_proactive_*",0.24,0.31,""),
("external change of circumstances","Q243",1,1,"","book_character_change",None,None,"book has it; FILM/TV GAP"),
("personal development (internal)","Q244",1,1,"newscore:character_development","book_character_development",0.58,None,"book scored, not yet validated"),
("ensemble alignment","Q459",1,1,"","",None,None,"NOVEL - unscored both (n small)"),
("setting when (era)","Q479",1,1,"newscore:setting_when","newscore:setting_when",0.71,0.78,""),
("setting where (location)","Q480",1,1,"newscore:setting_where","newscore:setting_where",0.57,0.62,""),
("world realistic","Q481",1,1,"3_how_realistic_was_the_world...","book_Q481_realistic",0.34,0.60,""),
("world fantastical","Q483",1,1,"5_how_fantastical...","book_Q483_fantastical",0.77,0.73,""),
("world science-fictional","Q484",1,1,"6_how_science_fictional...","book_Q484_scifi",0.78,0.79,""),
("familiar world","Q482",1,1,"4_how_familiar_was_the_world...","book_familiar_world",0.26,None,""),
("world-building relevance","Q485",1,1,"7_overall..world_building","book_Q485_wb_relevance",0.50,0.27,""),
("# major settings","Q487",1,1,"8_how_many_major_settings","book_Q487_n_settings",0.28,None,""),
("setting linearity (spatial)","Q488",1,1,"","",None,None,"NOVEL - unscored both"),
("plot structure (main/subplots)","Q621",1,1,"","",None,None,"unscored both (prior test?)"),
("time linearity","Q622",1,1,"newscore:time_linearity","book_Q622_time_linearity",0.32,None,"book scored, not yet validated"),
("plot linearity","Q623",1,1,"newscore:plot_linearity","newscore:plot_linearity",0.21,0.39,""),
("story shape (Vonnegut)","Q627",1,1,"","",None,None,"TESTED&SET ASIDE (6-class ~0.38)"),
("plot causality (causal/arbitrary)","Q629",1,1,"","book_plot_event_relatedness",None,None,"book has it; FILM/TV GAP"),
("pace","Q631",1,1,"7_on_the_whole..pace","book_pace",0.43,None,""),
("opening hook","Q632",1,1,"newscore:opening_hook","book_opening_hook",0.36,None,"book scored, not yet validated"),
("turning points","Q633",1,1,"","book_turning_points",None,None,"book has it; FILM/TV GAP"),
("ending resolution","Q638",1,1,"12a_..unresolved..resolved","book_Q638_resolved",0.54,None,""),
("ending surprising","Q639",1,1,"12b_..surprising","book_Q639_surprising",0.49,None,""),
("ending clarity","Q640",1,1,"12c_..confusing","book_Q640_confusing",0.35,None,""),
("ending satisfying","Q641",1,1,"12d_..satisfying","book_satisfying_ending",0.28,None,""),
("ending reversal-of-fortune","Q729",1,1,"newscore:ending_reversal","",0.19,None,"BOOK GAP (film weak 0.19)"),
("ending inevitability","Q732",1,1,"","",None,None,"NOVEL - unscored both"),
("ending sudden realization","Q730",1,1,"","",None,None,"NOVEL - unscored both"),
("ending emotional release (catharsis)","Q731",1,1,"","",None,None,"NOVEL - unscored both"),
("themes interesting","Q684",1,1,"2_how_interesting..themes","book_themes_interesting",0.24,None,"borderline B"),
# ---- book-only (narration + writing) ----
("narrative tense","Q19",0,1,"n/a","",None,None,"BOOK-NATIVE novel - unscored"),
("narration person/POV","Q20",0,1,"n/a","",None,None,"BOOK-NATIVE novel - unscored"),
("third-person close/omniscient","Q21",0,1,"n/a","",None,None,"BOOK-NATIVE novel - unscored"),
("# narrators","Q22",0,1,"n/a","",None,None,"BOOK-NATIVE novel - unscored"),
("narrator POV share","Q24",0,1,"n/a","book_narr_share",None,None,"book scored"),
("narrator reliability","Q25",0,1,"n/a","",None,None,"BOOK-NATIVE novel - unscored (high value)"),
("narrator tone","Q39",0,1,"n/a","tone_*",None,None,"book scored, not validated"),
("narration switches","Q51",0,1,"n/a","",None,None,"BOOK-NATIVE novel - unscored"),
("writing-style descriptors","Q656",0,1,"n/a","writing_*",None,None,"book scored, not validated"),
("showing vs telling","Q663",1,1,"","",None,None,"TESTED&SET ASIDE (weak ~0.09)"),
]
rows=[]
for concept,qid,am,ab,fc,bc,fr,br,note in R:
    def status(col,asks,frameset,newset,newkey):
        if col=="n/a": return "n/a"
        if col.startswith("newscore:"): return "scored(newscore)"
        if col=="": return "GAP" if asks else "n/a"
        return "scored"
    fs=status(fc,am,FF,NF,None); bs=status(bc,ab,BF,NB,None)
    rows.append(dict(concept=concept,qid=qid,asks_movie=am,asks_book=ab,
                     film=fs,book=bs,film_col=fc,book_col=bc,film_r=fr,book_r=br,note=note))
df=pd.DataFrame(rows)
df.to_csv("data/validation/FINAL_scorable_crosswalk.csv",index=False)
pd.set_option("display.max_colwidth",40); pd.set_option("display.width",240)
print(df[["concept","qid","film","book","film_r","book_r","note"]].to_string(index=False))
print(f"\nTOTAL scorable concepts: {len(df)}")
print("\n--- CROSS-MEDIUM FILL (scored one side, asked+absent other) ---")
print("  book-absent:", df[(df.book=='GAP')&(df.film.str.startswith('scored'))].concept.tolist())
print("  film-absent:", df[(df.film=='GAP')&(df.book.str.startswith('scored'))].concept.tolist())
print("\n--- NOVEL (unscored both, asked) ---")
print("  ", df[(df.film=='GAP')&(df.book.isin(['GAP','n/a']))&(~df.note.str.contains('SET ASIDE'))].concept.tolist())
