"""Synthesis: unique scorable (class-A) survey concepts (from the 9 chunk readers'
notes) cross-checked against the ACTUAL atlas frame columns + validation r's.
Every 'scored' cell names the frame column it matched, so nothing is asserted blind."""
import pandas as pd, re
frames={m:list(pd.read_parquet(f"data/atlas/century_frame_{m}.parquet").columns) for m in ("film","tv","book")}
scores={m:list(pd.read_csv(f"data/atlas/newscore/scores_{m}.csv",nrows=1).columns) for m in ("film","tv","book")}
ad=pd.read_csv("data/validation/attribute_dictionary.csv")
nv=pd.read_csv("data/validation/newattr_corpus_validation.csv")

# ---- unique class-A concepts consolidated from the 9 readers' notes ----
# (concept, qid, asks_movie, asks_book, keywords to locate a column/attribute)
C=[
("genre (select-all)","Q724",1,1,["genre_"]),
("plot-driven / character-driven","Q7",1,1,["plot_driven","character_driven","plot-driven","character-driven"]),
("mood / vibe (select-all)","Q8",1,1,["mood_"]),
("# protagonists","Q209",1,1,["n_protagonist","protagonists","Q209"]),
("# named side characters","Q211",1,1,["side_char","n_side","Q211","side-char"]),
("side-character importance","Q212",1,1,["side_char_import","imp_char","Q212","Q469"]),
("protagonist traits","Q215",1,1,["trait"]),
("marginalized-group identity","Q216",1,1,["marginal","identity_group","Q216"]),
("identity relevance","Q218",1,1,["identity_rel","id_relevance","Q218","identity"]),
("primary conflict type","Q237",1,1,["conflict","Q237"]),
("secondary conflict type","Q238",1,1,["secondary_conflict","Q238"]),
("likability (arc)","Q710",1,1,["likab","likeab"]),
("competence (arc)","Q241",1,1,["competen"]),
("proactivity (arc)","Q242",1,1,["proactiv"]),
("external change of circumstances","Q243",1,1,["external","circumstance","upheaval","Q243"]),
("personal development (internal arc)","Q244",1,1,["character_development","personal_develop","char_dev","Q244"]),
("ensemble alignment (reinforce/contradict)","Q459",1,1,["ensemble","Q459"]),
("setting when (era)","Q479",1,1,["setting_when","Q479","era"]),
("setting where (location)","Q480",1,1,["setting_where","Q480"]),
("world realistic","Q481",1,1,["realistic","Q481"]),
("world fantastical","Q483",1,1,["fantastical","Q483"]),
("world science-fictional","Q484",1,1,["scifi","science_fiction","Q484"]),
("world-building relevance","Q485",1,1,["wb_relevance","worldbuild","Q485"]),
("# major settings","Q487",1,1,["n_settings","Q487"]),
("setting linearity (spatial)","Q488",1,1,["setting_linear","spatial","Q488"]),
("plot structure (main/subplots)","Q621",1,1,["plot_structure","Q621"]),
("time linearity","Q622",1,1,["time_linearity","Q622"]),
("plot linearity","Q623",1,1,["plot_linearity","Q623"]),
("story shape (Vonnegut)","Q627",1,1,["story_shape","vonnegut","Q627"]),
("plot causality (causal/arbitrary)","Q629",1,1,["causal","Q629"]),
("pace","Q631",1,1,["pace","Q631"]),
("opening hook","Q632",1,1,["opening_hook","Q632"]),
("turning points","Q633",1,1,["turning_point","Q633"]),
("ending resolution","Q638",1,1,["resolved","resolution","Q638"]),
("ending reversal-of-fortune","Q729",1,1,["ending_reversal","reversal","Q729"]),
("ending inevitability","Q732",1,1,["inevitab","Q732"]),
("ending sudden realization","Q730",1,1,["realization","anagnorisis","Q730"]),
("ending emotional release (catharsis)","Q731",1,1,["catharsis","emotional_release","Q731"]),
("theme","Q684",1,1,["theme"]),
# book-only narration + writing
("narrative tense","Q19",0,1,["tense","Q19"]),
("narration person/POV","Q20",0,1,["person_pov","narration_person","Q20"]),
("third-person close/omniscient","Q21",0,1,["omniscient","third_person","Q21"]),
("# narrators","Q22",0,1,["n_narrator","Q22"]),
("narrator reliability","Q25",0,1,["reliab","Q25"]),
("narrator tone","Q39",0,1,["tone_"]),
("narration switches","Q51",0,1,["switch","Q51"]),
("writing-style descriptors","Q656",0,1,["writing_"]),
("showing vs telling","Q663","1?",1,["showtell","show_tell","showing"]),
]
def find(m,kws):
    for c in frames[m]:
        cl=c.lower()
        for k in kws:
            if k.lower() in cl: return c
    return ""
def r_lookup(kws):
    fr=br=None
    for _,row in ad.iterrows():
        blob=(str(row["attribute"])+" "+str(row["column"])).lower()
        if any(k.lower() in blob for k in kws):
            fr=row["film_r"] if pd.notna(row["film_r"]) else fr
            br=row["book_r"] if pd.notna(row["book_r"]) else br
    for _,row in nv.iterrows():
        if any(k.lower() in str(row["attribute"]).lower() for k in kws):
            if row["medium"]=="film" and pd.notna(row.get("r")): fr=row["r"]
            if row["medium"]=="book" and pd.notna(row.get("r")): br=row["r"]
    return fr,br
rows=[]
for concept,qid,mv,bk,kws in C:
    ff=find("film",kws) or (concept in " ".join(scores["film"]) and "scores:"+ [c for c in scores["film"] if any(k.lower() in c.lower() for k in kws)][0]) if any(any(k.lower() in c.lower() for k in kws) for c in scores["film"]) else find("film",kws)
    def col(m):
        c=find(m,kws)
        if not c:
            sc=[x for x in scores[m] if any(k.lower() in x.lower() for k in kws)]
            if sc: return "newscore:"+sc[0]
        return c
    cf,ct,cb=col("film"),col("tv"),col("book")
    fr,br=r_lookup(kws)
    rows.append(dict(concept=concept,qid=qid,asks_movie=mv,asks_book=bk,
        film=cf or "—", tv=ct or "—", book=cb or "—", film_r=fr, book_r=br))
out=pd.DataFrame(rows)
out.to_csv("data/validation/scorable_concept_crosswalk.csv",index=False)
# print
pd.set_option("display.max_colwidth",34); pd.set_option("display.width",200)
print(out.to_string(index=False))
print("\nrows:",len(out))
print("GAP film (asks_movie & film=—):", out[(out.asks_movie==1)&(out.film=='—')].concept.tolist())
print("GAP book (asks_book & book=—):", out[(out.asks_book==1)&(out.book=='—')].concept.tolist())
