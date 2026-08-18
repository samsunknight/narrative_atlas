# Which attribute dictionary is canonical

Two codebooks live in this project. Do not confuse them.

- **`attribute_dictionary.csv` here in the working tree = 195 rows** — an *exploratory superset* from
  the identity/narration/character expansion. It is NOT what the atlas paper reports and NOT what
  ships. Its `tier` column mixes two labeling schemes and it includes attributes the paper scopes out.

- **The shipped/paper codebook = 161 rows**, at
  `rep_build/narrative_atlas/data/validation/attribute_dictionary.csv`. This is canonical: the atlas
  paper's Table 1 (161 scored / 150 validated, nine constructs), Supplementary Table S1
  (`build_tab_attribute_dictionary.py` emits exactly 161), and `reproduce.py` (132/132) all agree on
  it.

When another paper (the HumanNarrative dataset paper, the review-audit paper) reconciles against the
atlas, it should compare to the **161-row shipped** codebook, not the 195-row working file. The
"atlas reports 161 but its dictionary is 195" discrepancy is an artifact of reading this working file.
