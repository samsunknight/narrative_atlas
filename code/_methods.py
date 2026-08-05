"""Canonical feature basis for the Narrative Atlas, defined once and imported by reproduce.py and
the figure generators so the structural spine and the cross-medium set never drift between scripts.
Any script that needs the spine or the sixteen cross-medium attributes imports them from here rather
than hardcoding its own list."""
import os, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project/package root; code/ sits under it
WINDOW = (1915, 2020)   # the analysis window; decades outside it are too sparse to estimate a centroid

# The sixteen cross-medium-validated structural attributes, on a scale identical across all three
# media: the four world-type attributes, cast size, the four protagonist qualities, the two
# plot-drive axes, immersive setting, and the five plot-structure additions (development, hook, time
# and plot linearity, peripeteia). Matched by substring against the shared column names.
VAL16_KEYS = ["science_fictional", "fantastical", "realistic_was_the_world", "world_building",
              "relatable_did_you_find", "competent_was_this_protagonist", "how_many_protagonists",
              "proactiv", "plot_driven", "character_driven", "immersive", "character_development",
              "opening_hook", "time_linearity", "plot_linearity", "ending_reversal"]

def load_spine(medium):
    """The released structural corpus for one medium, keyed by a common `id` column."""
    idx = {"film": "film_idx", "book": "book_idx", "tv": "tv_idx"}[medium]
    return pd.read_csv(os.path.join(ROOT, "data", "corpus", f"{medium}_structural_1890_2025.csv")).rename(columns={idx: "id"})

def spine_attrs(df):
    """The structural-spine attribute columns of a corpus frame (everything but the identifiers)."""
    return [c for c in df.columns if c not in ("id", "title", "year", "dec", "medium", "loglen")]

def cross_medium(df_or_cols):
    """The sixteen cross-medium columns present in a frame or column list, in column order."""
    cols = df_or_cols if isinstance(df_or_cols, (list, tuple)) else list(df_or_cols.columns)
    return [c for c in cols if any(k in c.lower() for k in VAL16_KEYS)]
