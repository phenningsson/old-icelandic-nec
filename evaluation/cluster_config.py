"""Clustering hyperparameters, EDIT them below and then run for example:

    python -m evaluation.tune --arm hybrid --type LOC

That re-clusters with these values, prints the scores, and re-renders the GUI into
evaluation/out/gui/tuned/ (refresh the browser tab served by evaluation.serve_gui).

The defaults reproduce the pipeline's current behaviour. Below each knob is the
direction to move it if the clustering is "too liberal" (non-relevant forms swept
into a cluster = over-merging / lumping) vs "too conservative" (real variants left apart, over-splitting).
"""

# ---------------------------------------------------------------------------
# 1. Distance combination  (only used by the 'hybrid' / 'contextual-hybrid' arms)
# ---------------------------------------------------------------------------
# dist = STRING_WEIGHT*string + EMBEDDING_WEIGHT*embedding   (should sum to 1.0)
STRING_WEIGHT = 0.60
EMBEDDING_WEIGHT = 0.40

# ---------------------------------------------------------------------------
# 2. String-similarity sub-weights  (should sum to 1.0)
# ---------------------------------------------------------------------------
# similarity = JW*jaro_winkler + LEV*(1-norm_levenshtein) + PREFIX*common_prefix_ratio
# Prefix is 0 in the default.
STRING_JW = 0.50
STRING_LEV = 0.50
STRING_PREFIX = 0.00

# ---------------------------------------------------------------------------
# 3. HDBSCAN  (the clustering itself)
# ---------------------------------------------------------------------------
# MIN_CLUSTER_SIZE: smallest group that counts as a cluster. 2 = allow pairs.
#   Raise (3,4,…) to suppress tiny/spurious clusters (fewer, larger clusters).
MIN_CLUSTER_SIZE = 2

# MIN_SAMPLES: how conservative. Higher = denser neighbourhood required to join a
#   cluster, so borderline/non-relevant forms are pushed to NOISE (-1 outliers) instead of being
#   swept in.
MIN_SAMPLES = 1

# CLUSTER_SELECTION_EPSILON: clusters closer than this distance are MERGED. 0.0 = no
#   extra merging. raising it makes clustering more liberal — keep at/near 0 to be
#   stricter.
CLUSTER_SELECTION_EPSILON = 0.0

# CLUSTER_SELECTION_METHOD: "eom" (excess of mass, the default) = fewer,
#   coherent clusters (less splitting); "leaf" = many fine-grained clusters (more
#   splitting).
CLUSTER_SELECTION_METHOD = "eom"

# ---------------------------------------------------------------------------
# 4. Hard distance cap (optional, post-HDBSCAN)
# ---------------------------------------------------------------------------
# If set (e.g. 0.45), any two forms whose pairwise distance exceeds the cap are not
# allowed in the same cluster: offending members are peeled off to noise (farthest-
# first) until every remaining pair is within the cap. None = disabled. This is a
# blunt but intuitive lever directly against "non-relevant terms in a cluster".
MAX_INTRA_CLUSTER_DISTANCE = 0.40
