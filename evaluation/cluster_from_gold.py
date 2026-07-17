"""Run the clustering pipeline on the GOLD forms (the canonical input).

The clusterer only needs the cased surface-form TYPE strings (both the embedding and
the string distance are computed from the form itself), so we feed it the gold forms
directly. This guarantees the prediction item set equals the gold item set by
construction. Contextual embeddings additionally read the raw diplomatic text.

Reuses the pipeline's own distance + HDBSCAN core (clustering/clustering_v5.py) so
the method is identical to the production pipeline; only the input source differs.

Distance arms:
  - string    : string distance only            (no model needed; the proposed system)
  - embedding : cosine on IceBERT [CLS] only     (needs model)
  - hybrid    : 0.60*string + 0.40*embedding     (needs model; ablation, not the proposed system)

Output: per entity type, a {form: predicted_label} JSON (HDBSCAN noise = -1),
which evaluation/evaluate.py consumes directly.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from evaluation import gold as gold_mod

_REPO = Path(__file__).resolve().parent.parent
_PIPELINE_PATH = _REPO / "clustering" / "clustering_v5.py"
# Three entity types, matching the source edition's name typing:
#   PER = personal names; PEO = ethnonyms / peoples; LOC = locations.
GOLD_FILES = {"PER": _REPO / "alexander_saga_persons.csv",
              "PEO": _REPO / "alexander_saga_people.csv",
              "LOC": _REPO / "alexander_saga_locations.csv"}
# Canonical ordered entity types (single source of truth for the harness).
ETYPES = ["PER", "PEO", "LOC"]


def _load_pipeline():
    spec = importlib.util.spec_from_file_location("clustering_v5", _PIPELINE_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gold_entities(csv_path):
    """Build the pipeline's entity dicts straight from a gold CSV (one per cased
    type), carrying frequency. Order is sorted for reproducibility."""
    rows = gold_mod.load_gold(csv_path)
    seen = {}
    for r in rows:
        # gold is one row per cased type; keep first, accumulate freq defensively
        if r.entity in seen:
            seen[r.entity]["frequency"] += r.frequency
        else:
            seen[r.entity] = {"text": r.entity, "frequency": r.frequency}
    return [seen[k] for k in sorted(seen)]


def _string_distance_matrix(pipe, texts):
    n = len(texts)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = pipe.string_distance(texts[i], texts[j])
            d[i, j] = d[j, i] = v
    return d


def _embedding_distance_matrix(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms
    cosine = np.dot(normalized, normalized.T)
    return np.clip(1.0 - cosine, 0, 2)


def _contextual_embeddings(texts, raw_text, model_path, pipe):
    """Mean-pooled contextual embeddings per form, with isolation fallback for any
    form that could not be located in the raw text."""
    from evaluation.contextual_embeddings import contextual_form_embeddings
    vecmap = contextual_form_embeddings(texts, raw_text, model_path)
    missing = [t for t in texts if vecmap.get(t) is None]
    if missing:
        print(f"    contextual: {len(missing)} form(s) not located in text; "
              f"falling back to isolation embedding: {missing[:10]}")
        for t, v in zip(missing, pipe.get_embeddings(missing, model_path)):
            vecmap[t] = v
    return np.array([vecmap[t] for t in texts])


def cluster_forms(entities, arm, model_path, pipe, raw_text=None):
    """Return predicted labels (list[int]) for `entities` under a distance arm."""
    texts = [e["text"] for e in entities]
    string_dist = _string_distance_matrix(pipe, texts)

    if arm == "string":
        dist = string_dist
    else:
        if arm in ("embedding", "hybrid"):
            embeddings = pipe.get_embeddings(texts, model_path)
        elif arm in ("contextual", "contextual-hybrid"):
            if raw_text is None:
                raise ValueError(f"arm {arm!r} requires raw_text")
            embeddings = _contextual_embeddings(texts, raw_text, model_path, pipe)
        elif arm in ("fasttext", "fasttext-hybrid"):
            from evaluation.fasttext_embeddings import form_vectors
            embeddings = form_vectors(texts)
        else:
            raise ValueError(f"unknown arm: {arm!r}")
        emb_dist = _embedding_distance_matrix(embeddings)
        if arm in ("embedding", "contextual", "fasttext"):
            dist = emb_dist
        else:  # hybrid / contextual-hybrid / fasttext-hybrid
            dist = pipe.STRING_WEIGHT * string_dist + pipe.EMBEDDING_WEIGHT * emb_dist
    # HDBSCAN's precomputed path requires float64; embedding distances are float32.
    dist = np.ascontiguousarray(dist, dtype=np.float64)
    np.fill_diagonal(dist, 0)
    return pipe.cluster_entities(dist).tolist()


def load_raw_text(text_path):
    """Load the raw diplomatic text (used by the contextual arm for occurrence search)."""
    with open(text_path, encoding="utf-8") as f:
        return f.read()


def run(arm, model_path, out_dir, text_path=None):
    pipe = _load_pipeline()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_text = None
    if arm in ("contextual", "contextual-hybrid"):
        raw_text = load_raw_text(text_path)
    written = {}
    for etype, csv_path in GOLD_FILES.items():
        entities = gold_entities(csv_path)
        labels = cluster_forms(entities, arm, model_path, pipe, raw_text=raw_text)
        pred = {e["text"]: int(lbl) for e, lbl in zip(entities, labels)}
        n_clusters = len({l for l in labels if l >= 0})
        n_noise = sum(1 for l in labels if l < 0)
        path = out_dir / f"{etype.lower()}_pred_{arm}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(pred, f, ensure_ascii=False, indent=2)
        written[etype] = str(path)
        print(f"[{etype}] arm={arm}: {len(entities)} forms -> "
              f"{n_clusters} clusters, {n_noise} noise  -> {path}")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="string",
                    choices=["string", "embedding", "hybrid", "contextual", "contextual-hybrid",
                             "fasttext", "fasttext-hybrid"],
                    help="Distance arm (default: string; only 'string' needs no model). "
                         "'contextual'/'contextual-hybrid' embed occurrences in context; "
                         "'fasttext'/'fasttext-hybrid' use the trained subword model.")
    ap.add_argument("--model", default="phenningsson/icebert-old-icelandic-v11-diplomatic",
                    help="Embedding model (HF id or local path); needed for non-string arms.")
    ap.add_argument("--text", default=str(_REPO / "alexander_saga_diplomatic.txt"),
                    help="Raw diplomatic text file (used by the contextual arms only).")
    ap.add_argument("--out", default=str(_REPO / "evaluation" / "out" / "predictions"))
    args = ap.parse_args(argv)
    run(args.arm, args.model, args.out, text_path=args.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
