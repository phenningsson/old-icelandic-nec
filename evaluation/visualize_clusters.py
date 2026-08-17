"""Render a clustering arm in the interactive GUI from clustering/clustering_v5.py.

Takes the clusters produced (gold forms clustered under a chosen distance arm)
and builds the drag-and-drop cluster editor + 2D scatter ("map") HTML, so a
user can inspect, merge/split, relabel, and see the clusters laid out.

- Clusters come from the arm's prediction file (evaluation/out/predictions/), so the
  view matches exactly what was scored. HDBSCAN noise (-1) shows in the Noise (unclustured outliers) panel.
- Each form is enriched with its real occurrences in the saga text (context +
  highlight) and its frequency.
- The scatter layout is UMAP on the arm's own precomputed distance matrix, so the map
  reflects the distance that actually produced the clustering (string / embedding /
  hybrid / contextual / contextual-hybrid).

Usage:
    python -m evaluation.visualize_clusters --arm hybrid            # PER + LOC
    python -m evaluation.visualize_clusters --arm string --type LOC
    python -m evaluation.visualize_clusters --arm hybrid --type LOC
Then open the printed HTML file in a browser.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np

from evaluation import gold as gold_mod
from evaluation import pipeline_io
from evaluation.cluster_from_gold import (
    GOLD_FILES, _load_pipeline, gold_entities, load_raw_text,
    _string_distance_matrix, _embedding_distance_matrix, _contextual_embeddings,
)
from evaluation.contextual_embeddings import find_occurrences

_REPO = Path(__file__).resolve().parent.parent
PRED_DIR = _REPO / "evaluation" / "out" / "predictions"
WORK_NAME = "Alexanders saga"


def _enrich_with_contexts(entities, raw_text, max_ctx=20, window=40):
    """Attach real text occurrences (for the GUI context) and variants/works."""
    n = len(raw_text)
    for e in entities:
        ctxs = []
        for a, b in find_occurrences(e["text"], raw_text)[:max_ctx]:
            ws, we = max(0, a - window), min(n, b + window)
            ctxs.append({"text": raw_text[ws:we], "work": WORK_NAME,
                         "doc_id": None, "start": a, "end": b})
        e["contexts"] = ctxs
        e["variants"] = [e["text"]]
    return entities


def _arm_distance_matrix(entities, arm, model_path, pipe, raw_text):
    texts = [e["text"] for e in entities]
    string_dist = _string_distance_matrix(pipe, texts)
    if arm == "string":
        dist = string_dist
    else:
        if arm in ("embedding", "hybrid"):
            emb = pipe.get_embeddings(texts, model_path)
        elif arm in ("fasttext", "fasttext-hybrid"):
            from evaluation.fasttext_embeddings import form_vectors
            emb = form_vectors(texts)
        else:  # contextual / contextual-hybrid
            emb = _contextual_embeddings(texts, raw_text, model_path, pipe)
        emb_dist = _embedding_distance_matrix(emb)
        if arm in ("embedding", "contextual", "fasttext"):
            dist = emb_dist
        else:
            dist = pipe.STRING_WEIGHT * string_dist + pipe.EMBEDDING_WEIGHT * emb_dist
    dist = np.ascontiguousarray(dist, dtype=np.float64)
    np.fill_diagonal(dist, 0)
    return dist


def _umap_coords(dist):
    import umap
    n_neighbors = min(15, len(dist) - 1)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1,
                        metric="precomputed", random_state=42)
    return reducer.fit_transform(dist)


def run(arm, etypes, model_path, dataset_path, out_dir):
    pipe = _load_pipeline()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    raw_text = load_raw_text(dataset_path)
    written = []
    for etype in etypes:
        entities = gold_entities(GOLD_FILES[etype])
        _enrich_with_contexts(entities, raw_text)
        pred = pipeline_io.load_pred_labels(PRED_DIR / f"{etype.lower()}_pred_{arm}.json")
        labels = [int(pred[e["text"]]) for e in entities]

        print(f"[{etype}] arm={arm}: {len(entities)} forms, "
              f"{len({l for l in labels if l >= 0})} clusters, "
              f"{sum(1 for l in labels if l < 0)} noise — computing layout...")
        dist = _arm_distance_matrix(entities, arm, model_path, pipe, raw_text)
        coords = _umap_coords(dist)

        # The GUI names the file {type}_cluster_explorer.html; tag the arm in the dir.
        sub = out_dir / arm
        sub.mkdir(parents=True, exist_ok=True)
        path = pipe.create_interactive_editor(
            entities, labels, None, etype, str(sub), coords_2d=coords)
        written.append(path)
        print(f"   -> {path}")

    print("\nOpen in a browser:")
    for p in written:
        print(f"   file://{p}")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="hybrid",
                    choices=["string", "embedding", "hybrid", "contextual", "contextual-hybrid",
                             "fasttext", "fasttext-hybrid"])
    ap.add_argument("--type", default="both", choices=["PER", "PEO", "LOC", "both"])
    ap.add_argument("--model", default="phenningsson/icebert-old-icelandic-v11-diplomatic")
    ap.add_argument("--text", default=str(_REPO / "alexander_saga_diplomatic.txt"))
    ap.add_argument("--out", default=str(_REPO / "evaluation" / "out" / "gui"))
    args = ap.parse_args(argv)
    etypes = ["PER", "PEO", "LOC"] if args.type == "both" else [args.type]
    run(args.arm, etypes, args.model, args.text, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
