"""Weight sensitivity sweep.

Vary the string/embedding combination weight and score the core metrics, to show
whether 0.60/0.40 is a sensible operating point or arbitrary. The distance is
    dist = w·string + (1−w)·embedding,   w ∈ {1.0, 0.8, 0.6, 0.4, 0.2, 0.0}
run for both embedding kinds: isolation ([CLS] of the bare form) and contextual
(mean-pooled in-context). Each distance matrix is computed once per entity type;
only the cheap combine + HDBSCAN + scoring repeats per weight.

w=1.0 is string-only (identical across kinds); w=0.0 is embedding-only.

Outputs a tidy CSV (evaluation/out/weight_sweep.csv) and JSON for plotting.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluation import gold as gold_mod
from evaluation import evaluate as E
from evaluation.cluster_from_gold import (
    GOLD_FILES, _load_pipeline, gold_entities, load_raw_text,
    _string_distance_matrix, _embedding_distance_matrix, _contextual_embeddings,
)
from evaluation.fasttext_embeddings import form_vectors

_REPO = Path(__file__).resolve().parent.parent
WEIGHTS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]   # string weight; embedding weight = 1−w


def _cluster(dist, pipe):
    dist = np.ascontiguousarray(dist, dtype=np.float64)
    np.fill_diagonal(dist, 0)
    return pipe.cluster_entities(dist).tolist()


def run(model_path, dataset_path, out_dir):
    pipe = _load_pipeline()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    raw_text = load_raw_text(dataset_path)
    rows_csv = []
    report = {}

    for etype, csv_path in GOLD_FILES.items():
        entities = gold_entities(csv_path)
        texts = [e["text"] for e in entities]
        gmap = gold_mod.gold_label_map(gold_mod.load_gold(csv_path))

        print(f"[{etype}] computing distance matrices ({len(texts)} forms)...")
        string_dist = _string_distance_matrix(pipe, texts)
        emb_iso = _embedding_distance_matrix(pipe.get_embeddings(texts, model_path))
        emb_ctx = _embedding_distance_matrix(
            _contextual_embeddings(texts, raw_text, model_path, pipe))
        emb_ft = _embedding_distance_matrix(form_vectors(texts))

        report[etype] = {}
        for kind, emb_dist in (("isolation", emb_iso), ("contextual", emb_ctx),
                               ("fasttext", emb_ft)):
            report[etype][kind] = []
            for w in WEIGHTS:
                dist = w * string_dist + (1.0 - w) * emb_dist
                labels = _cluster(dist, pipe)
                pred = {t: int(l) for t, l in zip(texts, labels)}
                res = E.score_pair(gmap, pred)
                wsm, wom = res["with_singletons"], res["without_singletons"]
                rec = {
                    "entity_type": etype, "embedding_kind": kind,
                    "string_weight": w, "embedding_weight": round(1.0 - w, 2),
                    "n_noise": res["n_noise"],
                    "v_measure": wsm["v_measure"],
                    "homogeneity": wsm["homogeneity"], "completeness": wsm["completeness"],
                    "v_measure_no_singletons": wom["v_measure"],
                    "b3_f1": wsm["b3_f1"], "b3_precision": wsm["b3_precision"],
                    "b3_recall": wsm["b3_recall"],
                    "b3_f1_no_singletons": wom["b3_f1"],
                    "lumping_error": wsm["lumping_error"],
                    "splitting_error": wsm["splitting_error"],
                }
                report[etype][kind].append(rec)
                rows_csv.append(rec)
                print(f"    {kind:10} w(str)={w:.1f} | V={rec['v_measure']:.3f} "
                      f"B3F1={rec['b3_f1']:.3f} (no1s {rec['b3_f1_no_singletons']:.3f}) "
                      f"noise={rec['n_noise']}")

    csv_path = out_dir / "weight_sweep.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader(); w.writerows(rows_csv)
    json_path = out_dir / "weight_sweep.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {csv_path} and {json_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="phenningsson/icebert-old-icelandic-v11-diplomatic")
    ap.add_argument("--text", default=str(_REPO / "alexander_saga_diplomatic.txt"))
    ap.add_argument("--out", default=str(_REPO / "evaluation" / "out"))
    args = ap.parse_args(argv)
    run(args.model, args.text, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
