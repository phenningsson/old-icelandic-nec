"""Apply hyperparameters, re-cluster, score, and re-render the GUI.

Two ways to set hyperparameters (CLI wins over the file, per run):
    1. edit evaluation/cluster_config.py  (the persistent default baseline), or
    2. pass CLI overrides, e.g.
         python -m evaluation.tune --arm hybrid --type LOC \
             --max-intra-distance 0.35 --string-prefix 0.15 --string-lev 0.45

Then refresh the browser tab (served by `python -m evaluation.serve_gui`).

Writes the tuned GUI to evaluation/out/gui/tuned/, tuned predictions to
evaluation/out/tuned/, and one row per run to evaluation/out/tune_log.csv. It does
not touch the canonical results/predictions.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evaluation import cluster_config as CFG
from evaluation import gold as gold_mod
from evaluation import evaluate as E
from evaluation.cluster_from_gold import (
    GOLD_FILES, _load_pipeline, gold_entities, load_raw_text,
    _embedding_distance_matrix, _contextual_embeddings,
)
from evaluation.visualize_clusters import _enrich_with_contexts, _umap_coords

_REPO = Path(__file__).resolve().parent.parent
TUNE_LOG = _REPO / "evaluation" / "out" / "tune_log.csv"

_CONFIG_KEYS = ("STRING_WEIGHT", "EMBEDDING_WEIGHT", "STRING_JW", "STRING_LEV",
                "STRING_PREFIX", "MIN_CLUSTER_SIZE", "MIN_SAMPLES",
                "CLUSTER_SELECTION_EPSILON", "CLUSTER_SELECTION_METHOD",
                "MAX_INTRA_CLUSTER_DISTANCE")

# CLI flag (argparse dest) -> cluster_config.py key
_CLI_TO_CFG = {
    "string_weight": "STRING_WEIGHT",
    "embedding_weight": "EMBEDDING_WEIGHT",
    "string_jw": "STRING_JW",
    "string_lev": "STRING_LEV",
    "string_prefix": "STRING_PREFIX",
    "min_cluster_size": "MIN_CLUSTER_SIZE",
    "min_samples": "MIN_SAMPLES",
    "epsilon": "CLUSTER_SELECTION_EPSILON",
    "selection_method": "CLUSTER_SELECTION_METHOD",
    "max_intra_distance": "MAX_INTRA_CLUSTER_DISTANCE",
}


def effective_config(overrides):
    """cluster_config.py defaults, with any CLI overrides applied on top."""
    ns = SimpleNamespace(**{k: getattr(CFG, k) for k in _CONFIG_KEYS})
    for key, val in overrides.items():
        setattr(ns, key, val)
    return ns


def _append_log(row, path=TUNE_LOG):
    """Append one tuning run to the comparison CSV (writes header if new)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def _string_sim(s1, s2, pipe, cfg):
    if s1 == s2:
        return 1.0
    jw = pipe.jaro_winkler_similarity(s1, s2)
    lev = 1.0 - pipe.normalized_levenshtein(s1, s2)
    ml = max(len(s1), len(s2))
    pr = (pipe.common_prefix_length(s1, s2) / ml) if ml else 0.0
    return cfg.STRING_JW * jw + cfg.STRING_LEV * lev + cfg.STRING_PREFIX * pr


def _string_dist_matrix(texts, pipe, cfg):
    n = len(texts)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = 1.0 - _string_sim(texts[i], texts[j], pipe, cfg)
            d[i, j] = d[j, i] = v
    return d


def _distance_matrix(texts, arm, model_path, pipe, raw_text, cfg):
    string_dist = _string_dist_matrix(texts, pipe, cfg)
    if arm == "string":
        dist = string_dist
    else:
        if arm in ("embedding", "hybrid"):
            emb = pipe.get_embeddings(texts, model_path)
        else:
            emb = _contextual_embeddings(texts, raw_text, model_path, pipe)
        emb_dist = _embedding_distance_matrix(emb)
        if arm in ("embedding", "contextual"):
            dist = emb_dist
        else:
            dist = cfg.STRING_WEIGHT * string_dist + cfg.EMBEDDING_WEIGHT * emb_dist
    dist = np.ascontiguousarray(dist, dtype=np.float64)
    np.fill_diagonal(dist, 0)
    return dist


def _cluster(dist, cfg):
    import hdbscan
    c = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=int(cfg.MIN_CLUSTER_SIZE),
        min_samples=int(cfg.MIN_SAMPLES),
        cluster_selection_epsilon=float(cfg.CLUSTER_SELECTION_EPSILON),
        cluster_selection_method=cfg.CLUSTER_SELECTION_METHOD,
    )
    return c.fit_predict(dist).tolist()


def _apply_distance_cap(labels, dist, cap):
    """Peel offending members to noise (most-offending first) until every remaining
    pair within each cluster is within `cap`."""
    labels = np.array(labels)
    for cid in sorted(set(labels)):
        if cid < 0:
            continue
        idx = np.where(labels == cid)[0]
        while len(idx) > 1:
            sub = dist[np.ix_(idx, idx)]
            if sub.max() <= cap:
                break
            offender = idx[np.argmax(sub.sum(axis=1))]
            labels[offender] = -1
            idx = idx[idx != offender]
    return labels.tolist()


def run(arm, etypes, model_path, dataset_path, cfg, note="", overridden=()):
    pipe = _load_pipeline()
    raw_text = load_raw_text(dataset_path)
    gui_dir = _REPO / "evaluation" / "out" / "gui" / "tuned"
    pred_dir = _REPO / "evaluation" / "out" / "tuned"
    gui_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    print("Effective hyperparameters (CLI override > cluster_config.py):")
    for k in _CONFIG_KEYS:
        tag = "  <-- CLI" if k in overridden else ""
        print(f"    {k} = {getattr(cfg, k)}{tag}")
    print()
    cfg_values = {k: getattr(cfg, k) for k in _CONFIG_KEYS}
    timestamp = datetime.now().isoformat(timespec="seconds")

    written = []
    for etype in etypes:
        entities = gold_entities(GOLD_FILES[etype])
        _enrich_with_contexts(entities, raw_text)
        texts = [e["text"] for e in entities]
        gmap = gold_mod.gold_label_map(gold_mod.load_gold(GOLD_FILES[etype]))

        dist = _distance_matrix(texts, arm, model_path, pipe, raw_text, cfg)
        labels = _cluster(dist, cfg)
        if cfg.MAX_INTRA_CLUSTER_DISTANCE is not None:
            labels = _apply_distance_cap(labels, dist, float(cfg.MAX_INTRA_CLUSTER_DISTANCE))

        pred = {t: int(l) for t, l in zip(texts, labels)}
        n_clusters = len({l for l in labels if l >= 0})
        n_noise = sum(1 for l in labels if l < 0)
        res = E.score_pair(gmap, pred)
        w, o = res["with_singletons"], res["without_singletons"]
        print(f"[{etype}] arm={arm}: {n_clusters} clusters, {n_noise} noise")
        print(f"    with singletons : V={w['v_measure']:.3f}  B3F1={w['b3_f1']:.3f}  "
              f"lump={w['lumping_error']:.3f} split={w['splitting_error']:.3f}")
        print(f"    without singletons: V={o['v_measure']:.3f}  B3F1={o['b3_f1']:.3f}")

        _append_log({
            "timestamp": timestamp, "note": note, "arm": arm, "entity_type": etype,
            **cfg_values,
            "n_clusters": n_clusters, "n_noise": n_noise,
            "v_measure": round(w["v_measure"], 4), "b3_f1": round(w["b3_f1"], 4),
            "lumping_error": round(w["lumping_error"], 4),
            "splitting_error": round(w["splitting_error"], 4),
            "v_measure_no_singletons": round(o["v_measure"], 4),
            "b3_f1_no_singletons": round(o["b3_f1"], 4),
        })

        with open(pred_dir / f"{etype.lower()}_pred_{arm}.json", "w", encoding="utf-8") as f:
            json.dump(pred, f, ensure_ascii=False, indent=2)
        coords = _umap_coords(dist)
        path = pipe.create_interactive_editor(entities, labels, None, etype,
                                              str(gui_dir), coords_2d=coords)
        written.append(path)
        print(f"    GUI -> {path}\n")

    print("Refresh the browser (served by `python -m evaluation.serve_gui`):")
    for p in written:
        rel = Path(p).relative_to(_REPO / "evaluation" / "out" / "gui")
        print(f"    http://localhost:8000/{rel}")
    print(f"\nLogged this run to {TUNE_LOG}")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="hybrid",
                    choices=["string", "embedding", "hybrid", "contextual", "contextual-hybrid"])
    ap.add_argument("--type", default="both", choices=["PER", "PEO", "LOC", "both"])
    ap.add_argument("--model", default="phenningsson/icebert-old-icelandic-v11-diplomatic")
    ap.add_argument("--text", default=str(_REPO / "alexander_saga_diplomatic.txt"))
    ap.add_argument("--note", default="",
                    help="Free-text label for this run, recorded in the comparison log. "
                         "If omitted, CLI overrides are auto-summarised into the note.")

    g = ap.add_argument_group(
        "hyperparameter overrides (default: use evaluation/cluster_config.py)")
    g.add_argument("--string-weight", type=float, default=None)
    g.add_argument("--embedding-weight", type=float, default=None)
    g.add_argument("--string-jw", type=float, default=None)
    g.add_argument("--string-lev", type=float, default=None)
    g.add_argument("--string-prefix", type=float, default=None)
    g.add_argument("--min-cluster-size", type=int, default=None)
    g.add_argument("--min-samples", type=int, default=None)
    g.add_argument("--epsilon", type=float, default=None,
                   help="CLUSTER_SELECTION_EPSILON")
    g.add_argument("--selection-method", choices=["leaf", "eom"], default=None,
                   help="CLUSTER_SELECTION_METHOD")
    g.add_argument("--max-intra-distance", type=float, default=None,
                   help="MAX_INTRA_CLUSTER_DISTANCE (hard cap; peels far members to noise)")
    args = ap.parse_args(argv)

    overrides = {cfg_key: getattr(args, cli)
                 for cli, cfg_key in _CLI_TO_CFG.items()
                 if getattr(args, cli) is not None}
    cfg = effective_config(overrides)
    note = args.note
    if not note and overrides:
        note = "cli: " + " ".join(f"{k}={v}" for k, v in overrides.items())

    etypes = ["PER", "PEO", "LOC"] if args.type == "both" else [args.type]
    run(args.arm, etypes, args.model, args.text, cfg,
        note=note, overridden=set(overrides))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
