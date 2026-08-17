"""Cluster every arm from the gold forms and score,
writing evaluation/out/results_all_arms.json (the results for every arm and type).

For each entity type (PER = personal names, PEO = ethnonyms, LOC = place names) and each
distance arm, this clusters directly from the gold surface forms
(evaluation.cluster_from_gold). The predicted and gold item sets match by
construction, and scores under the full policy (evaluation.evaluate.score_pair):
noise→singleton, with/without singletons, baselines.

Per-type, per-arm predictions are cached to evaluation/out/predictions/{et}_pred_{arm}.json

Run:  python -m evaluation.run_all_arms            # all arms, all types
      python -m evaluation.run_all_arms --arms string   # subset, e.g., only runs the string
"""

import argparse
import json
from pathlib import Path

from evaluation import gold as gold_mod
from evaluation import evaluate as E
from evaluation.cluster_from_gold import (
    GOLD_FILES, ETYPES, _load_pipeline, gold_entities, cluster_forms, load_raw_text,
)

_REPO = Path(__file__).resolve().parent.parent
OUT = _REPO / "evaluation" / "out"
PRED_DIR = OUT / "predictions"
DEFAULT_TEXT = _REPO / "alexander_saga_diplomatic.txt"   # raw text (contextual arm only)
ALL_ARMS = ["string", "embedding", "hybrid", "contextual", "contextual-hybrid",
            "fasttext", "fasttext-hybrid"]


def _score_type(label, entities, gold_map, rows, arms, model, pipe, raw_text):
    """Cluster every arm for one item-set and assemble its results block."""
    entry = {"gold_size_distribution": gold_mod.cluster_size_distribution(rows),
             "baselines": E.score_baselines(gold_map),
             "arms": {}}
    for arm in arms:
        labels = cluster_forms(entities, arm, model, pipe, raw_text=raw_text)
        pred = {e["text"]: int(l) for e, l in zip(entities, labels)}
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        with open(PRED_DIR / f"{label.lower()}_pred_{arm}.json", "w",
                  encoding="utf-8") as fh:
            json.dump(pred, fh, ensure_ascii=False, indent=2)
        res = E.score_pair(gold_map, pred)
        entry["arms"][arm] = res
        m = res["with_singletons"]
        print(f"  [{label}] {arm:18} B³F1={m['b3_f1']:.3f} "
              f"V={m['v_measure']:.3f} noise={res['n_noise']}")
    return entry


def run(arms=None, model="phenningsson/icebert-old-icelandic-v11-diplomatic",
        text_path=DEFAULT_TEXT, out_path=None):
    arms = arms or ALL_ARMS
    pipe = _load_pipeline()
    raw_text = load_raw_text(text_path) if any("contextual" in a for a in arms) else None
    results = {}

    for et in ETYPES:
        print("=" * 72); print(et); print("=" * 72)
        entities = gold_entities(GOLD_FILES[et])
        rows = gold_mod.load_gold(GOLD_FILES[et])
        gold_map = gold_mod.gold_label_map(rows)
        results[et] = _score_type(et, entities, gold_map, rows, arms, model, pipe, raw_text)

    out_path = out_path or (OUT / "results_all_arms.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path}")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default=",".join(ALL_ARMS),
                    help="comma-separated arms (default: all 7).")
    ap.add_argument("--model", default="phenningsson/icebert-old-icelandic-v11-diplomatic")
    ap.add_argument("--text", default=str(DEFAULT_TEXT),
                    help="raw diplomatic text file (used by the contextual arm only).")
    ap.add_argument("--out", default=str(OUT / "results_all_arms.json"))
    args = ap.parse_args(argv)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    run(arms, args.model, args.text, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
