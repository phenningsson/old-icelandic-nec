"""Scoring orchestration.

Routes a single pair of aligned label vectors through the full policy:
- noise (-1) -> singleton (primary), plus a noise-excluded variant
- with-singletons and without-singletons (gold size >= 2) reporting
- all core + supplementary metrics
- the two degenerate baselines (all-singletons, all-in-one)
- optional, flag-gated bootstrap CIs (off by default)

Alignment always runs an item-set check first and refuses to score a mismatched
pair; callers must resolve the symmetric difference upstream.
"""

import argparse
import json
from pathlib import Path

from evaluation import gold as gold_mod
from evaluation import metrics as M
from evaluation import pipeline_io
from evaluation.align import align_labels, item_set_check

_REPO = Path(__file__).resolve().parent.parent
GOLD_FILES = {
    "PER": _REPO / "alexander_saga_persons.csv",
    "PEO": _REPO / "alexander_saga_people.csv",
    "LOC": _REPO / "alexander_saga_locations.csv",
}


def score_pair(gold_map, pred_map, with_bootstrap=False):
    """Score one {entity: gold_label} / {entity: pred_label} pair (identical keys).

    `pred_map` may contain HDBSCAN noise label -1. Returns a nested dict with the
    'with_singletons', 'without_singletons', and 'noise_excluded' arms.
    """
    rep = item_set_check("(pair)", gold_map, pred_map)
    if not rep.matches:
        raise ValueError(f"cannot score mismatched item sets: {rep.summary()}")

    n_noise = M.count_noise(pred_map)
    pred_resolved = M.noise_to_singletons(sorted(gold_map), pred_map)
    forms, gold_labels, pred_labels = align_labels(gold_map, pred_resolved)

    result = {"n_items": len(forms), "n_noise": n_noise}

    # with singletons (primary)
    result["with_singletons"] = M.all_core_metrics(gold_labels, pred_labels)
    # without singletons (gold size >= 2)
    _, g2, p2 = M.restrict_to_nonsingleton_gold(forms, gold_labels, pred_labels)
    result["without_singletons"] = (M.all_core_metrics(g2, p2) if g2 else None)
    # noise-excluded robustness variant (uses original pred_map to find noise)
    fe, ge, pe = M.exclude_noise(forms, gold_labels, pred_labels, pred_map)
    result["noise_excluded"] = (M.all_core_metrics(ge, pe) if fe else None)

    if with_bootstrap:
        result["bootstrap_ci_with_singletons"] = M.bootstrap_ci(
            gold_labels, pred_labels, M.all_core_metrics)

    return result


def score_baselines(gold_map):
    """All-singletons and all-in-one baselines on the gold item set."""
    forms = sorted(gold_map)
    return {
        "all_singletons": score_pair(gold_map, M.baseline_all_singletons(forms)),
        "all_one_cluster": score_pair(gold_map, M.baseline_all_one(forms)),
    }


def _fmt(metrics):
    if metrics is None:
        return "    (no items)"
    return ("    V={v_measure:.3f} (h={homogeneity:.3f} c={completeness:.3f})  "
            "B3 P={b3_precision:.3f} R={b3_recall:.3f} F1={b3_f1:.3f}  "
            "lump={lumping_error:.3f} split={splitting_error:.3f}").format(**metrics)


def _print_result(title, res):
    print(f"\n{title}  (n={res['n_items']}, noise={res['n_noise']})")
    print("  with singletons:");    print(_fmt(res["with_singletons"]))
    print("  without singletons:"); print(_fmt(res["without_singletons"]))
    print("  noise-excluded:");     print(_fmt(res["noise_excluded"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", default=None,
                    help="JSON {entity: label} of pipeline predictions, per type. "
                         "Provide once per entity type via --pred-per / --pred-loc.")
    ap.add_argument("--pred-per", default=None, help="PER predictions JSON {entity:label}.")
    ap.add_argument("--pred-peo", default=None, help="PEO predictions JSON {entity:label}.")
    ap.add_argument("--pred-loc", default=None, help="LOC predictions JSON {entity:label}.")
    ap.add_argument("--cluster-key", default="full", choices=["full", "leading-int"])
    ap.add_argument("--baselines", action="store_true",
                    help="Score the degenerate baselines on the gold (no model needed).")
    ap.add_argument("--bootstrap", action="store_true",
                    help="Add bootstrap CIs (OFF by default; robustness indicator only).")
    ap.add_argument("--json", default=None, help="Optional path to dump results JSON.")
    args = ap.parse_args(argv)

    pred_paths = {"PER": args.pred_per, "PEO": args.pred_peo, "LOC": args.pred_loc}
    out = {}
    for etype, gold_csv in GOLD_FILES.items():
        rows = gold_mod.load_gold(gold_csv, cluster_key=args.cluster_key)
        gold_map = gold_mod.gold_label_map(rows)
        print("=" * 72); print(etype); print("=" * 72)
        entry = {"gold_size_distribution": gold_mod.cluster_size_distribution(rows)}

        if args.baselines:
            bl = score_baselines(gold_map)
            _print_result("BASELINE all-singletons", bl["all_singletons"])
            _print_result("BASELINE all-in-one", bl["all_one_cluster"])
            entry["baselines"] = bl

        if pred_paths[etype]:
            pred_map = pipeline_io.load_pred_labels(pred_paths[etype])
            rep = item_set_check(etype, gold_map, pred_map)
            if not rep.matches:
                print(rep.summary())
                print("  -> item-set mismatch; refusing to score. Resolve upstream.")
                entry["error"] = "item_set_mismatch"
            else:
                res = score_pair(gold_map, pred_map, with_bootstrap=args.bootstrap)
                _print_result("SYSTEM", res)
                entry["system"] = res
        out[etype] = entry

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        print(f"\nWrote results to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
