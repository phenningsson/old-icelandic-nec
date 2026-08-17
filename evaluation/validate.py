"""Validation; verifies the evaluation is sound and reproducible.

Run after (re)generating predictions/results. Exits non-zero if any check fails,
so it can gate a commit / CI. Checks:

  1. Item-set match: gold and prediction cover the identical form set for every arm.
  2. Metric agreement: the in-house B³ matches CluEval's B³ to < 1e-9 (the two
     independent implementations corroborate each other).
  3. Encoding: a known medieval form (`alexanꝺer`) loads and scores
     with its bytes intact.
  4. Baseline sanity: all-singletons → B³ precision 1.0; all-in-one → B³ recall 1.0.
  5. Gold reproduces: 174 PER / 18 PEO / 76 LOC scoring clusters from the CSVs.

Usage:  python -m evaluation.validate
"""

import sys
from pathlib import Path

from evaluation import gold as gm
from evaluation import pipeline_io
from evaluation import metrics as M
from evaluation import clueval_metrics as CE
from evaluation import evaluate as E
from evaluation.align import align_labels, item_set_check

_REPO = Path(__file__).resolve().parent.parent
GOLD = {"PER": _REPO / "alexander_saga_persons.csv",
        "PEO": _REPO / "alexander_saga_people.csv",
        "LOC": _REPO / "alexander_saga_locations.csv"}
PRED_DIR = _REPO / "evaluation" / "out" / "predictions"
ARMS = ["string", "embedding", "hybrid", "contextual", "contextual-hybrid",
        "fasttext", "fasttext-hybrid"]


def _check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    return ok


def run():
    ok = True
    print("1. Item-set match (gold == prediction) for every available arm:")
    for et in GOLD:
        gmap = gm.gold_label_map(gm.load_gold(GOLD[et]))
        for arm in ARMS:
            pf = PRED_DIR / f"{et.lower()}_pred_{arm}.json"
            if not pf.exists():
                continue
            rep = item_set_check(et, gmap, pipeline_io.load_pred_labels(pf))
            ok &= _check(f"{et}/{arm}", rep.matches,
                         f"gold={rep.n_gold} pred={rep.n_pred} "
                         f"+{len(rep.gold_only)}/-{len(rep.pred_only)}")

    print("2. In-house B³ == CluEval B³ (cross-implementation check):")
    for et in GOLD:
        gmap = gm.gold_label_map(gm.load_gold(GOLD[et]))
        pmap = pipeline_io.load_pred_labels(PRED_DIR / f"{et.lower()}_pred_string.json")
        _, gl, pl = align_labels(gmap, M.noise_to_singletons(sorted(gmap), pmap))
        ih, ce = M.b_cubed(gl, pl), CE.b_cubed(gl, pl)
        diff = max(abs(ih[k] - ce[k]) for k in ih)
        ok &= _check(f"{et} B³ agreement", diff < 1e-9, f"max|Δ|={diff:.1e}")

    print("3. Encoding check:")
    rows = gm.load_gold(GOLD["PER"])
    hit = [r for r in rows if r.entity == "alexanꝺer"]
    ok &= _check("'alexanꝺer' intact", len(hit) == 1,
                 f"cluster={hit[0].cluster_label!r}" if hit else "missing")

    print("4. Baseline sanity:")
    for et in GOLD:
        gmap = gm.gold_label_map(gm.load_gold(GOLD[et]))
        bl = E.score_baselines(gmap)
        p = bl["all_singletons"]["with_singletons"]["b3_precision"]
        r = bl["all_one_cluster"]["with_singletons"]["b3_recall"]
        ok &= _check(f"{et} baselines", p > 0.999 and r > 0.999,
                     f"all-1s P={p:.3f}, all-one R={r:.3f}")

    print("5. Gold counts reproduce:")
    for et, want in (("PER", 174), ("PEO", 18), ("LOC", 76)):
        n = gm.cluster_size_distribution(gm.load_gold(GOLD[et]))["n_clusters"]
        ok &= _check(f"{et} clusters == {want}", n == want, f"got {n}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "some CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
