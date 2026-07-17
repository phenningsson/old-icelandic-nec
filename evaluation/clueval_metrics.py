"""Thin adapter around the cloned CluEval tool (Kim & Kim 2023).

We use CluEval -- the established tool -- for B³ and splitting/lumping, rather than
an in-house reimplementation. CluEval's metric functions take clusters as
*lists of lists of instance IDs* (not label vectors), so this module converts our
aligned integer label vectors (from align.align_labels) into that format and calls
CluEval directly.

CluEval repo: clueval/CluEval (metrics.py: bcubed_precision_recall_fscore,
split_lump_error_precision_recall_fscore). Its split/lump is the Kim (2019)
max-overlap splitting/lumping error -- the real metric, not a B³-derived proxy.
"""

import importlib.util
from collections import defaultdict
from pathlib import Path

_CLUEVAL_DIR = Path(__file__).resolve().parent.parent / "clueval" / "CluEval"


def _load_clueval():
    """Import CluEval's metrics.py from the cloned repo (kept out of sys.path)."""
    path = _CLUEVAL_DIR / "metrics.py"
    if not path.exists():
        raise FileNotFoundError(
            f"CluEval not found at {path}. Clone it to clueval/CluEval "
            "(github.com/TEEDLab/CluEval).")
    spec = importlib.util.spec_from_file_location("clueval_metrics_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CLUEVAL = None


def _clueval():
    global _CLUEVAL
    if _CLUEVAL is None:
        _CLUEVAL = _load_clueval()
    return _CLUEVAL


def _to_cluster_lists(gold_labels, pred_labels):
    """Aligned label vectors -> (true_clusters, pred_clusters) as lists of lists of
    instance IDs. Instance ID = the item's position i (unique, hashable)."""
    gold = defaultdict(list)
    pred = defaultdict(list)
    for i, (g, p) in enumerate(zip(gold_labels, pred_labels)):
        gold[g].append(i)
        pred[p].append(i)
    return list(gold.values()), list(pred.values())


def b_cubed(gold_labels, pred_labels):
    """B³ precision/recall/F1 via CluEval."""
    true_c, pred_c = _to_cluster_lists(gold_labels, pred_labels)
    p, r, f = _clueval().bcubed_precision_recall_fscore(true_c, pred_c)
    return {"precision": float(p), "recall": float(r), "f1": float(f)}


def split_lump(gold_labels, pred_labels):
    """Splitting/lumping error via CluEval (Kim 2019).

    CluEval returns (precision, recall, f1) where precision = 1 − lumping error and
    recall = 1 − splitting error. We expose both the precision/recall/f1 and the
    error framing used in the paper's error analysis.
    """
    true_c, pred_c = _to_cluster_lists(gold_labels, pred_labels)
    p, r, f = _clueval().split_lump_error_precision_recall_fscore(true_c, pred_c)
    return {
        "precision": float(p), "recall": float(r), "f1": float(f),
        "lumping_error": 1.0 - float(p),
        "splitting_error": 1.0 - float(r),
    }
