"""Scoring metrics and policy.

- V-measure (+ homogeneity, completeness)   [sklearn]   -- primary (not reported or used in the paper, legacy metric, can safely ignore)
- B-cubed (B3) precision / recall / F1       [CluEval]   -- primary
- Splitting / Lumping error                  [CluEval]   -- supplementary

All metrics consume the aligned integer label vectors from align.align_labels.

B3 and split/lump come from the established CluEval tool (Kim & Kim 2023) via
`evaluation/clueval_metrics.py`. The in-house `b_cubed` below only acts as a sklearn/CluEval-independent
cross-check (validated to match CluEval on this data). CluEval's split/lump is the Kim (2019) max-overlap error.
"""

import numpy as np
from sklearn.metrics import homogeneity_completeness_v_measure

from evaluation import clueval_metrics


# Core metrics
def v_measure(gold_labels, pred_labels):
    """Homogeneity / completeness / V-measure via sklearn."""
    h, c, v = homogeneity_completeness_v_measure(gold_labels, pred_labels)
    return {"homogeneity": h, "completeness": c, "v_measure": v}


def b_cubed(gold_labels, pred_labels):
    """B-cubed precision / recall / F1 (Bagga & Baldwin 1998; Amigó et al. 2009).

    For each item i:
      precision_i = |gold(i) ∩ pred(i)| / |pred(i)|
      recall_i    = |gold(i) ∩ pred(i)| / |gold(i)|
    P, R = mean over items; F1 = 2PR/(P+R).
    """
    gold = np.asarray(gold_labels)
    pred = np.asarray(pred_labels)
    n = len(gold)
    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # group item indices by gold and by pred label
    gold_groups = _group_indices(gold)
    pred_groups = _group_indices(pred)

    precisions = np.empty(n)
    recalls = np.empty(n)
    for i in range(n):
        g = gold_groups[gold[i]]
        p = pred_groups[pred[i]]
        inter = len(g & p)
        precisions[i] = inter / len(p)
        recalls[i] = inter / len(g)

    P = float(precisions.mean())
    R = float(recalls.mean())
    F = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0
    return {"precision": P, "recall": R, "f1": F}


def all_core_metrics(gold_labels, pred_labels):
    """Every core + supplementary metric on one aligned vector pair.

    V-measure from sklearn; B3 and splitting/lumping from CluEval.
    """
    out = {}
    out.update(v_measure(gold_labels, pred_labels))
    b3 = clueval_metrics.b_cubed(gold_labels, pred_labels)
    out.update({f"b3_{k}": v for k, v in b3.items()})
    sl = clueval_metrics.split_lump(gold_labels, pred_labels)
    out.update({
        "splitlump_precision": sl["precision"],
        "splitlump_recall": sl["recall"],
        "splitlump_f1": sl["f1"],
        "lumping_error": sl["lumping_error"],
        "splitting_error": sl["splitting_error"],
    })
    return out


def _group_indices(labels):
    groups = {}
    for idx, lab in enumerate(labels):
        groups.setdefault(lab, set()).add(idx)
    return groups


# Label transforms
def noise_to_singletons(forms, pred_map, noise_label=-1):
    """Convert each HDBSCAN noise item (-1) into its own unique singleton.

    Returns a new {form: label} where every noise form gets a distinct synthetic
    label, so two noise items are never placed in the same cluster.
    """
    out = {}
    for f in forms:
        lab = pred_map[f]
        out[f] = f"__noise__{f}" if lab == noise_label else lab
    return out


def count_noise(pred_map, noise_label=-1):
    return sum(1 for v in pred_map.values() if v == noise_label)


def restrict_to_nonsingleton_gold(forms, gold_labels, pred_labels):
    """Keep only items whose GOLD cluster has size >= 2 (Cattan et al. 2020).

    Returns (forms', gold_labels', pred_labels') as filtered parallel lists.
    """
    sizes = {}
    for g in gold_labels:
        sizes[g] = sizes.get(g, 0) + 1
    keep = [i for i, g in enumerate(gold_labels) if sizes[g] >= 2]
    return ([forms[i] for i in keep],
            [gold_labels[i] for i in keep],
            [pred_labels[i] for i in keep])


def exclude_noise(forms, gold_labels, pred_labels, pred_map, noise_label=-1):
    """Drop items predicted as noise from both vectors (the noise-excluded
    robustness variant). `pred_map` is the original {form: label} (pre noise->
    singleton) used to identify which forms were noise."""
    keep = [i for i, f in enumerate(forms) if pred_map[f] != noise_label]
    return ([forms[i] for i in keep],
            [gold_labels[i] for i in keep],
            [pred_labels[i] for i in keep])


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------

def baseline_all_singletons(forms):
    """Every form is its own cluster."""
    return {f: f"__singleton__{f}" for f in forms}


def baseline_all_one(forms):
    """Every form in a single cluster."""
    return {f: "__all__" for f in forms}


# --------------------------------------------------------------------------
# Bootstrap CIs -- OPTIONAL, off by default
# --------------------------------------------------------------------------

def bootstrap_ci(gold_labels, pred_labels, metric_fn, n_resamples=1000,
                 seed=20260624, lo=2.5, hi=97.5):
    """Item-level resampling robustness indicator (not a significance test).

    Resamples item indices with replacement, recomputes `metric_fn` (which must
    return a dict of float metrics), and returns {metric: (lo_pct, hi_pct)}.
    """
    rng = np.random.default_rng(seed)
    gold = np.asarray(gold_labels)
    pred = np.asarray(pred_labels)
    n = len(gold)
    acc = {}
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        res = metric_fn(gold[idx].tolist(), pred[idx].tolist())
        for k, v in res.items():
            acc.setdefault(k, []).append(v)
    return {k: (float(np.percentile(vals, lo)), float(np.percentile(vals, hi)))
            for k, vals in acc.items()}
