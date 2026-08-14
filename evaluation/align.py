"""Alignment + item-set check.
"""

from dataclasses import dataclass, field


@dataclass
class ItemSetReport:
    entity_type: str
    n_gold: int
    n_pred: int
    n_shared: int
    gold_only: list = field(default_factory=list)   # forms in gold, not in pred
    pred_only: list = field(default_factory=list)   # forms in pred, not in gold

    @property
    def matches(self) -> bool:
        return not self.gold_only and not self.pred_only

    def summary(self) -> str:
        status = "OK (identical item sets)" if self.matches else "MISMATCH"
        return (f"[{self.entity_type}] {status}: "
                f"gold={self.n_gold} pred={self.n_pred} shared={self.n_shared} "
                f"gold_only={len(self.gold_only)} pred_only={len(self.pred_only)}")


def item_set_check(entity_type, gold_keys, pred_keys):
    """Compare the gold and prediction key sets. Returns an ItemSetReport.

    `gold_keys` / `pred_keys` are iterables of cased surface-form strings.
    """
    gold = set(gold_keys)
    pred = set(pred_keys)
    return ItemSetReport(
        entity_type=entity_type,
        n_gold=len(gold),
        n_pred=len(pred),
        n_shared=len(gold & pred),
        gold_only=sorted(gold - pred),
        pred_only=sorted(pred - gold),
    )


def align_labels(gold_map, pred_map):
    """Given {entity: gold_label} and {entity: pred_label} over the identical key
    set, return (forms, gold_labels, pred_labels) as parallel lists with the
    labels encoded as contiguous integers (one encoding per side).

    Raises if the key sets differ -- alignment must only run after a clean
    item-set check.
    """
    gset, pset = set(gold_map), set(pred_map)
    if gset != pset:
        raise ValueError(
            "align_labels requires identical key sets; run item_set_check first "
            f"(gold_only={len(gset - pset)}, pred_only={len(pset - gset)})")

    forms = sorted(gset)
    gold_labels = _encode([gold_map[f] for f in forms])
    pred_labels = _encode([pred_map[f] for f in forms])
    return forms, gold_labels, pred_labels


def _encode(labels):
    """Map arbitrary hashable labels to contiguous ints [0..k), stable by first
    appearance."""
    mapping = {}
    out = []
    for lab in labels:
        if lab not in mapping:
            mapping[lab] = len(mapping)
        out.append(mapping[lab])
    return out
