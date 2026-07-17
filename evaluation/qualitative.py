"""Qualitative / philological tier.

Necessary-but-not-sufficient: the metrics need a per-cluster error view for the
philological analysis. For a chosen arm (default: the proposed hybrid), per entity
type, this exports:

  1. Per-GOLD-cluster diff: each gold cluster, its forms, and which predicted
     clusters they landed in -> the splitting view (a gold cluster split across >1
     predicted cluster is fragmented).
  2. Per-PRED-cluster diff: each predicted cluster, its forms, and which gold
     clusters it mixes -> the LUMPING view (a predicted cluster spanning >1 gold
     cluster over-merges distinct entities).
  3. A readable error-analysis report: the worst splits and lumps, with forms.
  4. Strata: uncertainty ("Osäker...") and ethnonym ("folkslagsbeteckning") rows
     from gold Notes, with metrics on the confident / name-only subsets so a
     metric-"wrong" but philologically-defensible case can be set aside.

HDBSCAN noise (-1) is resolved to per-form singletons (the scoring policy), and
flagged as noise in the output.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from evaluation import gold as gold_mod
from evaluation import metrics as M
from evaluation import pipeline_io
from evaluation.align import align_labels

_REPO = Path(__file__).resolve().parent.parent
GOLD = {"PER": _REPO / "alexander_saga_persons.csv",
        "PEO": _REPO / "alexander_saga_people.csv",
        "LOC": _REPO / "alexander_saga_locations.csv"}
PRED_DIR = _REPO / "evaluation" / "out" / "predictions"
OUT = _REPO / "evaluation" / "out" / "qualitative"


def _strata_flags(notes):
    n = (notes or "").lower()
    return {"uncertain": "osäk" in n, "ethnonym": "folkslag" in n}


def _resolve_pred(form, label):
    """noise (-1) -> a unique singleton id, flagged."""
    return (f"noise:{form}", True) if label == -1 else (f"c{label}", False)


def _gold_cluster_diff(rows, pred_map):
    """Per gold cluster: forms + predicted-cluster breakdown (splitting view)."""
    by_gold = defaultdict(list)
    for r in rows:
        by_gold[r.cluster_label].append(r)
    out = []
    for label, members in by_gold.items():
        preds = []
        for m in members:
            pid, is_noise = _resolve_pred(m.entity, pred_map[m.entity])
            preds.append(pid)
        breakdown = Counter(preds)
        n_pred = len(breakdown)
        n_noise = sum(1 for p in preds if p.startswith("noise:"))
        out.append({
            "gold_cluster": label, "size": len(members),
            "n_pred_clusters": n_pred, "is_split": n_pred > 1,
            "n_to_noise": n_noise,
            "pred_breakdown": "; ".join(f"{k}×{v}" for k, v in breakdown.most_common()),
            "forms": " | ".join(m.entity for m in members),
        })
    out.sort(key=lambda d: (-d["n_pred_clusters"], -d["size"]))
    return out


def _pred_cluster_diff(rows, pred_map):
    """Per predicted cluster: forms + gold-cluster breakdown (lumping view)."""
    gold_of = {r.entity: r.cluster_label for r in rows}
    by_pred = defaultdict(list)
    noise_forms = set()
    for r in rows:
        pid, is_noise = _resolve_pred(r.entity, pred_map[r.entity])
        by_pred[pid].append(r.entity)
        if is_noise:
            noise_forms.add(r.entity)
    out = []
    for pid, forms in by_pred.items():
        golds = Counter(gold_of[f] for f in forms)
        out.append({
            "pred_cluster": pid, "size": len(forms),
            "is_noise_singleton": pid.startswith("noise:"),
            "n_gold_clusters": len(golds), "is_lumped": len(golds) > 1,
            "gold_breakdown": "; ".join(f"{k}×{v}" for k, v in golds.most_common()),
            "forms": " | ".join(forms),
        })
    out.sort(key=lambda d: (-d["n_gold_clusters"], -d["size"]))
    return out


def _strata_metrics(rows, pred_map):
    """Score V/B3 on all items and on confident / name-only / confident-name subsets."""
    gmap = {r.entity: r.cluster_label for r in rows}
    pres = M.noise_to_singletons(sorted(gmap), pred_map)
    forms, gold_labels, pred_labels = align_labels(gmap, pres)
    flags = {r.entity: _strata_flags(r.notes) for r in rows}

    def score(keep_idx):
        if not keep_idx:
            return None
        g = [gold_labels[i] for i in keep_idx]
        p = [pred_labels[i] for i in keep_idx]
        m = M.all_core_metrics(g, p)
        return {"n": len(keep_idx), "v_measure": m["v_measure"], "b3_f1": m["b3_f1"],
                "lumping_error": m["lumping_error"], "splitting_error": m["splitting_error"]}

    idx_all = list(range(len(forms)))
    idx_conf = [i for i in idx_all if not flags[forms[i]]["uncertain"]]
    idx_name = [i for i in idx_all if not flags[forms[i]]["ethnonym"]]
    idx_cn = [i for i in idx_all if not flags[forms[i]]["uncertain"]
              and not flags[forms[i]]["ethnonym"]]
    return {
        "all": score(idx_all),
        "confident_only": score(idx_conf),
        "names_only": score(idx_name),
        "confident_names": score(idx_cn),
        "n_uncertain": sum(1 for f in forms if flags[f]["uncertain"]),
        "n_ethnonym": sum(1 for f in forms if flags[f]["ethnonym"]),
    }


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def _error_report(etype, gdiff, pdiff, strata, rows):
    lines = [f"# {etype} error analysis (arm-specific)", ""]
    splits = [d for d in gdiff if d["is_split"]]
    lines.append(f"## Splitting: {len(splits)} of {len(gdiff)} gold clusters fragmented")
    for d in splits[:15]:
        lines.append(f"  [{d['gold_cluster']}] size {d['size']} -> {d['n_pred_clusters']} "
                     f"pred clusters ({d['n_to_noise']} to noise): {d['pred_breakdown']}")
        lines.append(f"      forms: {d['forms']}")
    lumps = [d for d in pdiff if d["is_lumped"]]
    lines.append("")
    lines.append(f"## Lumping: {len(lumps)} predicted clusters merge >1 gold cluster")
    for d in lumps[:15]:
        lines.append(f"  [{d['pred_cluster']}] size {d['size']} mixes {d['n_gold_clusters']} "
                     f"gold clusters: {d['gold_breakdown']}")
        lines.append(f"      forms: {d['forms']}")
    lines.append("")
    lines.append("## Strata (metrics V / B3 F1):")
    for k in ("all", "confident_only", "names_only", "confident_names"):
        s = strata[k]
        if s:
            lines.append(f"  {k:18} n={s['n']:4d}  V={s['v_measure']:.3f}  "
                         f"B3F1={s['b3_f1']:.3f}  lump={s['lumping_error']:.3f} "
                         f"split={s['splitting_error']:.3f}")
    lines.append(f"  (uncertain items: {strata['n_uncertain']}, "
                 f"ethnonym items: {strata['n_ethnonym']})")
    # list the ethnonym clusters explicitly
    eth = sorted({r.cluster_label for r in rows if _strata_flags(r.notes)["ethnonym"]})
    unc = [(r.entity, r.cluster_label) for r in rows if _strata_flags(r.notes)["uncertain"]]
    lines.append("")
    lines.append(f"## Ethnonym clusters ({len(eth)}): {', '.join(eth)}")
    lines.append(f"## Uncertain items ({len(unc)}): "
                 + ", ".join(f"{e}({c})" for e, c in unc[:30]))
    return "\n".join(lines)


def run(arm, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for etype in ("PER", "PEO", "LOC"):
        rows = gold_mod.load_gold(GOLD[etype])
        pred_map = pipeline_io.load_pred_labels(PRED_DIR / f"{etype.lower()}_pred_{arm}.json")
        gdiff = _gold_cluster_diff(rows, pred_map)
        pdiff = _pred_cluster_diff(rows, pred_map)
        strata = _strata_metrics(rows, pred_map)

        _write_csv(out_dir / f"{etype}_gold_cluster_diff_{arm}.csv", gdiff)
        _write_csv(out_dir / f"{etype}_pred_cluster_diff_{arm}.csv", pdiff)
        report = _error_report(etype, gdiff, pdiff, strata, rows)
        (out_dir / f"{etype}_error_analysis_{arm}.txt").write_text(report, encoding="utf-8")
        summary[etype] = {
            "n_gold_clusters": len(gdiff),
            "n_split": sum(1 for d in gdiff if d["is_split"]),
            "n_pred_clusters_nontrivial": sum(1 for d in pdiff if not d["is_noise_singleton"]),
            "n_lumped": sum(1 for d in pdiff if d["is_lumped"]),
            "strata": strata,
        }
        print(report)
        print()
    with open(out_dir / f"strata_summary_{arm}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote per-cluster diffs + error reports + strata to {out_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="hybrid",
                    choices=["string", "embedding", "hybrid", "contextual", "contextual-hybrid"],
                    help="Which prediction arm to analyse (default: hybrid = proposed system).")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    run(args.arm, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
