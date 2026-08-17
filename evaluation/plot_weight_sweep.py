"""Code for generating the figure for the weight-sensitivity sweep (see ablation and the full paper).

Reads evaluation/out/weight_sweep.csv (+ results_all_arms.json for baselines).
Writes evaluation/out/weight_sweep.png / .pdf.
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent
OUT = _REPO / "evaluation" / "out"
WS = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]   # plotted left->right (embedding->string)
TYPES = ("PER", "PEO", "LOC")
# Reported arms only (isolation [CLS] dropped): label -> (embedding_kind, colour, marker)
ARMS = [("IceBERT", "contextual", "C0", "o"),
        ("FastText", "fasttext", "C1", "s")]


def _load():
    rows = list(csv.DictReader(open(OUT / "weight_sweep.csv", encoding="utf-8")))
    by = {(r["entity_type"], r["embedding_kind"], r["string_weight"]): r for r in rows}
    res = json.load(open(OUT / "results_all_arms.json", encoding="utf-8"))
    base = {et: res[et]["baselines"]["all_singletons"]["with_singletons"]["b3_f1"]
            for et in TYPES}
    return by, base


def make_figure(by, base):
    xs = [float(w) for w in WS]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, et in zip(axes, TYPES):
        for label, kind, colour, marker in ARMS:
            ys = [float(by[(et, kind, w)]["b3_f1"]) for w in WS]
            ax.plot(xs, ys, marker=marker, color=colour, label=label, linewidth=1.8)
        ax.axhline(base[et], color="dimgray", ls=(0, (6, 4)), lw=1.0,
                   label="all-singletons baseline")
        ax.set_title(et)
        ax.set_xlabel("string weight $w$\n($w{=}1$: string-only,  $w{=}0$: embedding-only)")
        ax.set_xticks(xs)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("$B^3$ F$_1$ (with singletons)")
    axes[0].set_ylim(0.35, 1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.06), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"weight_sweep.{ext}", dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT/'weight_sweep.png'} and .pdf")


def print_table(by):
    """B3 F1 (with singletons) per weight, for the reported arms (appendix table)."""
    print(f"\n{'type':5} {'arm':22} " + " ".join(f"w={w}" for w in WS))
    for et in TYPES:
        for label, kind, _, _ in ARMS:
            vals = " ".join(f"{float(by[(et,kind,w)]['b3_f1']):.3f}" for w in WS)
            print(f"{et:5} {label:22} {vals}")


def main():
    by, base = _load()
    print_table(by)
    make_figure(by, base)


if __name__ == "__main__":
    main()
