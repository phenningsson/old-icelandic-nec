"""Prediction-side I/O: load predicted cluster labels.

Scoring needs predicted cluster labels per cased surface-form type. The clusterer
writes a {entity: label} JSON (see evaluation/cluster_from_gold.py); `-1` (HDBSCAN
noise) is kept as-is here and converted to singletons at scoring time (metrics).
"""

import json


def load_pred_labels(json_path):
    """Load predicted labels: a JSON object {entity: label}. HDBSCAN noise stays
    as -1 here; noise->singleton conversion happens at scoring time (metrics)."""
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)
    return {str(k): v for k, v in data.items()}
