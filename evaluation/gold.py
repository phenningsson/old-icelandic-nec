"""Load the gold-standard clustering CSVs.

Schema (all files): entity, frequency, cluster_id, subcluster_id, Notes

The clustering unit is the surface-form `entity` (no lowercasing). Each row
maps one form to one gold cluster.

Cluster key
-----------
The cluster key is the normalised full `cluster_id` string (e.g. "1 Alexander"):
each distinct label the annotator wrote is one cluster. Whitespace is normalized
(strip + collapse internal runs) for uniformity. A trailing `/`-suffix marks the annotator's
sub-variant naming convention for one cluster (e.g. "7 Saracen" / "7 Saracen/serkr"),
so it is collapsed to the base label before the first `/`. The normalised strings
are then factorised to contiguous integer ids at scoring time (align._encode).
The leading integer alone is not a safe key: some integers are mistakenly reused across
different names, e.g. "17 Cebalinus" vs "17 Zoroas".

Special rows:
- Blank `cluster_id` (`maceꝺo`) -> its own synthetic singleton cluster.
- Empty `entity` -> skipped.
"""

import csv
import re
from collections import defaultdict
from dataclasses import dataclass


def normalize_cluster_id(cluster_id: str) -> str:
    """Strip and collapse internal whitespace runs to a single space."""
    return re.sub(r"\s+", " ", cluster_id).strip()


@dataclass
class GoldRow:
    entity: str
    frequency: int
    cluster_id_full: str          # e.g. "1 Alexander" ("" if blank in source)
    cluster_label: str            # the resolved cluster key used for scoring
    leading_int: str              # leading integer of cluster_id_full ("" if blank)
    subcluster_id: str            # Norse-World sub-level; not scored
    notes: str                    # annotator notes; reporting strata only


def _resolve_label(cluster_id_full: str, entity: str, cluster_key: str) -> str:
    """Resolve the scoring cluster key for one row.

    Blank cluster_id -> a unique synthetic singleton label so it scores as a
    singleton and never collides with another row.
    """
    cid = normalize_cluster_id(cluster_id_full)
    if not cid:
        return f"__singleton__{entity}"
    if cluster_key == "leading-int":
        return cid.split()[0]
    if cluster_key == "full":
        # A trailing slash sub-tag (e.g. "7 Saracen/serkr", "41 Indian/indímaðr") is
        # the annotator's naming convention for sub-variants of one cluster, not a
        # marker for separate clusters (ruled 2026-06-25). Collapse to the base label
        # before the first '/'. This merges the 2 slash super-clusters (7, 41 — both
        # now PEO/ethnonym clusters) while leaving reused-integer/different-name cases
        # (e.g. "17 Cebalinus" vs "17 Zoroas") correctly separate.
        # Scoring clusters: 174 PER / 18 PEO / 76 LOC.
        return cid.split("/")[0].strip()
    raise ValueError(f"unknown cluster_key: {cluster_key!r}")


def load_gold(csv_path, cluster_key: str = "full"):
    """Load a gold CSV into a list of GoldRow.

    Returns the rows in file order (skipping empty-entity rows). Callers that
    want the {entity: label} mapping should use `gold_label_map`.
    """
    rows = []
    with open(csv_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            entity = (raw.get("entity") or "").strip()
            if not entity:
                continue  # defensive: skip empty-entity rows
            cluster_id_full = normalize_cluster_id(raw.get("cluster_id") or "")
            leading_int = cluster_id_full.split()[0] if cluster_id_full else ""
            freq_raw = (raw.get("frequency") or "").strip()
            rows.append(GoldRow(
                entity=entity,
                frequency=int(freq_raw) if freq_raw.isdigit() else 0,
                cluster_id_full=cluster_id_full,
                cluster_label=_resolve_label(cluster_id_full, entity, cluster_key),
                leading_int=leading_int,
                subcluster_id=(raw.get("subcluster_id") or "").strip(),
                notes=(raw.get("Notes") or "").strip(),
            ))
    return rows


def gold_label_map(rows):
    """Build {entity: cluster_label}. Raises if a entity appears twice
    (the gold is expected to have one row per type)."""
    out = {}
    for r in rows:
        if r.entity in out and out[r.entity] != r.cluster_label:
            raise ValueError(
                f"duplicate gold entity {r.entity!r} with conflicting labels "
                f"{out[r.entity]!r} vs {r.cluster_label!r}")
        out[r.entity] = r.cluster_label
    return out


def cluster_size_distribution(rows):
    """Gold cluster-size distribution (counts of size 1, 2, 3-5, 6-10, >10),
    plus singleton count/share."""
    sizes = defaultdict(int)
    for r in rows:
        sizes[r.cluster_label] += 1
    buckets = {"1": 0, "2": 0, "3-5": 0, "6-10": 0, ">10": 0}
    for n in sizes.values():
        if n == 1:
            buckets["1"] += 1
        elif n == 2:
            buckets["2"] += 1
        elif n <= 5:
            buckets["3-5"] += 1
        elif n <= 10:
            buckets["6-10"] += 1
        else:
            buckets[">10"] += 1
    n_clusters = len(sizes)
    n_singletons = buckets["1"]
    return {
        "n_items": len(rows),
        "n_clusters": n_clusters,
        "n_singletons": n_singletons,
        "singleton_share": (n_singletons / n_clusters) if n_clusters else 0.0,
        "max_size": max(sizes.values()) if sizes else 0,
        "buckets": buckets,
    }


def cluster_id_anomalies(rows):
    """Surface cluster_id quirks that affect (or could affect) the cluster key.

    Reports, without auto-resolving, the leading integers mapping to >1 distinct
    normalised label (why leading-int is unsafe), split into:
      * `slash_variants`: same base name, slash-suffixed sub-labels 
      * `distinct_names`: genuinely different names sharing a number,
        kept separate by the full-string key.
    """
    by_int = defaultdict(set)
    for r in rows:
        if r.cluster_id_full:
            by_int[r.leading_int].add(r.cluster_id_full)
    reused = {k: sorted(v) for k, v in by_int.items() if len(v) > 1}

    slash, distinct = {}, {}
    for k, labels in reused.items():
        bases = {lbl.split("/")[0].strip().casefold() for lbl in labels}
        (slash if len(bases) == 1 else distinct)[k] = labels

    return {
        "leading_int_reused": reused,
        "slash_variants": slash,
        "distinct_names": distinct,
    }
