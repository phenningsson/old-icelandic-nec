#!/usr/bin/env python
"""
Entity Clustering v4 - Interactive Editor with Visualizations
==============================================================

Features:
- Drag & drop cluster editor
- 2D Scatter plot visualization (UMAP)
- Network graph visualization
- Source/work metadata tracking
- Export curated clusters to JSON

Usage:
    python entity_clustering_v4_visual.py
"""

import os
# Device is auto-selected (CUDA if available, else CPU); see get_embeddings.
# To force CPU, export CUDA_VISIBLE_DEVICES="" in the environment before running.

import argparse
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from collections import defaultdict

# ============================================================================
# Configuration
# ============================================================================

# Paths are resolved relative to this script so it runs from any working
# directory. The model / dataset / output can be chosen on the command line
# (see parse_args), and each CLI flag falls back to an environment variable and
# then to a sensible default.
SCRIPT_DIR = Path(__file__).resolve().parent          # .../clustering
REPO_DIR = SCRIPT_DIR.parent                          # .../west_norse_scripts

# Domain-adapted IceBERT MLM checkpoint for diplomatic Old Icelandic (public on Hugging Face).
DEFAULT_MODEL = os.environ.get("MODEL_PATH", "phenningsson/icebert-old-icelandic-v11-diplomatic")
# Input for the standalone pipeline mode: a JSON of NER-tagged entity spans. The
# evaluation harness does not use this; it clusters directly from the gold CSVs
# (see evaluation/cluster_from_gold.py). Set ENTITIES_FILE to run this file directly.
DEFAULT_ENTITIES = os.environ.get("ENTITIES_FILE", str(REPO_DIR / "entities.json"))
DEFAULT_OUTPUT = os.environ.get("OUTPUT_DIR", str(SCRIPT_DIR / "clustering_results_alexander"))

# Hybrid distance weights
STRING_WEIGHT = 0.60
EMBEDDING_WEIGHT = 0.40

# String-similarity sub-weights. The prefix term is down-weighted to 0: the error
# analysis showed prefix-sharing drove false merges of distinct names (e.g.
# Arbela/Arabia/Araxes/Armenia), and Jaro-Winkler already rewards shared prefixes.
# (Principled config motivated by the qualitative error analysis.)
STRING_SIM_JW = 0.50
STRING_SIM_LEV = 0.50
STRING_SIM_PREFIX = 0.00

# Clustering parameters
MIN_CLUSTER_SIZE = 2
MIN_SAMPLES = 1
CLUSTER_SELECTION_EPSILON = 0.0
# "eom" (excess of mass) over "leaf": the error analysis showed leaf over-fragments
# person clusters across many tiny clusters (high splitting error); eom yields
# fewer, coherent clusters.
CLUSTER_SELECTION_METHOD = "eom"

# Hard cap on within-cluster distance: after HDBSCAN, any form farther than this
# from the rest of its cluster is peeled off to noise (most-offending first), so
# non-relevant forms swept into a cluster are removed (reduces lumping). None=off.
MAX_INTRA_CLUSTER_DISTANCE = 0.40

CLUSTER_BY_TYPE = True

# ============================================================================
# String Similarity Metrics
# ============================================================================

def jaro_winkler_similarity(s1, s2):
    """Jaro-Winkler similarity."""
    try:
        from Levenshtein import jaro_winkler
        return jaro_winkler(s1, s2)
    except ImportError:
        return jaro_similarity(s1, s2)

def jaro_similarity(s1, s2):
    """Basic Jaro similarity."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    match_distance = max(len1, len2) // 2 - 1
    match_distance = max(0, match_distance)
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3
    return jaro

def normalized_levenshtein(s1, s2):
    """Normalized Levenshtein distance."""
    try:
        from Levenshtein import distance
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 0.0
        return distance(s1, s2) / max_len
    except ImportError:
        return basic_levenshtein(s1, s2) / max(len(s1), len(s2), 1)

def basic_levenshtein(s1, s2):
    """Basic Levenshtein distance."""
    if len(s1) < len(s2):
        return basic_levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def common_prefix_length(s1, s2):
    """Length of common prefix."""
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return i
    return min_len

def combined_string_similarity(s1, s2):
    """Combined string similarity on the cased surface forms.

    No lowercasing and no character normalization: the clustering unit is the
    cased surface-form type, to align with the cased gold standard. Do not reintroduce .lower().
    """
    if s1 == s2:
        return 1.0
    jw_sim = jaro_winkler_similarity(s1, s2)
    lev_dist = normalized_levenshtein(s1, s2)
    lev_sim = 1.0 - lev_dist
    prefix_len = common_prefix_length(s1, s2)
    max_len = max(len(s1), len(s2))
    prefix_ratio = prefix_len / max_len if max_len > 0 else 0
    combined = (STRING_SIM_JW * jw_sim + STRING_SIM_LEV * lev_sim
                + STRING_SIM_PREFIX * prefix_ratio)
    return combined

def string_distance(s1, s2):
    """String distance (0 = identical, 1 = completely different)."""
    return 1.0 - combined_string_similarity(s1, s2)

# ============================================================================
# Entity Loading
# ============================================================================

def load_entities(filepath):
    """Load entities from NER dataset, including source metadata (work, id)."""
    entities_by_type = defaultdict(list)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        peek = f.read(10000)
        f.seek(0)
        if peek.strip().startswith('['):
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]
    
    for entry in data:
        text = entry.get("text", "")
        ent_types = entry.get("entity_types") or entry.get("ner_tags") or []
        ent_starts = entry.get("entity_start_chars") or entry.get("starts") or []
        ent_ends = entry.get("entity_end_chars") or entry.get("ends") or []
        work = entry.get("work", "Unknown")
        doc_id = entry.get("id", None)
        
        for etype, start, end in zip(ent_types, ent_starts, ent_ends):
            # Note: Dataset uses INCLUSIVE end indices, but Python slicing is exclusive
            # So we use end+1 to get the full entity including the last character
            entity_text = text[start:end+1]
            if entity_text.strip():
                entities_by_type[etype].append({
                    "text": entity_text,
                    "start": start,
                    "end": end + 1,  # Store as exclusive for consistency
                    "context": text[max(0, start-30):end+31],
                    "work": work,
                    "doc_id": doc_id
                })
    
    return entities_by_type

def deduplicate_entities(entities):
    """Deduplicate entities, preserving all contexts and metadata."""
    seen = {}
    unique = []
    
    for ent in entities:
        # Deduplicate on the cased surface form: the clustering unit is the cased
        # surface-form type, to align with the cased gold standard. Do not lowercase here.
        key = ent["text"]
        context_obj = {
            "text": ent.get("context", ""),
            "work": ent.get("work", "Unknown"),
            "doc_id": ent.get("doc_id"),
            "start": ent.get("start"),
            "end": ent.get("end")
        }
        
        if key not in seen:
            ent["frequency"] = 1
            ent["contexts"] = [context_obj]
            ent["variants"] = {ent["text"]}
            seen[key] = len(unique)
            unique.append(ent)
        else:
            existing = unique[seen[key]]
            existing["frequency"] = existing.get("frequency", 1) + 1
            existing_contexts = existing.get("contexts", [])
            is_duplicate = any(
                c.get("text") == context_obj["text"] and c.get("work") == context_obj["work"]
                for c in existing_contexts
            )
            if not is_duplicate:
                existing_contexts.append(context_obj)
                existing["contexts"] = existing_contexts
            existing.setdefault("variants", set()).add(ent["text"])
    
    for ent in unique:
        if "variants" in ent:
            ent["variants"] = list(ent["variants"])
    
    return unique

# ============================================================================
# Embedding Computation
# ============================================================================

def get_embeddings(texts, model_path):
    """Get embeddings for texts using IceBERT."""
    from transformers import AutoTokenizer, AutoModel
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {model_path} (device={device})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()

    embeddings = []
    batch_size = 32

    print(f"Computing embeddings for {len(texts)} entities...")
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(batch_embeddings)

    return np.array(embeddings)

# ============================================================================
# Distance Matrix And Clustering
# ============================================================================

def compute_hybrid_distance_matrix(entities, embeddings):
    """Compute hybrid distance matrix."""
    n = len(entities)
    texts = [e["text"] for e in entities]
    
    print(f"Computing distance matrix for {n} entities...")
    
    string_dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = string_distance(texts[i], texts[j])
            string_dist[i, j] = d
            string_dist[j, i] = d
    
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms
    cosine_sim = np.dot(normalized, normalized.T)
    embedding_dist = 1.0 - cosine_sim
    embedding_dist = np.clip(embedding_dist, 0, 2)
    
    hybrid_dist = STRING_WEIGHT * string_dist + EMBEDDING_WEIGHT * embedding_dist
    np.fill_diagonal(hybrid_dist, 0)
    
    return hybrid_dist, string_dist

def apply_distance_cap(labels, distance_matrix, cap):
    """Peel offending members to noise (most-offending first) until every remaining
    pair within each cluster is within `cap`. Reduces lumping (see
    MAX_INTRA_CLUSTER_DISTANCE)."""
    labels = np.asarray(labels).copy()
    for cid in sorted(set(labels)):
        if cid < 0:
            continue
        idx = np.where(labels == cid)[0]
        while len(idx) > 1:
            sub = distance_matrix[np.ix_(idx, idx)]
            if sub.max() <= cap:
                break
            offender = idx[np.argmax(sub.sum(axis=1))]
            labels[offender] = -1
            idx = idx[idx != offender]
    return labels

def cluster_entities(distance_matrix):
    """Cluster entities using HDBSCAN, then apply the intra-cluster distance cap."""
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
        cluster_selection_method=CLUSTER_SELECTION_METHOD,
    )
    labels = clusterer.fit_predict(distance_matrix)
    if MAX_INTRA_CLUSTER_DISTANCE is not None:
        labels = apply_distance_cap(labels, distance_matrix, MAX_INTRA_CLUSTER_DISTANCE)
    return labels

# ============================================================================
# HTML Generation
# ============================================================================

def generate_html(entity_type, initial_data):
    """Generate the full HTML with tabs and visualizations."""
    return '''<!DOCTYPE html>
<html lang="is">
<head>
    <meta charset="utf-8">
    <title>''' + entity_type + ''' Cluster Explorer</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * { box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { margin: 0; color: #333; font-size: 24px; }
        .toolbar { display: flex; gap: 10px; }
        .btn { padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 14px; font-weight: 500; transition: all 0.2s; }
        .btn-primary { background: #4CAF50; color: white; }
        .btn-primary:hover { background: #45a049; }
        .btn-secondary { background: #2196F3; color: white; }
        .btn-secondary:hover { background: #1976D2; }
        .btn-warning { background: #ff9800; color: white; }
        .btn-warning:hover { background: #f57c00; }
        .tab-nav { display: flex; gap: 5px; margin-bottom: 15px; background: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .tab-btn { padding: 12px 24px; border: none; border-radius: 6px; cursor: pointer; font-size: 15px; font-weight: 500; background: #f0f0f0; color: #666; transition: all 0.2s; }
        .tab-btn:hover { background: #e0e0e0; }
        .tab-btn.active { background: #4CAF50; color: white; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .stats { background: #e8f5e9; padding: 10px 15px; border-radius: 5px; margin-bottom: 15px; display: flex; gap: 20px; flex-wrap: wrap; }
        .stat { font-size: 14px; }
        .stat b { color: #2e7d32; }
        .search-container { margin-bottom: 15px; }
        .search-box { width: 100%; padding: 12px; font-size: 16px; border: 2px solid #ddd; border-radius: 8px; }
        .search-box:focus { outline: none; border-color: #4CAF50; }
        .main-container { display: flex; gap: 20px; }
        .clusters-panel { flex: 3; display: flex; flex-direction: column; gap: 15px; max-height: 70vh; overflow-y: auto; }
        .noise-panel { flex: 1; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-height: 70vh; overflow-y: auto; position: sticky; top: 20px; }
        .noise-panel h3 { margin-top: 0; color: #666; }
        .cluster-card { background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid #4CAF50; }
        .cluster-card.drag-over { border-color: #2196F3; background: #e3f2fd; }
        .cluster-card.selected { border-color: #ff9800; box-shadow: 0 0 0 2px #ff9800; }
        .cluster-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #eee; }
        .cluster-title { font-weight: 600; color: #333; }
        .cluster-actions { display: flex; gap: 5px; }
        .cluster-actions button { padding: 4px 8px; font-size: 12px; border: none; border-radius: 3px; cursor: pointer; }
        .btn-select { background: #fff3e0; color: #e65100; }
        .btn-select.active { background: #ff9800; color: white; }
        .btn-delete { background: #ffebee; color: #c62828; }
        .entity-list { display: flex; flex-wrap: wrap; gap: 8px; min-height: 40px; }
        .entity { background: #e8f5e9; padding: 6px 12px; border-radius: 20px; font-size: 14px; cursor: grab; user-select: none; display: flex; align-items: center; gap: 6px; transition: all 0.2s; border: 2px solid transparent; }
        .entity:hover { background: #c8e6c9; }
        .entity.dragging { opacity: 0.5; }
        .entity .freq { background: #81c784; color: white; padding: 2px 6px; border-radius: 10px; font-size: 11px; font-weight: bold; }
        .entity .freq.high { background: #388e3c; }
        .entity .freq.very-high { background: #1b5e20; }
        .entity .works-badge { background: #e3f2fd; color: #1565c0; padding: 2px 6px; border-radius: 10px; font-size: 10px; font-weight: bold; }
        .entity .info-btn { color: #666; cursor: pointer; font-size: 12px; margin-left: 2px; padding: 0 4px; }
        .entity .info-btn:hover { color: #2196F3; }
        .entity .remove { color: #999; cursor: pointer; font-weight: bold; margin-left: 4px; }
        .entity .remove:hover { color: #f44336; }
        .noise-entity { background: #f5f5f5; border-color: #ddd; }
        .noise-entity:hover { background: #eeeeee; }
        .drop-zone { border: 2px dashed #ccc; border-radius: 8px; padding: 20px; text-align: center; color: #999; margin-top: 15px; transition: all 0.2s; }
        .drop-zone.drag-over { border-color: #4CAF50; background: #e8f5e9; color: #4CAF50; }
        .viz-container { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .viz-controls { display: flex; gap: 15px; margin-bottom: 15px; flex-wrap: wrap; align-items: center; }
        .viz-controls label { font-weight: 500; color: #555; }
        .viz-controls select, .viz-controls input[type="range"] { padding: 8px 12px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px; }
        #scatter-plot, #network-plot { width: 100%; height: 600px; }
        .viz-legend { margin-top: 15px; padding: 15px; background: #f8f9fa; border-radius: 6px; font-size: 13px; }
        .modal-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); justify-content: center; align-items: center; z-index: 1000; }
        .modal-overlay.active { display: flex; }
        .modal { background: white; padding: 25px; border-radius: 10px; max-width: 700px; width: 90%; max-height: 80vh; overflow-y: auto; }
        .modal h2 { margin-top: 0; color: #333; }
        .modal-actions { display: flex; gap: 10px; justify-content: flex-end; margin-top: 20px; }
        .context-list { max-height: 400px; overflow-y: auto; margin: 15px 0; }
        .context-item { background: #f8f9fa; padding: 12px; margin: 8px 0; border-radius: 6px; border-left: 3px solid #4CAF50; font-size: 13px; line-height: 1.5; }
        .context-source { font-size: 11px; color: #666; margin-bottom: 6px; padding-bottom: 6px; border-bottom: 1px solid #e0e0e0; font-weight: 500; }
        .context-source .work-name { color: #1976D2; font-weight: 600; }
        .context-source .doc-id { color: #888; margin-left: 8px; }
        .context-text { font-family: 'Courier New', monospace; }
        .context-item .highlight { background: #fff59d; padding: 1px 3px; border-radius: 2px; font-weight: bold; }
        .works-summary { margin: 10px 0; padding: 10px; background: #e3f2fd; border-radius: 6px; font-size: 13px; }
        .works-summary strong { color: #1565c0; }
        .entity-meta { display: grid; grid-template-columns: auto 1fr; gap: 8px 15px; margin-bottom: 15px; font-size: 14px; }
        .entity-meta dt { font-weight: 600; color: #666; }
        .entity-meta dd { margin: 0; }
        .variants-list { display: flex; flex-wrap: wrap; gap: 5px; }
        .variant-tag { background: #e3f2fd; padding: 3px 8px; border-radius: 12px; font-size: 12px; }
        .toast { position: fixed; bottom: 20px; right: 20px; background: #333; color: white; padding: 12px 24px; border-radius: 8px; opacity: 0; transition: opacity 0.3s; z-index: 1001; }
        .toast.show { opacity: 1; }
        .hidden { display: none !important; }
        .undo-stack { position: fixed; bottom: 20px; left: 20px; background: white; padding: 10px 15px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.2); font-size: 13px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 ''' + entity_type + ''' Cluster Explorer</h1>
        <div class="toolbar">
            <button class="btn btn-secondary" onclick="undo()">↶ Undo</button>
            <button class="btn btn-warning" onclick="mergeSelected()">Merge Selected</button>
            <button class="btn btn-primary" onclick="createNewCluster()">+ New Cluster</button>
            <button class="btn btn-primary" onclick="exportClusters()">💾 Export JSON</button>
        </div>
    </div>
    
    <div class="tab-nav">
        <button class="tab-btn active" onclick="showTab('editor')">📝 Cluster Editor</button>
        <button class="tab-btn" onclick="showTab('scatter')">📊 UMAP Scatter Plot</button>
    </div>
    
    <div class="stats" id="stats"></div>
    
    <div class="tab-content active" id="tab-editor">
        <div class="search-container">
            <input type="text" class="search-box" id="search" placeholder="Search entities..." onkeyup="filterEntities()">
        </div>
        <div class="main-container">
            <div class="clusters-panel" id="clusters-panel"></div>
            <div class="noise-panel">
                <h3>🔇 Unclustered (Noise)</h3>
                <div class="entity-list" id="noise-list"></div>
                <div class="drop-zone" id="noise-drop">Drop here to remove from cluster</div>
            </div>
        </div>
    </div>
    
    <div class="tab-content" id="tab-scatter">
        <div class="viz-container">
            <div class="viz-controls">
                <label>Color by:</label>
                <select id="scatter-color" onchange="updateScatterPlot()">
                    <option value="cluster">Cluster</option>
                    <option value="frequency">Frequency</option>
                    <option value="works">Number of Sources</option>
                </select>
                <label>Size by:</label>
                <select id="scatter-size" onchange="updateScatterPlot()">
                    <option value="frequency">Frequency</option>
                    <option value="fixed">Fixed</option>
                </select>
                <label><input type="checkbox" id="scatter-labels" onchange="updateScatterPlot()"> Show labels</label>
                <label><input type="checkbox" id="scatter-noise" onchange="updateScatterPlot()" checked> Show noise</label>
            </div>
            <div id="scatter-plot"></div>
            <div class="viz-legend"><strong>Tips:</strong> Hover over points for details. Click to see full entity info. Scroll to zoom, drag to pan.</div>
        </div>
    </div>
    
    <div class="modal-overlay" id="context-modal">
        <div class="modal">
            <h2 id="modal-title">Entity Details</h2>
            <dl class="entity-meta" id="entity-meta"></dl>
            <h3>Contexts (<span id="context-count">0</span>)</h3>
            <div class="context-list" id="context-list"></div>
            <div class="modal-actions">
                <button class="btn btn-secondary" onclick="closeModal('context-modal')">Close</button>
            </div>
        </div>
    </div>
    
    <div class="toast" id="toast"></div>
    <div class="undo-stack" id="undo-info">Undo stack: <span id="undo-count">0</span> actions</div>

    <script>
        let data = ''' + json.dumps(initial_data, ensure_ascii=False) + ''';
        let selectedClusters = new Set();
        let undoStack = [];
        let draggedEntity = null;
        let dragSource = null;
        
        const clusterColors = ['#e6194b','#3cb44b','#ffe119','#4363d8','#f58231','#911eb4','#42d4f4','#f032e6','#bfef45','#fabed4','#469990','#dcbeff','#9A6324','#fffac8','#800000','#aaffc3','#808000','#ffd8b1','#000075','#a9a9a9'];
        
        function getClusterColor(id) { return id < 0 ? '#cccccc' : clusterColors[id % clusterColors.length]; }
        
        function showTab(name) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelector(`[onclick="showTab('${name}')"]`).classList.add('active');
            document.getElementById('tab-' + name).classList.add('active');
            if (name === 'scatter') setTimeout(updateScatterPlot, 100);
        }
        
        function updateScatterPlot() {
            const colorBy = document.getElementById('scatter-color').value;
            const sizeBy = document.getElementById('scatter-size').value;
            const showLabels = document.getElementById('scatter-labels').checked;
            const showNoise = document.getElementById('scatter-noise').checked;
            
            let entities = data.allEntities.filter(e => showNoise || e.cluster >= 0);
            const x = entities.map(e => e.x);
            const y = entities.map(e => e.y);
            const texts = entities.map(e => e.text);
            
            let colors, colorscale, showscale = true, colorbarTitle = '';
            if (colorBy === 'cluster') { colors = entities.map(e => getClusterColor(e.cluster)); showscale = false; }
            else if (colorBy === 'frequency') { colors = entities.map(e => e.frequency); colorscale = 'Viridis'; colorbarTitle = 'Frequency'; }
            else if (colorBy === 'works') { colors = entities.map(e => e.works ? e.works.length : 1); colorscale = 'Blues'; colorbarTitle = 'Sources'; }
            
            let sizes = sizeBy === 'frequency' ? entities.map(e => Math.min(5 + Math.sqrt(e.frequency) * 4, 30)) : entities.map(() => 10);
            
            const hoverTexts = entities.map(e => `<b>${e.text}</b><br>${e.cluster >= 0 ? 'Cluster ' + e.cluster : 'Noise'}<br>Freq: ${e.frequency}<br>Sources: ${e.works ? e.works.length : 0}`);
            
            Plotly.newPlot('scatter-plot', [{
                x, y, mode: showLabels ? 'markers+text' : 'markers', type: 'scatter',
                text: texts, textposition: 'top center', textfont: { size: 9 },
                hovertext: hoverTexts, hoverinfo: 'text',
                marker: { size: sizes, color: colors, colorscale, showscale: showscale && colorBy !== 'cluster', colorbar: { title: colorbarTitle }, line: { width: 1, color: '#fff' } },
                customdata: entities.map(e => ({ id: e.id, cluster: e.cluster }))
            }], { title: "''' + entity_type + ''' Entities - UMAP", hovermode: 'closest', xaxis: { title: 'UMAP 1' }, yaxis: { title: 'UMAP 2' } });
            
            document.getElementById('scatter-plot').on('plotly_click', function(d) {
                if (d.points[0].customdata) showEntityDetails(d.points[0].customdata.id, d.points[0].customdata.cluster);
            });
        }
        
        function getTotalOccurrences() {
            let total = 0;
            data.clusters.forEach(c => c.entities.forEach(e => total += e.frequency || 1));
            data.noise.forEach(e => total += e.frequency || 1);
            return total;
        }
        
        function saveState() { undoStack.push(JSON.stringify(data)); if (undoStack.length > 50) undoStack.shift(); document.getElementById('undo-count').textContent = undoStack.length; }
        function undo() { if (undoStack.length === 0) { showToast('Nothing to undo'); return; } data = JSON.parse(undoStack.pop()); document.getElementById('undo-count').textContent = undoStack.length; render(); showToast('Undone'); }
        function showToast(msg) { const t = document.getElementById('toast'); t.textContent = msg; t.classList.add('show'); setTimeout(() => t.classList.remove('show'), 2000); }
        
        function updateStats() {
            const clustered = data.clusters.reduce((s, c) => s + c.entities.length, 0);
            document.getElementById('stats').innerHTML = `<span class="stat"><b>${data.clusters.length}</b> clusters</span><span class="stat"><b>${clustered}</b> clustered</span><span class="stat"><b>${data.noise.length}</b> noise</span><span class="stat"><b>${clustered + data.noise.length}</b> unique</span><span class="stat"><b>${getTotalOccurrences()}</b> occurrences</span><span class="stat"><b>${selectedClusters.size}</b> selected</span>`;
        }
        
        function render() { renderClusters(); renderNoise(); updateStats(); rebuildAllEntities(); }
        
        function rebuildAllEntities() {
            data.allEntities = [];
            data.clusters.forEach(c => c.entities.forEach(e => { e.cluster = c.id; data.allEntities.push(e); }));
            data.noise.forEach(e => { e.cluster = -1; data.allEntities.push(e); });
            data.allEntities.sort((a, b) => a.id - b.id);
        }
        
        function renderClusters() {
            const panel = document.getElementById('clusters-panel');
            const sorted = [...data.clusters].sort((a, b) => b.entities.length - a.entities.length);
            panel.innerHTML = sorted.map(c => {
                const freq = c.entities.reduce((s, e) => s + (e.frequency || 1), 0);
                const color = getClusterColor(c.id);
                return `<div class="cluster-card ${selectedClusters.has(c.id) ? 'selected' : ''}" style="border-left-color:${color}" data-cluster-id="${c.id}" ondragover="handleDragOver(event)" ondragleave="handleDragLeave(event)" ondrop="handleDrop(event,${c.id})">
                    <div class="cluster-header"><span class="cluster-title"><span style="display:inline-block;width:12px;height:12px;background:${color};border-radius:50%;margin-right:6px;"></span>Cluster ${c.id} (${c.entities.length} unique, ${freq} occ)</span>
                    <div class="cluster-actions"><button class="btn-select ${selectedClusters.has(c.id)?'active':''}" onclick="toggleSelect(${c.id})">${selectedClusters.has(c.id)?'✓ Selected':'Select'}</button><button class="btn-delete" onclick="deleteCluster(${c.id})">Delete</button></div></div>
                    <div class="entity-list">${c.entities.map(e => renderEntity(e, c.id)).join('')}</div></div>`;
            }).join('');
        }
        
        function renderEntity(e, cid) {
            const freq = e.frequency || 1;
            const fc = freq >= 10 ? 'very-high' : freq >= 5 ? 'high' : '';
            const works = e.works ? e.works.length : 0;
            return `<div class="entity" draggable="true" data-entity-id="${e.id}" ondragstart="handleDragStart(event,${e.id},${cid})" ondragend="handleDragEnd(event)" ondblclick="showEntityDetails(${e.id},${cid})">${e.text}<span class="freq ${fc}">×${freq}</span>${works > 1 ? `<span class="works-badge">📖${works}</span>` : ''}<span class="info-btn" onclick="event.stopPropagation();showEntityDetails(${e.id},${cid})">ℹ</span><span class="remove" onclick="event.stopPropagation();moveToNoise(${e.id},${cid})">×</span></div>`;
        }
        
        function renderNoise() {
            const list = document.getElementById('noise-list');
            list.innerHTML = data.noise.map(e => {
                const freq = e.frequency || 1;
                const fc = freq >= 10 ? 'very-high' : freq >= 5 ? 'high' : '';
                return `<div class="entity noise-entity" draggable="true" ondragstart="handleDragStart(event,${e.id},-1)" ondragend="handleDragEnd(event)" ondblclick="showEntityDetails(${e.id},-1)">${e.text}<span class="freq ${fc}">×${freq}</span><span class="info-btn" onclick="event.stopPropagation();showEntityDetails(${e.id},-1)">ℹ</span></div>`;
            }).join('');
            const dz = document.getElementById('noise-drop');
            dz.ondragover = handleDragOver;
            dz.ondragleave = handleDragLeave;
            dz.ondrop = e => handleDrop(e, -1);
        }
        
        function showEntityDetails(eid, cid) {
            let e = cid === -1 ? data.noise.find(x => x.id === eid) : (data.clusters.find(c => c.id === cid)?.entities.find(x => x.id === eid));
            if (!e) e = data.allEntities.find(x => x.id === eid);
            if (!e) return;
            document.getElementById('modal-title').textContent = e.text;
            const vars = (e.variants || [e.text]).map(v => `<span class="variant-tag">${v}</span>`).join('');
            const works = e.works || [];
            document.getElementById('entity-meta').innerHTML = `<dt>Frequency:</dt><dd><strong>${e.frequency || 1}</strong> occurrences</dd><dt>Variants:</dt><dd><div class="variants-list">${vars}</div></dd><dt>Cluster:</dt><dd>${cid >= 0 ? 'Cluster ' + cid : 'Noise'}</dd>`;
            const ctxs = e.contexts || [];
            document.getElementById('context-count').textContent = ctxs.length;
            const worksHtml = works.length > 0 ? `<div class="works-summary"><strong>${works.length}</strong> source(s): ${works.join(', ')}</div>` : '';
            document.getElementById('context-list').innerHTML = ctxs.length > 0 ? worksHtml + ctxs.map(ctx => {
                const txt = typeof ctx === 'string' ? ctx : (ctx.text || '');
                const work = typeof ctx === 'object' ? (ctx.work || 'Unknown') : 'Unknown';
                const docId = typeof ctx === 'object' ? ctx.doc_id : null;
                const hl = txt.replace(new RegExp(`(${e.text})`, 'gi'), '<span class="highlight">$1</span>');
                return `<div class="context-item"><div class="context-source">📖 <span class="work-name">${work}</span>${docId ? `<span class="doc-id">(doc #${docId})</span>` : ''}</div><div class="context-text">${hl}</div></div>`;
            }).join('') : '<p style="color:#999">No context available.</p>';
            document.getElementById('context-modal').classList.add('active');
        }
        
        function handleDragStart(ev, eid, cid) { draggedEntity = eid; dragSource = cid; ev.target.classList.add('dragging'); ev.dataTransfer.effectAllowed = 'move'; }
        function handleDragEnd(ev) { ev.target.classList.remove('dragging'); document.querySelectorAll('.drag-over').forEach(el => el.classList.remove('drag-over')); }
        function handleDragOver(ev) { ev.preventDefault(); ev.currentTarget.classList.add('drag-over'); }
        function handleDragLeave(ev) { ev.currentTarget.classList.remove('drag-over'); }
        function handleDrop(ev, tid) { ev.preventDefault(); ev.currentTarget.classList.remove('drag-over'); if (draggedEntity === null || dragSource === tid) return; saveState(); moveEntity(draggedEntity, dragSource, tid); draggedEntity = null; dragSource = null; }
        
        function moveEntity(eid, from, to) {
            let e;
            if (from === -1) { const i = data.noise.findIndex(x => x.id === eid); if (i >= 0) e = data.noise.splice(i, 1)[0]; }
            else { const c = data.clusters.find(x => x.id === from); if (c) { const i = c.entities.findIndex(x => x.id === eid); if (i >= 0) e = c.entities.splice(i, 1)[0]; } }
            if (!e) return;
            if (to === -1) data.noise.push(e); else { const c = data.clusters.find(x => x.id === to); if (c) c.entities.push(e); }
            data.clusters = data.clusters.filter(c => c.entities.length > 0);
            render(); showToast('Entity moved');
        }
        
        function moveToNoise(eid, cid) { saveState(); moveEntity(eid, cid, -1); }
        function toggleSelect(cid) { selectedClusters.has(cid) ? selectedClusters.delete(cid) : selectedClusters.add(cid); render(); }
        
        function mergeSelected() {
            if (selectedClusters.size < 2) { showToast('Select at least 2 clusters'); return; }
            saveState();
            const ids = Array.from(selectedClusters);
            const tid = Math.min(...ids);
            const tc = data.clusters.find(c => c.id === tid);
            ids.forEach(id => { if (id !== tid) { const s = data.clusters.find(c => c.id === id); if (s) tc.entities.push(...s.entities); } });
            data.clusters = data.clusters.filter(c => !ids.includes(c.id) || c.id === tid);
            selectedClusters.clear(); render(); showToast(`Merged ${ids.length} clusters`);
        }
        
        function createNewCluster() { saveState(); const nid = data.nextClusterId++; data.clusters.push({ id: nid, entities: [] }); render(); showToast(`Created cluster ${nid}`); }
        
        function deleteCluster(cid) {
            const c = data.clusters.find(x => x.id === cid);
            if (!c) return;
            if (c.entities.length > 0 && !confirm(`Move ${c.entities.length} entities to noise?`)) return;
            saveState(); data.noise.push(...c.entities); data.clusters = data.clusters.filter(x => x.id !== cid); selectedClusters.delete(cid); render(); showToast('Cluster deleted');
        }
        
        function filterEntities() {
            const s = document.getElementById('search').value.toLowerCase();
            document.querySelectorAll('.entity').forEach(el => el.classList.toggle('hidden', !el.textContent.toLowerCase().includes(s)));
            document.querySelectorAll('.cluster-card').forEach(c => c.classList.toggle('hidden', c.querySelectorAll('.entity:not(.hidden)').length === 0 && s !== ''));
        }
        
        function closeModal(id) { document.getElementById(id).classList.remove('active'); }
        
        function exportClusters() {
            const exp = {
                entityType: data.entityType, exportedAt: new Date().toISOString(),
                clusters: data.clusters.map(c => ({ id: c.id, entities: c.entities.map(e => ({ text: e.text, frequency: e.frequency || 1, variants: e.variants || [e.text], works: e.works || [], contexts: (e.contexts || []).map(ctx => ({ text: typeof ctx === 'string' ? ctx : ctx.text, work: typeof ctx === 'object' ? ctx.work : 'Unknown', doc_id: typeof ctx === 'object' ? ctx.doc_id : null })) })) })),
                noise: data.noise.map(e => ({ text: e.text, frequency: e.frequency || 1, variants: e.variants || [e.text], works: e.works || [], contexts: (e.contexts || []).map(ctx => ({ text: typeof ctx === 'string' ? ctx : ctx.text, work: typeof ctx === 'object' ? ctx.work : 'Unknown', doc_id: typeof ctx === 'object' ? ctx.doc_id : null })) })),
                statistics: { totalClusters: data.clusters.length, totalClustered: data.clusters.reduce((s, c) => s + c.entities.length, 0), totalNoise: data.noise.length, totalOccurrences: getTotalOccurrences() }
            };
            const blob = new Blob([JSON.stringify(exp, null, 2)], {type: 'application/json'});
            const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `${data.entityType}_clusters_curated.json`; a.click();
            showToast('Exported');
        }
        
        document.addEventListener('keydown', e => { if (e.key === 'Escape') document.querySelectorAll('.modal-overlay.active').forEach(m => m.classList.remove('active')); });
        document.querySelectorAll('.modal-overlay').forEach(o => o.addEventListener('click', e => { if (e.target === o) o.classList.remove('active'); }));
        
        render();
    </script>
</body>
</html>'''

# ============================================================================
# Interactive Editor
# ============================================================================

def create_interactive_editor(entities, labels, embeddings, entity_type, output_dir,
                              coords_2d=None):
    """Create interactive HTML editor with UMAP visualization.

    `coords_2d` (optional Nx2 array) lets a caller supply a precomputed 2D layout
    (e.g. UMAP on a precomputed distance matrix, so the scatter reflects the actual
    clustering distance). If None, UMAP is run on `embeddings` with cosine metric.
    """
    if coords_2d is None:
        print("  Computing UMAP projection...")
        import umap

        n_neighbors = min(15, len(entities) - 1)
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, metric="cosine", random_state=42)
        coords_2d = reducer.fit_transform(embeddings)

    clusters = defaultdict(list)
    noise = []
    
    for i, (ent, label) in enumerate(zip(entities, labels)):
        contexts = ent.get("contexts", [])[:20]
        variants = ent.get("variants", [ent["text"]])
        works = list(set(c.get("work", "Unknown") for c in contexts if c.get("work")))
        
        ent_data = {
            "id": i, "text": ent["text"], "contexts": contexts, "variants": variants,
            "works": works, "frequency": ent.get("frequency", 1), "numSources": len(contexts),
            "x": float(coords_2d[i, 0]), "y": float(coords_2d[i, 1]), "cluster": int(label)
        }
        if label >= 0:
            clusters[int(label)].append(ent_data)
        else:
            noise.append(ent_data)
    
    clusters_list = [{"id": cid, "entities": ents} for cid, ents in sorted(clusters.items())]
    
    all_entities = []
    for cluster in clusters_list:
        all_entities.extend(cluster["entities"])
    all_entities.extend(noise)
    all_entities.sort(key=lambda e: e["id"])
    
    initial_data = {
        "entityType": entity_type, "clusters": clusters_list, "noise": noise,
        "nextClusterId": max(clusters.keys()) + 1 if clusters else 0,
        "allEntities": all_entities
    }
    
    html = generate_html(entity_type, initial_data)
    
    output_path = Path(output_dir) / f"{entity_type.lower()}_cluster_explorer.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_path

# ============================================================================
# MAIN
# ============================================================================

def resolve_entities_path(value):
    """Resolve a dataset argument: use it as-is if it exists, otherwise treat it
    as a filename relative to the repo root."""
    path = Path(value)
    if path.exists():
        return path
    repo_relative = REPO_DIR / value
    if repo_relative.exists():
        return repo_relative
    return path  # let the caller raise a clear FileNotFoundError


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Entity clustering pipeline for Old Icelandic NER spans.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Defaults: domain-adapted IceBERT v11-diplomatic MLM on the diplomatic
  # Alexanders saga dataset (the evaluation setup). Requires authenticated HF
  # access to the private model repo (HF_TOKEN / huggingface-cli login).
  python clustering/clustering_v5.py

  # A different dataset / a local model checkpoint
  python clustering/clustering_v5.py \\
      --model /path/to/local/checkpoint \\
      --entities some_other_flat_ner_dataset.json \\
      --output clustering/clustering_results_other
""",
    )
    parser.add_argument(
        "-m", "--model", default=DEFAULT_MODEL,
        help="Embedding model: a local path or a HuggingFace model id. Default is "
             "the domain-adapted diplomatic IceBERT MLM "
             "'phenningsson/icebert-old-icelandic-v11-diplomatic' (private HF repo; "
             "needs authenticated access). (default: %(default)s)",
    )
    parser.add_argument(
        "-e", "--entities", default=DEFAULT_ENTITIES,
        help="NER dataset JSON to cluster. May be an absolute/relative path or a "
             "filename resolved against the repo root "
             "(e.g. alexander_diplomatic_flat_ner_dataset.json). (default: %(default)s)",
    )
    parser.add_argument(
        "-o", "--output", default=DEFAULT_OUTPUT,
        help="Directory for the generated HTML explorers. Use a fresh directory "
             "when changing model/dataset so you do not overwrite the committed "
             "reference output. (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    model_path = args.model
    entities_file = resolve_entities_path(args.entities)
    output_dir = Path(args.output)

    print("="*70)
    print("Entity Clustering v4 - Interactive Editor with Visualizations")
    print("="*70)
    print(f"\nModel:   {model_path}")
    print(f"Dataset: {entities_file}")
    print(f"Output:  {output_dir}")
    print(f"\nString weight: {STRING_WEIGHT}, Embedding weight: {EMBEDDING_WEIGHT}\n")

    if not entities_file.exists():
        raise FileNotFoundError(f"Dataset not found: {entities_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading entities from {entities_file}...")
    entities_by_type = load_entities(entities_file)
    print(f"Loaded {sum(len(v) for v in entities_by_type.values())} entities")

    for etype, entities in entities_by_type.items():
        print(f"\n{'='*70}\nProcessing {etype}\n{'='*70}")

        entities = deduplicate_entities(entities)
        print(f"After deduplication: {len(entities)}")

        if len(entities) < MIN_CLUSTER_SIZE:
            print("Skipping - too few entities")
            continue

        texts = [e["text"] for e in entities]
        embeddings = get_embeddings(texts, model_path)
        hybrid_dist, string_dist = compute_hybrid_distance_matrix(entities, embeddings)

        print("Clustering...")
        labels = cluster_entities(hybrid_dist)
        n_clusters = len(set(l for l in labels if l >= 0))
        n_noise = sum(1 for l in labels if l < 0)
        print(f"Found {n_clusters} clusters, {n_noise} noise")

        editor_path = create_interactive_editor(entities, labels, embeddings, etype, output_dir)
        print(f"Created: {editor_path}")

    print(f"\n{'='*70}\nCOMPLETE\n{'='*70}")
    print(f"\nOutput: {output_dir}")
    print("Open the *_cluster_explorer.html files in a browser.")

if __name__ == "__main__":
    main()