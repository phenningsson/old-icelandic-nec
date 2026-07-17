# Old Icelandic Named Entity Clustering

Automated **Named Entity Clustering (NEC)** for diplomatic Old Icelandic that groups the
spelling and inflectional variants of one name into a single cluster (e.g., `alexanꝺer`,
`alexanꝺro`, `alexanꝺrum` → *Alexander*). Code and data accompanying the paper **comming soon**, evaluated on a domain expert philologist's clustering of the diplomatic *Alexanders saga* (**AM 519 a 4to**).

Please note that the task is *variant clustering*, i.e., grouping surface forms of the same name. It is **not** entity linking (no external knowledge base), **not** coreference (referent is irrelevant, it doesn't matter *which* Gunnar it is, just the name variations *of* Gunnar), and **not** normalisation (we cluster the surface forms themselves from the diplomatic digital edition).

## Proposed system

A purely **lexical (string-only)** method; no embeddings, no training, no external resources:

- **String distance:** Combined Levenshtein and Jaro-Winkler string similiarity measures (50/50 weighting).
- **Clustering:** HDBSCAN (excess-of-mass, `min_cluster_size=2`, `min_samples=1`) on the precomputed distance matrix, then an intra-cluster distance cap (0.40); noise → singletons.
- Run separately per entity type: **PER** (personal names), **PEO** (ethnonyms), **LOC** (locations).

## Main results (B³ F1, with singletons, see paper for more)

| Type | forms → clusters | B³ F1 |
|---|---|---|
| PER (persons) | 317 → 174 | **0.901** |
| PEO (ethnonyms) | 74 → 18 | **0.597** |
| LOC (locations) | 128 → 76 | **0.938** |

We also perform an ablation against neural embeddings, a domain-adapted IceBERT (contextual) and a
domain-trained FastText subword model, alone and combined with the string distance. The results are mixed but show that the string measures drive performance, while embeddings can add small but noticeable gains in performance. 

## Reproduce

The **string-only system needs no model**; the neural arms use two models that are **public on Hugging Face and download automatically**:
`phenningsson/icebert-old-icelandic-v11-diplomatic` and
`phenningsson/fasttext-old-icelandic-diplomatic`.

```bash
pip install -r requirements.txt          # on a CUDA GPU, install the matching torch build (see below)

python -m evaluation.run_all_arms        # cluster every arm from the gold forms + score
python -m evaluation.validate            # soundness/reproducibility checks — must exit 0
python -m evaluation.weight_sweep        # weight-sensitivity sweep
python -m evaluation.plot_weight_sweep   # weight-sensitivity figure
python -m evaluation.qualitative --arm string   # error-analysis strata + per-cluster diffs
```

Per-arm, per-type scores are written to `evaluation/out/results_all_arms.json`. Just the
headline (string-only, no model): `python -m evaluation.run_all_arms --arms string`.

**Environment.** Python 3.13; key packages `numpy`, `scikit-learn`, `hdbscan`,
`python-Levenshtein`, `gensim`, `matplotlib`, `torch`, `transformers`, `huggingface_hub`
(see `requirements.txt`). Only the IceBERT arms need `torch`; on an NVIDIA GPU install the
CUDA build, e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu130`.
Clustering is deterministic. UMAP (`random_state=42`) affects only the GUI layout, not the scores.

## Configuration (hyperparameters)

| component | setting |
|---|---|
| String similarity | `0.50·JaroWinkler + 0.50·(1−normLevenshtein)` |
| Embedding distance | `1 − cosine` (L2-normalised) |
| Hybrid distance | `default at 0.60·string + 0.40·embedding` (weight swept over `{0.0,…,1.0}`) |
| HDBSCAN | `metric=precomputed`, `min_cluster_size=2`, `min_samples=1`, selection `eom`, `epsilon=0.0` |
| Intra-cluster cap | 0.40 (peel the farthest member until all pairs ≤ 0.40 to remove outliers/false positives) |

The hyperparameters can be set in `clustering/clustering_v5.py`.

## Interactive GUI

An interactive cluster explorer; human-in-the-loop curation interface in which a
user can inspect, merge, split, and relabel clusters and export the result.
Pre-rendered examples (one per entity type) are in [`examples/gui/`](examples/gui/). Open them
directly in a browser (you may need to download them as files and then open the file in a browser). To regenerate:

```bash
python -m evaluation.visualize_clusters --arm string
python -m evaluation.serve_gui
```

## Repository layout

```
clustering/clustering_v5.py     the clustering pipeline (distance + HDBSCAN + GUI generator)
evaluation/                     evaluation harness
  run_all_arms.py               cluster every arm + score
  cluster_from_gold.py          cluster directly from the domain expert's gold forms (all arms)
  evaluate.py, metrics.py, clueval_metrics.py, align.py, gold.py, pipeline_io.py   evaluation scoring 
  contextual_embeddings.py, fasttext_embeddings.py   neural ablation arms
  weight_sweep.py, plot_weight_sweep.py              weight-sensitivity table + figure
  qualitative.py                error-analysis + per-cluster diffs
  validate.py                   reproducibility harness (exits non-zero on failure)
  visualize_clusters.py, serve_gui.py, tune.py, cluster_config.py   interactive GUI
clueval/CluEval/                metric tool (from Kim & Kim 2023, MIT license)
alexander_saga_persons.csv      gold: PER (personal names)
alexander_saga_people.csv       gold: PEO (ethnonyms / peoples)
alexander_saga_locations.csv    gold: LOC (locations)
alexander_saga_diplomatic.txt   raw diplomatic text (used by the contextual embedding arm)
examples/gui/                   example GUI cluster explorers (open in a browser)
```

## Data & licensing

- **Code** is released under the MIT License (see [`LICENSE`](LICENSE)).
- **Gold standard and text.** The gold clustering (three entity types, PER / PEO / LOC,
  matching the source edition's entities) and the diplomatic text derive from the **Menota**
  diplomatic edition *AM 519 a 4to : Alexanders saga* by **Andrea de Leeuw van Weenen** (Medieval
  Nordic Text Archive, <https://clarino.uib.no/menota/text/menota/AM-519a-4to>), licensed **CC-BY-SA 4.0**. The full source edition is not redistributed here, see Menota. Any reuse of the included data (gold CSVs, diplomatic text) must preserve that attribution and share-alike.
- **FastText corpus.** The subword model was trained on the diplomatic corpus released with the domain-adapted IceBERT MLM; the *exact trained model* is hosted on Hugging Face and downloaded automatically,
  so the corpus is not needed to reproduce results. The model is available [`here`](https://huggingface.co/phenningsson/fasttext-old-icelandic-diplomatic).
- **Domain-adapted IceBERT.** Is available [`here`](https://huggingface.co/phenningsson/icebert-old-icelandic-v11-diplomatic), and relates to [`this project`](https://huggingface.co/Riksarkivet/oldbertur-diplomatic-old-icelandic-ner).
- **CluEval** (`clueval/CluEval`) is under its own MIT License; cite Kim & Kim (2023).

## Citation

If you use this code or data, please cite the accompanying paper (**Coming soon**) and the sources above as relevant.
