# Examples

## GUI cluster explorers (`gui/`)

Self-contained, interactive HTML renderings of the clustering GUI, one per entity type
(`per`, `peo`, `loc`). Each shows the string-only clustering of the gold surface forms as
a UMAP scatter plus editable cluster lists, i.e., the human-in-the-loop curation interface. Open
any file directly in a browser (no server needed).

Regenerate them for any arm from the gold forms:

```bash
python -m evaluation.visualize_clusters --arm string      # writes evaluation/out/gui/<arm>/
python -m evaluation.serve_gui                             # or just open the .html files
```
