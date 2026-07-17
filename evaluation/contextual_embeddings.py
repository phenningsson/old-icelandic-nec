"""Contextual per-form embeddings (an ablation arm).

Instead of embedding each surface form in isolation (the bare form's [CLS] vector),
this embeds each occurrence of a form in its sentence-window context and mean-pools
per form:

  for each gold form F:
      find every whole-token occurrence of F in the raw text
      for each occurrence, embed the surrounding window and mean-pool the subword
          token vectors that fall inside F's character span
      average the per-occurrence vectors into one contextual vector for F

Forms are located by search over the raw text; a form with no locatable occurrence
returns None (the caller falls back to the isolation embedding). Contextual vectors
encode usage and referent rather than surface form, so this arm is not expected to
help when grouping spelling variants of a name -- which is what the ablation tests.
"""

import re
import numpy as np


def find_occurrences(form, raw_text):
    """Whole-token char spans (start, end) of `form` in `raw_text`.

    `(?<!\\w)`/`(?!\\w)` boundaries (Unicode-aware, so medieval letters like ꝺ ſ ı
    count as word chars) prevent matching `ſerke` inside `ſerker`. Multiword forms
    (`ꞇroıo manna`) match as a whole.
    """
    pat = re.compile(r"(?<!\w)" + re.escape(form) + r"(?!\w)")
    return [(m.start(), m.end()) for m in pat.finditer(raw_text)]


def contextual_form_embeddings(forms, raw_text, model_path,
                               window=100, max_length=128, batch_size=16):
    """Return {form: np.ndarray|None} of mean-pooled contextual embeddings."""
    from transformers import AutoTokenizer, AutoModel
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()

    # gather all occurrence windows, tagged by form, in one flat list to batch
    jobs = []  # (form, window_text, span_start_in_window, span_end_in_window)
    n_text = len(raw_text)
    for f in forms:
        for a, b in find_occurrences(f, raw_text):
            ws, we = max(0, a - window), min(n_text, b + window)
            jobs.append((f, raw_text[ws:we], a - ws, b - ws))

    per_form = {f: [] for f in forms}
    with torch.no_grad():
        for i in range(0, len(jobs), batch_size):
            batch = jobs[i:i + batch_size]
            texts = [j[1] for j in batch]
            enc = tok(texts, return_offsets_mapping=True, padding=True,
                      truncation=True, max_length=max_length, return_tensors="pt")
            offsets = enc.pop("offset_mapping")
            enc = enc.to(device)
            out = model(**enc).last_hidden_state.cpu()  # [B, T, H]
            for bi, (f, _txt, sa, sb) in enumerate(batch):
                om = offsets[bi].tolist()
                idx = [j for j, (ts, te) in enumerate(om)
                       if not (ts == 0 and te == 0) and ts < sb and te > sa]
                vec = (out[bi, idx, :].mean(0) if idx else out[bi, 0, :]).numpy()
                per_form[f].append(vec)

    return {f: (np.mean(v, axis=0) if v else None) for f, v in per_form.items()}
