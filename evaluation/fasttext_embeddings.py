"""Domain subword (FastText) embeddings for the FastText ablation arm.

A char-n-gram FastText model encodes surface form directly (unlike a BERT [CLS]
vector of a short bare form), so this arm tests whether a surface-aware embedding
helps. The model is trained on the diplomatic Old Icelandic corpus, cased and
UTF-8, with no lowercasing; the char n-grams capture the medieval orthography
(`ꝺ`, `ſ`, `ı`) and give every form a vector via its subwords, including
out-of-vocabulary and multiword forms such as `ꞇroıo manna`.

`form_vectors` fetches the exact trained model from Hugging Face on first use, so
the arm reproduces the reported numbers without retraining. To retrain from scratch:
    python -m evaluation.fasttext_embeddings --train --corpus path/to/corpus[.txt|/dir]
"""

import argparse
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = _REPO / "evaluation" / "models" / "fasttext_diplomatic.model"
# The exact trained model used for the reported results (so the FastText arm
# reproduces the paper numbers exactly, rather than depending on a fresh, non-
# deterministic retrain). Downloaded on demand if the local model is absent.
HF_MODEL_REPO = "phenningsson/fasttext-old-icelandic-diplomatic"
# gensim FastText.load needs the .model plus its ngram-vector sidecar.
_HF_MODEL_FILES = ("fasttext_diplomatic.model",
                   "fasttext_diplomatic.model.wv.vectors_ngrams.npy")


def read_corpus(path):
    """Read training sentences from a UTF-8 text file or a directory of .txt files.

    One sentence/segment per line, whitespace-tokenised, cased (never lowercase).
    Returns a list of token lists.
    """
    path = Path(path)
    files = sorted(path.glob("*.txt")) if path.is_dir() else [path]
    if not files or (len(files) == 1 and not files[0].exists()):
        raise FileNotFoundError(f"corpus not found at {path}")
    sentences = []
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                toks = line.split()
                if toks:
                    sentences.append(toks)
    if not sentences:
        raise ValueError(f"no tokens read from {path}")
    return sentences


def train_fasttext(corpus_path, out_path=DEFAULT_MODEL_PATH, vector_size=100,
                   window=5, min_n=2, max_n=5, epochs=20, min_count=1, sg=1,
                   seed=42, workers=4):
    """Train a char-n-gram FastText model on the diplomatic corpus and save it.

    Defaults: skip-gram (sg=1), 100-d, char n-grams 2–5, min_count=1 (small corpus).
    Tune via the CLI flags once we see the corpus size.
    """
    from gensim.models import FastText
    sentences = read_corpus(corpus_path)
    print(f"Training FastText on {len(sentences)} sentences "
          f"(vector_size={vector_size}, n-grams {min_n}-{max_n}, epochs={epochs})...")
    model = FastText(vector_size=vector_size, window=window, min_n=min_n, max_n=max_n,
                     min_count=min_count, sg=sg, seed=seed, workers=workers)
    model.build_vocab(corpus_iterable=sentences)
    model.train(corpus_iterable=sentences, total_examples=model.corpus_count, epochs=epochs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    print(f"Saved model to {out_path}  (vocab={len(model.wv)} words)")
    return out_path


def _ensure_model(model_path):
    """Download the exact trained FastText model from Hugging Face if it is not
    present locally (so results reproduce exactly). Returns the local .model path."""
    model_path = Path(model_path)
    if model_path.exists():
        return model_path
    from huggingface_hub import hf_hub_download
    model_path.parent.mkdir(parents=True, exist_ok=True)
    for fn in _HF_MODEL_FILES:
        hf_hub_download(repo_id=HF_MODEL_REPO, filename=fn,
                        local_dir=str(model_path.parent))
    return model_path


def form_vectors(forms, model_path=DEFAULT_MODEL_PATH):
    """Return an (N, dim) array of FastText vectors for `forms` (subword-derived, so
    every form — including OOV/multiword — gets a vector). The exact trained model is
    fetched from Hugging Face on first use if not present locally."""
    from gensim.models import FastText
    model_path = _ensure_model(model_path)
    model = FastText.load(str(model_path))
    return np.array([model.wv[f] for f in forms], dtype=np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true", help="train the model")
    ap.add_argument("--corpus", help="UTF-8 text file or directory of .txt files")
    ap.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    ap.add_argument("--vector-size", type=int, default=100)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-n", type=int, default=2)
    ap.add_argument("--max-n", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--sg", type=int, default=1, choices=[0, 1], help="1=skip-gram, 0=cbow")
    args = ap.parse_args(argv)

    if args.train:
        if not args.corpus:
            ap.error("--train requires --corpus")
        train_fasttext(args.corpus, args.out, vector_size=args.vector_size,
                       window=args.window, min_n=args.min_n, max_n=args.max_n,
                       epochs=args.epochs, min_count=args.min_count, sg=args.sg)
    else:
        ap.error("nothing to do; pass --train --corpus <path>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
