"""
Prepare a JSON input for this app from FLORES-200 parallel sentences.

For each requested language, loads the FLORES-200 devtest split, embeds every
sentence with a HuggingFace encoder, and mean-pools the LAST hidden layer into
one vector per sentence. FLORES-200 is parallel by row index, so a sentence's
row index is stable across languages and becomes its `pairId` in the output —
the app uses that to find "this point's translations" on click.

Requires a one-time HuggingFace login and accepting the FLORES-200 terms:
    python3 -c "from huggingface_hub import login; login()"
    https://huggingface.co/datasets/facebook/flores

Usage (run from the repo root):
    python3 scripts/prepare_flores_embeddings.py \
        --languages eng_Latn,deu_Latn,fra_Latn,spa_Latn,zho_Hans \
        --n-sentences 200 \
        --model xlm-roberta-base \
        --output data/flores_embeddings.json
"""

from __future__ import annotations

import os

# transformers lazily imports TensorFlow if it's installed at all, even though
# we only use torch; on machines with a broken/incompatible TF install this
# hangs forever inside TF's native self-check. This must be set before
# `import transformers` anywhere in the process.
os.environ.setdefault("USE_TF", "0")

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> list[float]:
    """Mean-pool token vectors for one sentence, ignoring padding."""
    m = mask.unsqueeze(-1).float()
    vec = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
    return vec.squeeze(0).detach().tolist()


def embed_language(sentences: list[str], tokenizer, model) -> list[list[float]]:
    """Embed every sentence, returning one last-layer vector per sentence."""
    vectors = []
    for sent in sentences:
        enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            out = model(**enc)
        vectors.append(_mean_pool(out.hidden_states[-1], enc["attention_mask"]))
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--languages", default="eng_Latn,deu_Latn,fra_Latn,spa_Latn,zho_Hans",
                         help="Comma-separated FLORES-200 language codes")
    parser.add_argument("--n-sentences", type=int, default=200)
    parser.add_argument("--model", default="xlm-roberta-base")
    parser.add_argument("--output", default="data/flores_embeddings.json")
    args = parser.parse_args()

    lang_codes = [c.strip() for c in args.languages.split(",") if c.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, output_hidden_states=True)
    model.eval()

    points = []
    for code in lang_codes:
        print(f"Loading + embedding {code} ...")
        ds = load_dataset("facebook/flores", code, split="devtest").select(range(args.n_sentences))
        sentences = [row["sentence"] for row in ds]
        vectors = embed_language(sentences, tokenizer, model)
        for pair_id, (text, vector) in enumerate(zip(sentences, vectors)):
            points.append({
                "id": f"{code}-{pair_id}",
                "lang": code,
                "pairId": pair_id,
                "text": text,
                "vector": vector,
            })

    output = {
        "model": args.model,
        "layer": "last",
        "dim": len(points[0]["vector"]),
        "lang_codes": lang_codes,
        "points": points,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output))
    print(f"Wrote {len(points)} points ({len(lang_codes)} languages x {args.n_sentences} sentences) to {out_path}")


if __name__ == "__main__":
    main()
