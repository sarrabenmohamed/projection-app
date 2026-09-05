"""
FLORES-200 + XLM-R embedding pipeline: for each requested language, embeds
every sentence into a mean-pooled sentence vector (last hidden layer, padding
masked out) plus a per-token vector for every non-special token. A sentence's
row index is stable across languages and becomes its `pairId`, which the app
uses to link translations together.

One-time setup: python3 -c "from huggingface_hub import login; login()"
then accept terms at https://huggingface.co/datasets/facebook/flores
"""

from __future__ import annotations

import os

# transformers lazily imports TensorFlow if it's installed at all, even though
# we only use torch; on machines with a broken/incompatible TF install this
# hangs forever inside TF's native self-check. This must be set before
# `import transformers` anywhere in the process.
os.environ.setdefault("USE_TF", "0")

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> list[float]:
    """Mean-pool token vectors for one sentence, ignoring padding."""
    m = mask.unsqueeze(-1).float()
    vec = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
    return vec.squeeze(0).detach().tolist()


def _token_vectors(hidden: torch.Tensor, tokens_str: list[str], special_tokens_mask: list[int]) -> list[dict]:
    """Per-token vectors for one sentence, excluding special tokens (<s>, </s>, padding)."""
    return [
        {"token": tok, "vector": hidden[0, i].detach().tolist()}
        for i, (tok, is_special) in enumerate(zip(tokens_str, special_tokens_mask))
        if not is_special
    ]


def embed_sentence(sent: str, tokenizer, model) -> tuple[list[float], list[dict]]:
    """Embed one sentence, returning (pooled sentence vector, per-token vectors)."""
    enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128, return_special_tokens_mask=True)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    hidden = out.hidden_states[-1]
    pooled = _mean_pool(hidden, enc["attention_mask"])
    tokens_str = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    tvecs = _token_vectors(hidden, tokens_str, enc["special_tokens_mask"][0].tolist())
    return pooled, tvecs


def build_dataset(lang_codes: list[str], n_sentences: int, model_name: str) -> dict:
    """Loads FLORES-200 for each language and embeds it, returning the same
    JSON-shaped dict server.py caches to disk and loads on startup."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()

    points = []
    for code in lang_codes:
        print(f"Loading + embedding {code} ...")
        ds = load_dataset("facebook/flores", code, split="devtest").select(range(n_sentences))
        for pair_id, row in enumerate(ds):
            text = row["sentence"]
            vector, token_vectors = embed_sentence(text, tokenizer, model)
            points.append({
                "id": f"{code}-{pair_id}",
                "lang": code,
                "pairId": pair_id,
                "text": text,
                "vector": vector,
                "tokenVectors": token_vectors,
            })

    return {
        "model": model_name,
        "layer": "last",
        "dim": len(points[0]["vector"]),
        "lang_codes": lang_codes,
        "points": points,
    }
