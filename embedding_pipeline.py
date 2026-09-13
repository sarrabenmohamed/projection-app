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


def _max_pool(hidden: torch.Tensor, mask: torch.Tensor) -> list[float]:
    """Max-pool token vectors for one sentence, ignoring padding (masked
    positions are set to -inf first so they never win the max)."""
    m = mask.unsqueeze(-1).bool()
    masked = hidden.masked_fill(~m, float("-inf"))
    vec = masked.max(dim=1).values
    return vec.squeeze(0).detach().tolist()


def _cls_pool(hidden: torch.Tensor, mask: torch.Tensor) -> list[float]:
    """Use the first token's (<s>) hidden state directly, no pooling across
    the sequence. `mask` is unused — kept so all three pooling functions
    share one call signature and can be selected interchangeably."""
    del mask
    return hidden[:, 0, :].squeeze(0).detach().tolist()


POOLING_FNS = {"mean": _mean_pool, "max": _max_pool, "cls": _cls_pool}


def _token_vectors(hidden: torch.Tensor, tokens_str: list[str], special_tokens_mask: list[int]) -> list[dict]:
    """Per-token vectors for one sentence, excluding special tokens (<s>, </s>, padding)."""
    return [
        {"token": tok, "vector": hidden[0, i].detach().tolist()}
        for i, (tok, is_special) in enumerate(zip(tokens_str, special_tokens_mask))
        if not is_special
    ]


def embed_sentence(sent: str, tokenizer, model, pooling: str = "mean") -> tuple[list[float], list[dict]]:
    """Embed one sentence, returning (pooled sentence vector, per-token vectors).
    `pooling` selects how the token vectors become one sentence vector — one
    of POOLING_FNS' keys ("mean", "max", "cls"); token vectors themselves are
    always the raw per-token hidden states, unaffected by this choice."""
    enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128, return_special_tokens_mask=True)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    hidden = out.hidden_states[-1]
    pooled = POOLING_FNS[pooling](hidden, enc["attention_mask"])
    tokens_str = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    tvecs = _token_vectors(hidden, tokens_str, enc["special_tokens_mask"][0].tolist())
    return pooled, tvecs


def build_dataset(lang_codes: list[str], n_sentences: int, model_name: str, pooling: str = "mean") -> dict:
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
            vector, token_vectors = embed_sentence(text, tokenizer, model, pooling)
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
        "pooling": pooling,
        "dim": len(points[0]["vector"]),
        "lang_codes": lang_codes,
        "points": points,
    }
