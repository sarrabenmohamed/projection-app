"""The real app with a fake embedder, so browser tests can load and edit
FLORES-200 datasets in seconds without downloading a model or the dataset.

Run with `uvicorn --app-dir tests/e2e fake_server:app` from the repo root.
Test-only routes: GET /__calls lists what the embedder was asked for, and
POST /__reset forgets that and empties the per-language cache.
"""
import atexit
import shutil
import tempfile
import zlib
from pathlib import Path

import numpy as np

import server
from backend.embedding_pipeline import UnknownLanguageError

KNOWN_LANGUAGES = {"eng_Latn", "fra_Latn", "deu_Latn", "spa_Latn", "ita_Latn", "arb_Arab", "rus_Cyrl", "jpn_Jpan"}
DIM = 12

calls = []
cache_dir = Path(tempfile.mkdtemp(prefix="e2e-flores-"))
server.flores_loader.data_dir = cache_dir
atexit.register(shutil.rmtree, cache_dir, ignore_errors=True)


def _vector(code, pair_id):
    """A language's own direction dominates, so languages form clusters; the
    sentence's shared part keeps translations somewhat related."""
    def draw(seed):
        return np.random.default_rng(seed).normal(size=DIM)
    return (3 * draw(zlib.crc32(code.encode())) + draw(1000 + pair_id) + 0.2 * draw(zlib.crc32(f"{code}-{pair_id}".encode()))).tolist()


def fake_build_dataset(lang_codes, n_sentences, model_name, pooling="mean", should_cancel=None, start=0):
    unknown = [c for c in lang_codes if c not in KNOWN_LANGUAGES]
    if unknown:
        raise UnknownLanguageError(f'"{unknown[0]}" is not a FLORES-200 language code (e.g. eng_Latn, deu_Latn)')
    calls.append({"lang_codes": list(lang_codes), "n_sentences": n_sentences, "start": start, "model": model_name, "pooling": pooling})
    points = [
        {
            "id": f"{code}-{pair_id}",
            "lang": code,
            "pairId": pair_id,
            "text": f"{code} sentence {pair_id}",
            "vector": _vector(code, pair_id),
            "tokenVectors": [{"token": f"▁t{k}", "vector": _vector(f"{code}/{k}", pair_id)} for k in range(3)],
        }
        for code in lang_codes
        for pair_id in range(start, n_sentences)
    ]
    return {"model": model_name, "layer": "last", "pooling": pooling, "dim": DIM, "lang_codes": list(lang_codes), "points": points}


server.embedding_pipeline.build_dataset = fake_build_dataset
app = server.app


@app.get("/__calls")
def list_calls():
    return calls


@app.post("/__reset")
def reset():
    calls.clear()
    for cached in cache_dir.glob("*.json"):
        cached.unlink()
    return {"reset": True}

