# Token Embeddings + Backend/Script Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fold the FLORES-200 embedding prep script into the backend (compute-or-cache on startup), store per-token embeddings alongside each sentence's pooled vector, and let the frontend overlay a selected point's tokens into the same UMAP axes as the main sentence scatter.

**Architecture:** `scripts/prepare_flores_embeddings.py`'s logic moves into a new `embedding_pipeline.py` module imported by `server.py`; `server.py` computes-or-loads-a-matching-cache of the default dataset on startup instead of requiring a manual script run. The dataset schema gains an optional `tokenVectors` field per point. A UMAP reducer is fit once per dataset "generation" and cached in memory so `GET /project/tokens/{id}` can `.transform()` a point's token vectors into the exact same 2D space as the main sentence projection (no refit, no drift). The frontend overlays those token points on the same canvas/axes as an independent per-point toggle, coexisting with the existing click-to-isolate-translations behavior.

**Tech Stack:** Python (FastAPI, torch, transformers, datasets, umap-learn, scikit-learn, numpy, pytest), vanilla JS (D3 v7, Canvas, ES modules, no build step, no frontend test framework).

**Spec:** `docs/superpowers/specs/2026-09-03-token-embeddings-and-backend-merge-design.md`

## Global Constraints

- No new heavy dependencies for the shared-space projection — use classic `umap.UMAP().transform()`, not `ParametricUMAP` (which needs a real TensorFlow/Keras backend that this repo already works around a hang for, in `server.py:37`).
- `tokenVectors` is optional everywhere; datasets (uploaded or backend-generated) that omit it keep working exactly as today.
- Special tokens (`<s>`, `</s>`) are excluded from stored `tokenVectors` but still included in the mean-pool that produces `vector`, unchanged from today.
- Backend config via env vars: `PROJECTION_LANGUAGES` (default `eng_Latn,deu_Latn,fra_Latn,spa_Latn,zho_Hans`), `PROJECTION_N_SENTENCES` (default `200`), `PROJECTION_MODEL` (default `xlm-roberta-base`) — same defaults the script's CLI flags had.
- Cache file path unchanged: `data/flores_embeddings.json`.
- `GET /points`, `GET /project`, `POST /dataset` stay backward compatible — only additive changes (`hasTokens`, optional `tokenVectors`).
- Token overlay coexists with translation-isolation in the main view; it never replaces the sentence scatter. Only one point's token overlay is shown at a time.

---

### Task 1: `embedding_pipeline.py` — pure embedding functions + dataset builder

**Files:**
- Create: `embedding_pipeline.py`
- Delete: `scripts/prepare_flores_embeddings.py` (and the now-empty `scripts/` directory)
- Modify: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `pyproject.toml`
- Modify: `.gitignore`
- Test: `tests/test_embedding_pipeline.py`

**Interfaces:**
- Produces: `embedding_pipeline._mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> list[float]`, `embedding_pipeline._token_vectors(hidden: torch.Tensor, tokens_str: list[str], special_tokens_mask: list[int]) -> list[dict]` (each dict is `{"token": str, "vector": list[float]}`), `embedding_pipeline.embed_sentence(sent: str, tokenizer, model) -> tuple[list[float], list[dict]]`, `embedding_pipeline.build_dataset(lang_codes: list[str], n_sentences: int, model_name: str) -> dict` (same JSON shape the old script wrote to disk, now with `tokenVectors` on every point). Task 3 calls `build_dataset` directly.

- [ ] **Step 1: Write the failing tests for the pure helper functions**

Create `tests/test_embedding_pipeline.py`:

```python
import pytest
import torch

import embedding_pipeline as ep


def test_mean_pool_averages_only_unmasked_tokens():
    hidden = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [99.0, 99.0]]])  # 1 batch, 3 tokens, dim 2
    mask = torch.tensor([[1, 1, 0]])  # third token is padding
    pooled = ep._mean_pool(hidden, mask)
    assert pooled == pytest.approx([2.0, 0.0])


def test_token_vectors_excludes_special_tokens():
    hidden = torch.tensor([[[9.0, 9.0], [1.0, 2.0], [3.0, 4.0], [9.0, 9.0]]])
    tokens_str = ["<s>", "▁cat", "▁sat", "</s>"]
    special_mask = [1, 0, 0, 1]
    tvecs = ep._token_vectors(hidden, tokens_str, special_mask)
    assert [t["token"] for t in tvecs] == ["▁cat", "▁sat"]
    assert tvecs[0]["vector"] == pytest.approx([1.0, 2.0])
    assert tvecs[1]["vector"] == pytest.approx([3.0, 4.0])


def test_token_vectors_same_dim_as_hidden_size():
    hidden = torch.zeros((1, 2, 768))
    tvecs = ep._token_vectors(hidden, ["a", "b"], [0, 0])
    assert all(len(t["vector"]) == 768 for t in tvecs)
```

- [ ] **Step 2: Run the tests to verify they fail with an import error**

Run: `python3 -m pytest tests/test_embedding_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'embedding_pipeline'`

- [ ] **Step 3: Create `embedding_pipeline.py`**

```python
"""
FLORES-200 + XLM-R embedding pipeline for the projection backend.

For each requested language, loads the FLORES-200 devtest split and embeds
every sentence with a HuggingFace encoder, producing:
  - a mean-pooled sentence vector (the LAST hidden layer, averaged over
    tokens, padding masked out)
  - a per-token vector for every non-special token in that same last layer

FLORES-200 is parallel by row index, so a sentence's row index is stable
across languages and becomes its `pairId` in the output — the app uses that
to find "this point's translations" on click.

Requires a one-time HuggingFace login and accepting the FLORES-200 terms:
    python3 -c "from huggingface_hub import login; login()"
    https://huggingface.co/datasets/facebook/flores
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
```

- [ ] **Step 4: Delete the old script and its now-empty directory**

```bash
git rm scripts/prepare_flores_embeddings.py
rmdir scripts
```

- [ ] **Step 5: Merge `requirements.txt` into one list and add a dev requirements file**

Replace the full contents of `requirements.txt`:

```
# Embedding pipeline (embedding_pipeline.py)
torch
transformers
datasets

# Backend server (server.py)
fastapi
uvicorn
umap-learn
scikit-learn
numpy
```

Create `requirements-dev.txt`:

```
-r requirements.txt
pytest
```

- [ ] **Step 6: Add pytest config so `tests/` can import root-level modules**

Create `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 7: Update the `.gitignore` comment that referenced the deleted script**

In `.gitignore`, replace:

```
# Generated embeddings data (regenerate via scripts/prepare_flores_embeddings.py) —
# well over GitHub's 100MB push limit at full size, and not source anyway.
data/
```

with:

```
# Generated embeddings data (regenerated automatically by server.py on
# startup via embedding_pipeline.py) — well over GitHub's 100MB push limit
# at full size, and not source anyway.
data/
```

- [ ] **Step 8: Install dev requirements and run the tests**

```bash
pip install -r requirements-dev.txt
python3 -m pytest tests/test_embedding_pipeline.py -v
```

Expected: 3 passed.

- [ ] **Step 9: Commit**

```bash
git add embedding_pipeline.py requirements.txt requirements-dev.txt pyproject.toml .gitignore tests/test_embedding_pipeline.py
git commit -m "Fold embedding prep script into embedding_pipeline.py, add token vectors"
```

---

### Task 2: `server.py` — accept and expose `tokenVectors`

**Files:**
- Modify: `server.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `validate_and_normalize` now accepts an optional `tokenVectors: list[{"token": str, "vector": list[float]}]` per point and raises `ValueError` if malformed; `GET /points` response points gain `"hasTokens": bool`. Later tasks (3, 4) build on this same `validate_and_normalize` and the `/points` point shape.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_server.py`:

```python
import copy
import json

import pytest
from fastapi.testclient import TestClient

import server

SAMPLE_DATASET = {
    "model": "test-model",
    "layer": "last",
    "dim": 3,
    "lang_codes": ["en", "de"],
    "points": [
        {"id": "en-0", "lang": "en", "pairId": 0, "text": "hi", "vector": [0.1, 0.2, 0.3]},
        {"id": "de-0", "lang": "de", "pairId": 0, "text": "hallo", "vector": [0.15, 0.25, 0.35]},
        {
            "id": "en-1", "lang": "en", "pairId": 1, "text": "bye", "vector": [0.9, 0.8, 0.7],
            "tokenVectors": [
                {"token": "bye", "vector": [0.9, 0.8, 0.7]},
                {"token": "!", "vector": [0.5, 0.5, 0.5]},
            ],
        },
    ],
}


@pytest.fixture
def client():
    server.DATASET["data"] = server.validate_and_normalize(copy.deepcopy(SAMPLE_DATASET))
    return TestClient(server.app)


def test_validate_and_normalize_accepts_dataset_without_token_vectors():
    normalized = server.validate_and_normalize(copy.deepcopy(SAMPLE_DATASET))
    assert normalized["points"][0].get("tokenVectors") is None


def test_validate_and_normalize_accepts_token_vectors():
    normalized = server.validate_and_normalize(copy.deepcopy(SAMPLE_DATASET))
    assert normalized["points"][2]["tokenVectors"][0] == {"token": "bye", "vector": [0.9, 0.8, 0.7]}


def test_validate_and_normalize_rejects_wrong_length_token_vector():
    bad = copy.deepcopy(SAMPLE_DATASET)
    bad["points"][2]["tokenVectors"] = [{"token": "bye", "vector": [0.9, 0.8]}]  # dim 2, dataset dim is 3
    with pytest.raises(ValueError):
        server.validate_and_normalize(bad)


def test_validate_and_normalize_rejects_token_vector_missing_token_string():
    bad = copy.deepcopy(SAMPLE_DATASET)
    bad["points"][2]["tokenVectors"] = [{"vector": [0.9, 0.8, 0.7]}]
    with pytest.raises(ValueError):
        server.validate_and_normalize(bad)


def test_points_endpoint_reports_has_tokens(client):
    res = client.get("/points")
    body = res.json()
    by_id = {p["id"]: p for p in body["points"]}
    assert by_id["en-0"]["hasTokens"] is False
    assert by_id["en-1"]["hasTokens"] is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_server.py -v`
Expected: FAIL — `test_validate_and_normalize_rejects_wrong_length_token_vector` and `test_validate_and_normalize_rejects_token_vector_missing_token_string` fail because no validation exists yet (no `ValueError` raised); `test_points_endpoint_reports_has_tokens` fails with `KeyError: 'hasTokens'`.

- [ ] **Step 3: Add `tokenVectors` validation to `validate_and_normalize`**

In `server.py`, find:

```python
        vec = p.get("vector")
        if (not isinstance(vec, list) or len(vec) != dim
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vec)):
            raise ValueError(f'points[{i}].vector must be an array of {dim} finite numbers')

    return {
```

Replace with:

```python
        vec = p.get("vector")
        if (not isinstance(vec, list) or len(vec) != dim
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vec)):
            raise ValueError(f'points[{i}].vector must be an array of {dim} finite numbers')

        tvecs = p.get("tokenVectors")
        if tvecs is not None:
            if not isinstance(tvecs, list) or len(tvecs) == 0:
                raise ValueError(f'points[{i}].tokenVectors must be a non-empty array')
            for j, t in enumerate(tvecs):
                tvec = t.get("vector") if isinstance(t, dict) else None
                if (not isinstance(t, dict) or not isinstance(t.get("token"), str)
                        or not isinstance(tvec, list) or len(tvec) != dim
                        or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in tvec)):
                    raise ValueError(
                        f'points[{i}].tokenVectors[{j}] must be {{"token": str, "vector": array of {dim} finite numbers}}'
                    )

    return {
```

- [ ] **Step 4: Add `hasTokens` to `GET /points`**

In `server.py`, find:

```python
        "points": [
            {"id": p["id"], "lang": p["lang"], "pairId": p.get("pairId"), "text": p.get("text")}
            for p in d["points"]
        ],
```

Replace with:

```python
        "points": [
            {
                "id": p["id"], "lang": p["lang"], "pairId": p.get("pairId"), "text": p.get("text"),
                "hasTokens": p.get("tokenVectors") is not None,
            }
            for p in d["points"]
        ],
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_server.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "Accept optional tokenVectors and report hasTokens from /points"
```

---

### Task 3: `server.py` — compute-or-cache the default dataset on startup

**Files:**
- Modify: `server.py`
- Modify: `README.md`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `embedding_pipeline.build_dataset(lang_codes, n_sentences, model_name) -> dict` (Task 1).
- Produces: `server._pipeline_config() -> dict` (`{"lang_codes": list[str], "n_sentences": int, "model": str}`), `server._cache_matches(cached: dict, config: dict) -> bool`, `server._load_or_build_dataset(cache_path: Path, config: dict, builder: Callable) -> dict`, and a `DATASET["generation"]` counter (`int`, starts at `0`, incremented every time `DATASET["data"]` is (re)set) that Task 4 uses to invalidate the cached UMAP reducer.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_server.py`:

```python
def test_pipeline_config_reads_env_vars(monkeypatch):
    monkeypatch.setenv("PROJECTION_LANGUAGES", "eng_Latn,deu_Latn")
    monkeypatch.setenv("PROJECTION_N_SENTENCES", "50")
    monkeypatch.setenv("PROJECTION_MODEL", "some-model")
    assert server._pipeline_config() == {
        "lang_codes": ["eng_Latn", "deu_Latn"], "n_sentences": 50, "model": "some-model",
    }


def test_cache_matches_true_when_config_matches():
    cached = {"model": "m", "lang_codes": ["a", "b"], "points": [{"lang": "a"}, {"lang": "a"}, {"lang": "b"}, {"lang": "b"}]}
    config = {"model": "m", "lang_codes": ["a", "b"], "n_sentences": 2}
    assert server._cache_matches(cached, config) is True


def test_cache_matches_false_on_model_mismatch():
    cached = {"model": "old", "lang_codes": ["a"], "points": [{"lang": "a"}]}
    config = {"model": "new", "lang_codes": ["a"], "n_sentences": 1}
    assert server._cache_matches(cached, config) is False


def test_cache_matches_false_on_sentence_count_mismatch():
    cached = {"model": "m", "lang_codes": ["a"], "points": [{"lang": "a"}]}
    config = {"model": "m", "lang_codes": ["a"], "n_sentences": 2}
    assert server._cache_matches(cached, config) is False


def test_load_or_build_dataset_uses_cache_on_hit(tmp_path):
    cache_path = tmp_path / "cache.json"
    cached = {"model": "m", "lang_codes": ["a"], "points": [{"lang": "a", "id": "a-0", "vector": [0.0]}]}
    cache_path.write_text(json.dumps(cached))
    config = {"model": "m", "lang_codes": ["a"], "n_sentences": 1}

    def fake_builder(*args):
        raise AssertionError("builder should not be called on a cache hit")

    result = server._load_or_build_dataset(cache_path, config, fake_builder)
    assert result == cached


def test_load_or_build_dataset_rebuilds_on_cache_miss(tmp_path):
    cache_path = tmp_path / "cache.json"
    config = {"model": "m", "lang_codes": ["a"], "n_sentences": 1}
    built = {"model": "m", "lang_codes": ["a"], "points": [{"lang": "a", "id": "a-0", "vector": [0.0]}]}

    def fake_builder(lang_codes, n_sentences, model_name):
        assert (lang_codes, n_sentences, model_name) == (["a"], 1, "m")
        return built

    result = server._load_or_build_dataset(cache_path, config, fake_builder)
    assert result == built
    assert json.loads(cache_path.read_text()) == built
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_server.py -v`
Expected: FAIL — `AttributeError: module 'server' has no attribute '_pipeline_config'` (and similarly for the other two new functions).

- [ ] **Step 3: Add the new imports**

In `server.py`, find:

```python
from __future__ import annotations

import math
import sys
import types
from pathlib import Path
from typing import Literal
```

Replace with:

```python
from __future__ import annotations

import json
import math
import os
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Callable, Literal
```

- [ ] **Step 4: Import `embedding_pipeline` and add default config constants**

In `server.py`, find:

```python
import umap
from sklearn.manifold import TSNE

DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "flores_embeddings.json"
```

Replace with:

```python
import umap
from sklearn.manifold import TSNE

import embedding_pipeline

DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "flores_embeddings.json"
DEFAULT_LANGUAGES = "eng_Latn,deu_Latn,fra_Latn,spa_Latn,zho_Hans"
DEFAULT_N_SENTENCES = 200
DEFAULT_MODEL = "xlm-roberta-base"
```

- [ ] **Step 5: Add a generation counter to `DATASET`**

In `server.py`, find:

```python
DATASET: dict = {"data": None}
```

Replace with:

```python
DATASET: dict = {"data": None, "generation": 0}
```

- [ ] **Step 6: Replace the startup logic with the config/cache/build pipeline**

In `server.py`, find:

```python
@app.on_event("startup")
def load_default_dataset() -> None:
    if not DEFAULT_DATA_PATH.exists():
        print(f"No default dataset at {DEFAULT_DATA_PATH} — POST /dataset to load one.")
        return
    import json
    raw = json.loads(DEFAULT_DATA_PATH.read_text())
    DATASET["data"] = validate_and_normalize(raw)
    print(f"Loaded default dataset: {len(DATASET['data']['points'])} points")
```

Replace with:

```python
def _pipeline_config() -> dict:
    langs = os.environ.get("PROJECTION_LANGUAGES", DEFAULT_LANGUAGES)
    return {
        "lang_codes": [c.strip() for c in langs.split(",") if c.strip()],
        "n_sentences": int(os.environ.get("PROJECTION_N_SENTENCES", DEFAULT_N_SENTENCES)),
        "model": os.environ.get("PROJECTION_MODEL", DEFAULT_MODEL),
    }


def _cache_matches(cached: dict, config: dict) -> bool:
    if cached.get("model") != config["model"]:
        return False
    if sorted(cached.get("lang_codes") or []) != sorted(config["lang_codes"]):
        return False
    counts = Counter(p["lang"] for p in cached.get("points", []))
    return all(counts.get(code) == config["n_sentences"] for code in config["lang_codes"])


def _load_or_build_dataset(cache_path: Path, config: dict, builder: Callable) -> dict:
    """builder(lang_codes, n_sentences, model_name) -> dataset dict, same shape as the cache file."""
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if _cache_matches(cached, config):
            return cached

    built = builder(config["lang_codes"], config["n_sentences"], config["model"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(built))
    return built


@app.on_event("startup")
def load_default_dataset() -> None:
    config = _pipeline_config()
    print(f"Preparing default dataset: {config}")
    raw = _load_or_build_dataset(DEFAULT_DATA_PATH, config, embedding_pipeline.build_dataset)
    DATASET["data"] = validate_and_normalize(raw)
    DATASET["generation"] += 1
    print(f"Loaded default dataset: {len(DATASET['data']['points'])} points")
```

- [ ] **Step 7: Increment the generation counter when a dataset is uploaded**

In `server.py`, find:

```python
@app.post("/dataset")
async def set_dataset(raw: dict) -> dict:
    try:
        DATASET["data"] = validate_and_normalize(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    d = DATASET["data"]
    return {"n_points": len(d["points"]), "lang_codes": d["lang_codes"], "dim": d["dim"]}
```

Replace with:

```python
@app.post("/dataset")
async def set_dataset(raw: dict) -> dict:
    try:
        DATASET["data"] = validate_and_normalize(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    DATASET["generation"] += 1
    d = DATASET["data"]
    return {"n_points": len(d["points"]), "lang_codes": d["lang_codes"], "dim": d["dim"]}
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_server.py -v`
Expected: 11 passed.

- [ ] **Step 9: Update the README setup instructions**

In `README.md`, find the `## 2. Get a dataset` and `## 3. Start the backend` sections (everything from `## 2. Get a dataset` up to, but not including, `## 4. Start the frontend`) and replace them with:

```markdown
## 2. One-time Hugging Face setup

The backend computes the default FLORES-200 dataset itself on first
startup, but FLORES-200 requires accepting its terms once per account:

```bash
python3 -c "from huggingface_hub import login; login()"
```
then accept the terms at https://huggingface.co/datasets/facebook/flores

(Skip this if you only plan to upload your own embeddings — see step 4.)

## 3. Start the backend

```bash
python3 -m uvicorn server:app --port 8500
```
Leave this running. On first startup (or whenever the requested config
changes), it computes the default dataset — 5 languages x 200 sentences,
`xlm-roberta-base`, last layer, plus per-token vectors — and caches it to
`data/flores_embeddings.json`; later startups with the same config load
that cache instead (seconds instead of minutes). Serves `/points`,
`/project`, `/project/tokens/{id}`, and `/dataset` on port 8500.

Customize via env vars, e.g. more languages:
```bash
PROJECTION_LANGUAGES=eng_Latn,deu_Latn,fra_Latn,spa_Latn,zho_Hans,jpn_Jpan,kor_Hang \
PROJECTION_N_SENTENCES=200 \
PROJECTION_MODEL=xlm-roberta-base \
    python3 -m uvicorn server:app --port 8500
```
More languages = proportionally longer on a cache miss (each language is a
separate model pass over `PROJECTION_N_SENTENCES` sentences).

```

- [ ] **Step 10: Commit**

```bash
git add server.py README.md tests/test_server.py
git commit -m "Compute or cache the default dataset in server.py instead of a separate script"
```

---

### Task 4: `server.py` — shared UMAP reducer + `/project/tokens/{id}`

**Files:**
- Modify: `server.py`
- Modify: `README.md`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `DATASET["generation"]` (Task 3), `_current_dataset()` (existing).
- Produces: `server._fit_umap(vectors: np.ndarray) -> umap.UMAP`, `server._get_umap_reducer(d: dict) -> umap.UMAP` (fits once per `DATASET["generation"]`, reused after that), and the new endpoint `GET /project/tokens/{point_id}` returning `{"id": ..., "tokens": [{"token": str, "x": float, "y": float}, ...]}`. Task 6 (frontend) calls this endpoint.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_server.py`:

```python
def test_project_umap_refits_only_when_dataset_changes(client, monkeypatch):
    fit_calls = []
    orig_fit = server.umap.UMAP.fit

    def spy_fit(self, X, *a, **kw):
        fit_calls.append(1)
        return orig_fit(self, X, *a, **kw)

    monkeypatch.setattr(server.umap.UMAP, "fit", spy_fit)

    client.get("/project?method=umap")
    client.get("/project?method=umap")
    assert len(fit_calls) == 1  # second call reuses the cached reducer

    client.post("/dataset", json=copy.deepcopy(SAMPLE_DATASET))
    client.get("/project?method=umap")
    assert len(fit_calls) == 2  # dataset change invalidates the cache


def test_project_tokens_success(client):
    res = client.get("/project/tokens/en-1")
    assert res.status_code == 200
    body = res.json()
    assert body["id"] == "en-1"
    assert {t["token"] for t in body["tokens"]} == {"bye", "!"}
    assert all(isinstance(t["x"], float) and isinstance(t["y"], float) for t in body["tokens"])


def test_project_tokens_404_for_unknown_id(client):
    res = client.get("/project/tokens/does-not-exist")
    assert res.status_code == 404


def test_project_tokens_400_when_point_has_no_token_vectors(client):
    res = client.get("/project/tokens/en-0")
    assert res.status_code == 400
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_server.py -v`
Expected: FAIL — `test_project_umap_refits_only_when_dataset_changes` fails because `/project` refits every call (2 fits before the `/dataset` POST, not 1); `/project/tokens/{id}` tests fail with 404 "Not Found" (route doesn't exist).

- [ ] **Step 3: Bump the dataset generation in the test fixture**

The reducer cache (added in the next step) is keyed on `DATASET["generation"]`. Without bumping it per test, a reducer fit by an earlier test could be silently reused for a later test's different dataset. In `tests/test_server.py`, find:

```python
def client():
    server.DATASET["data"] = server.validate_and_normalize(copy.deepcopy(SAMPLE_DATASET))
    return TestClient(server.app)
```

Replace with:

```python
def client():
    server.DATASET["data"] = server.validate_and_normalize(copy.deepcopy(SAMPLE_DATASET))
    server.DATASET["generation"] += 1
    return TestClient(server.app)
```

- [ ] **Step 4: Add a reducer cache next to `DATASET`**

In `server.py`, find:

```python
DATASET: dict = {"data": None, "generation": 0}
```

Replace with:

```python
DATASET: dict = {"data": None, "generation": 0}
_REDUCER_CACHE: dict = {"generation": -1, "reducer": None}
```

- [ ] **Step 5: Replace `project()` to fit-once/reuse for UMAP, and add the token endpoint**

In `server.py`, find:

```python
@app.get("/project")
def project(method: Literal["umap", "tsne"] = "umap") -> dict:
    d = _current_dataset()
    points = d["points"]
    n = len(points)
    if n < 2:
        raise HTTPException(status_code=400, detail="need at least 2 points to project")

    vectors = np.array([p["vector"] for p in points], dtype=np.float32)

    if method == "umap":
        n_neighbors = max(1, min(15, n - 1))
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
    else:
        perplexity = max(1, min(30, n - 1))
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")

    coords = reducer.fit_transform(vectors)

    return {
        "method": method,
        "points": [{"id": p["id"], "x": float(coords[i][0]), "y": float(coords[i][1])} for i, p in enumerate(points)],
    }
```

Replace with:

```python
def _fit_umap(vectors: np.ndarray) -> umap.UMAP:
    n = len(vectors)
    n_neighbors = max(1, min(15, n - 1))
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
    reducer.fit(vectors)
    return reducer


def _get_umap_reducer(d: dict) -> umap.UMAP:
    """Fits once per dataset "generation" and reuses the same reducer after
    that — both so repeated /project?method=umap calls skip refitting, and
    so /project/tokens/{id} can .transform() token vectors into the exact
    same space the sentence points were fit into."""
    if _REDUCER_CACHE["generation"] != DATASET["generation"]:
        vectors = np.array([p["vector"] for p in d["points"]], dtype=np.float32)
        _REDUCER_CACHE["reducer"] = _fit_umap(vectors)
        _REDUCER_CACHE["generation"] = DATASET["generation"]
    return _REDUCER_CACHE["reducer"]


@app.get("/project")
def project(method: Literal["umap", "tsne"] = "umap") -> dict:
    d = _current_dataset()
    points = d["points"]
    n = len(points)
    if n < 2:
        raise HTTPException(status_code=400, detail="need at least 2 points to project")

    if method == "umap":
        reducer = _get_umap_reducer(d)
        coords = reducer.embedding_
    else:
        vectors = np.array([p["vector"] for p in points], dtype=np.float32)
        perplexity = max(1, min(30, n - 1))
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
        coords = reducer.fit_transform(vectors)

    return {
        "method": method,
        "points": [{"id": p["id"], "x": float(coords[i][0]), "y": float(coords[i][1])} for i, p in enumerate(points)],
    }


@app.get("/project/tokens/{point_id}")
def project_tokens(point_id: str) -> dict:
    d = _current_dataset()
    point = next((p for p in d["points"] if str(p["id"]) == point_id), None)
    if point is None:
        raise HTTPException(status_code=404, detail=f"no point with id {point_id!r}")

    token_vectors = point.get("tokenVectors")
    if not token_vectors:
        raise HTTPException(status_code=400, detail=f"point {point_id!r} has no token vectors")
    if len(token_vectors) < 2:
        raise HTTPException(status_code=400, detail="need at least 2 tokens to project")

    reducer = _get_umap_reducer(d)
    vectors = np.array([t["vector"] for t in token_vectors], dtype=np.float32)
    coords = reducer.transform(vectors)

    return {
        "id": point["id"],
        "tokens": [
            {"token": t["token"], "x": float(coords[i][0]), "y": float(coords[i][1])}
            for i, t in enumerate(token_vectors)
        ],
    }
```

- [ ] **Step 6: Update the module docstring's endpoint list**

In `server.py`, find:

```python
Endpoints:
    POST /dataset            body = same JSON shape the frontend's file
                              picker validates client-side; stores it as the
                              current dataset.
    GET  /points             metadata only (no vectors) for rendering.
    GET  /project?method=... 2D coordinates, computed on the stored vectors.
"""
```

Replace with:

```python
Endpoints:
    POST /dataset                body = same JSON shape the frontend's file
                                  picker validates client-side; stores it as
                                  the current dataset.
    GET  /points                 metadata only (no vectors) for rendering.
    GET  /project?method=...     2D coordinates, computed on the stored
                                  sentence vectors.
    GET  /project/tokens/{id}    2D coordinates for one point's token
                                  vectors, embedded into the same UMAP space
                                  as /project?method=umap via .transform().
"""
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_server.py -v`
Expected: 15 passed.

- [ ] **Step 8: Update the README project layout**

In `README.md`, find:

```
index.html                          # page shell, loads client-js/app.js as a module
css/style.css                       # all styling
client-js/
  app.js                            # orchestrator: selection state, wiring
  projection-view.js                # composes the four view collaborators below
  point-layer.js                    # point data, color/opacity, canvas painting
  zoomable-axes.js                  # SVG axis overlay + d3.zoom
  tooltip.js                        # hover tooltip
  lang-filter.js                    # searchable multi-select language filter
  backend-client.js                 # fetch wrapper for server.py's endpoints
  detail-panel.js                   # sidebar detail-on-click panel
server.py                           # FastAPI backend: /points, /project, /dataset
scripts/prepare_flores_embeddings.py  # FLORES-200 + XLM-R embedding prep pipeline
requirements.txt
```

Replace with:

```
index.html                          # page shell, loads client-js/app.js as a module
css/style.css                       # all styling
client-js/
  app.js                            # orchestrator: selection state, wiring
  projection-view.js                # composes the four view collaborators below
  point-layer.js                    # point data, color/opacity, canvas painting
  zoomable-axes.js                  # SVG axis overlay + d3.zoom
  tooltip.js                        # hover tooltip
  lang-filter.js                    # searchable multi-select language filter
  backend-client.js                 # fetch wrapper for server.py's endpoints
  detail-panel.js                   # sidebar detail-on-click panel
server.py                           # FastAPI backend: /points, /project, /project/tokens/{id}, /dataset
embedding_pipeline.py                # FLORES-200 + XLM-R embedding pipeline (sentence + token vectors)
tests/                               # pytest suite for embedding_pipeline.py and server.py
requirements.txt
requirements-dev.txt                # adds pytest, for running tests/
```

- [ ] **Step 9: Commit**

```bash
git add server.py README.md tests/test_server.py
git commit -m "Add shared-space UMAP reducer caching and GET /project/tokens/{id}"
```

---

### Task 5: Frontend — backend client, DOM elements, data plumbing

**Files:**
- Modify: `client-js/backend-client.js`
- Modify: `index.html`
- Modify: `css/style.css`
- Modify: `client-js/app.js`

**Interfaces:**
- Consumes: `GET /project/tokens/{id}` (Task 4), `hasTokens` field from `GET /points` (Task 2).
- Produces: `backend.fetchTokenProjection(id) -> Promise<{id, tokens: [{token, x, y}]}>`; `this.points` entries in `app.js` gain a `hasTokens: boolean` field; DOM element `#detail-tokens-toggle` exists (hidden by default). Task 6 wires the click behavior and rendering.

- [ ] **Step 1: Add `fetchTokenProjection` to the backend client**

In `client-js/backend-client.js`, find:

```js
  fetchProjection(method) {
    return this._request(`/project?method=${method}`);
  }
```

Replace with:

```js
  fetchProjection(method) {
    return this._request(`/project?method=${method}`);
  }

  fetchTokenProjection(id) {
    return this._request(`/project/tokens/${encodeURIComponent(id)}`);
  }
```

- [ ] **Step 2: Add the "Show tokens" button and format-help doc to `index.html`**

Find:

```html
<pre>{
  "model": "any string (informational)",
  "layer": "any string (informational)",
  "dim": 768,
  "lang_codes": ["en", "de"],
  "points": [
    {
      "id": "unique-string-or-number",
      "lang": "en",
      "text": "shown in tooltip + detail panel",
      "vector": [0.1, -0.4, ...],
      "pairId": 0
    }
  ]
}</pre>
```

Replace with:

```html
<pre>{
  "model": "any string (informational)",
  "layer": "any string (informational)",
  "dim": 768,
  "lang_codes": ["en", "de"],
  "points": [
    {
      "id": "unique-string-or-number",
      "lang": "en",
      "text": "shown in tooltip + detail panel",
      "vector": [0.1, -0.4, ...],
      "pairId": 0,
      "tokenVectors": [
        {"token": "shown", "vector": [0.2, -0.1, ...]}
      ]
    }
  ]
}</pre>
        <p><code>tokenVectors</code> is optional — if present, a "Show tokens" button appears when this point is selected, projecting its tokens into the same UMAP space as the main plot.</p>
```

Find:

```html
    <div id="detail-placeholder">Click a point to see its translations</div>
    <div id="detail-lang"></div>
    <div id="detail-text"></div>
    <div id="detail-translations"></div>
```

Replace with:

```html
    <div id="detail-placeholder">Click a point to see its translations</div>
    <div id="detail-lang"></div>
    <div id="detail-text"></div>
    <div id="detail-translations"></div>
    <button type="button" id="detail-tokens-toggle">Show tokens</button>
```

- [ ] **Step 3: Style the new button**

In `css/style.css`, find:

```css
#detail-translations { font-size: 0.8em; color: #555; line-height: 1.6; display: none; margin-top: 12px; }
#detail-translations b { display: block; margin-bottom: 4px; }
```

Replace with:

```css
#detail-translations { font-size: 0.8em; color: #555; line-height: 1.6; display: none; margin-top: 12px; }
#detail-translations b { display: block; margin-bottom: 4px; }

#detail-tokens-toggle { display: none; margin-top: 12px; font-size: 0.78em; padding: 4px 10px; cursor: pointer; }
```

- [ ] **Step 4: Thread `hasTokens` through `app.js`'s point mapping**

In `client-js/app.js`, find:

```js
    this.points = meta.points.map(p => {
      const c = coordsById.get(p.id);
      return {
        id: p.id, lang: p.lang, text: p.text,
        // no pairId at all -> each point is its own group, so it isolates alone on click
        groupKey: p.pairId != null ? `pair:${p.pairId}` : `id:${p.id}`,
        x: c.x, y: c.y,
      };
    });
```

Replace with:

```js
    this.points = meta.points.map(p => {
      const c = coordsById.get(p.id);
      return {
        id: p.id, lang: p.lang, text: p.text, hasTokens: p.hasTokens,
        // no pairId at all -> each point is its own group, so it isolates alone on click
        groupKey: p.pairId != null ? `pair:${p.pairId}` : `id:${p.id}`,
        x: c.x, y: c.y,
      };
    });
```

- [ ] **Step 5: Manually verify the button appears (inert) and data flows**

Start the backend and frontend per the README, open the app in a browser, click a point, and confirm: if backend's default dataset was regenerated with Task 1-4 changes (has `tokenVectors`), the "Show tokens" button is visible under the translations panel for at least the points that have them, and absent/hidden for points without. It does nothing yet when clicked — that's Task 6.

- [ ] **Step 6: Commit**

```bash
git add client-js/backend-client.js index.html css/style.css client-js/app.js
git commit -m "Wire up data plumbing for per-point token projection"
```

---

### Task 6: Frontend — token overlay rendering and toggle interaction

**Files:**
- Modify: `client-js/point-layer.js`
- Modify: `client-js/projection-view.js`
- Modify: `client-js/detail-panel.js`
- Modify: `client-js/app.js`
- Modify: `README.md`

**Interfaces:**
- Consumes: `backend.fetchTokenProjection(id)` (Task 5), `hasTokens` on points (Task 5).
- Produces: `PointLayer.setTokenOverlay(tokenPoints: {id, text, x, y}[])`, `PointLayer.clearTokenOverlay()`, `PointLayer.tokenOverlayPointAt(x, y, radius) -> point | undefined`; `DetailPanel.setTokensShown(shown: boolean)`; `App.onToggleTokens()`. No further tasks depend on these — this is the last task.

- [ ] **Step 1: Add overlay storage and drawing to `PointLayer`**

In `client-js/point-layer.js`, find:

```js
  constructor(onChange) {
    this._onChange = onChange; // called after every opacity update, so the owner redraws
    this.points = [];
    this.baseXScale = null;
    this.baseYScale = null;
    this._timer = null;
    this._activeGroupKey = null;
    this._selectedLangs = new Set();
    this._langColorScale = null;
  }

  setPoints(points, baseXScale, baseYScale) {
    this._activeGroupKey = null;
    this._selectedLangs = new Set();
    this._langColorScale = null;
    this.points = points.map(p => ({ ...p, opacity: 1 }));
    this.baseXScale = baseXScale;
    this.baseYScale = baseYScale;
  }
```

Replace with:

```js
  constructor(onChange) {
    this._onChange = onChange; // called after every opacity update, so the owner redraws
    this.points = [];
    this.baseXScale = null;
    this.baseYScale = null;
    this._timer = null;
    this._activeGroupKey = null;
    this._selectedLangs = new Set();
    this._langColorScale = null;
    this._tokenOverlay = [];
  }

  setPoints(points, baseXScale, baseYScale) {
    this._activeGroupKey = null;
    this._selectedLangs = new Set();
    this._langColorScale = null;
    this._tokenOverlay = [];
    this.points = points.map(p => ({ ...p, opacity: 1 }));
    this.baseXScale = baseXScale;
    this.baseYScale = baseYScale;
  }

  // Token points for one currently-selected sentence, embedded into the
  // same UMAP space as `this.points` via the backend's shared reducer —
  // drawn on the same axes but tracked separately so they're unaffected by
  // the isolate-fade opacity animation and don't participate in click-to-
  // isolate grouping.
  setTokenOverlay(tokenPoints) {
    this._tokenOverlay = tokenPoints;
    this._onChange();
  }

  clearTokenOverlay() {
    this._tokenOverlay = [];
    this._onChange();
  }

  tokenOverlayPointAt(x, y, radius) {
    for (const d of this._tokenOverlay) {
      const dx = this.baseXScale(d.x) - x;
      const dy = this.baseYScale(d.y) - y;
      if (Math.hypot(dx, dy) <= radius) return d;
    }
    return undefined;
  }
```

- [ ] **Step 2: Draw the overlay as diamonds after the sentence points**

In `client-js/point-layer.js`, find:

```js
      ctx.globalAlpha = d.opacity;
      ctx.lineWidth = (hovered ? 1.5 : 0.5) / transform.k;
      ctx.strokeStyle = hovered ? '#333' : '#fff';
      ctx.stroke();
    }
    ctx.restore();
    ctx.globalAlpha = 1;
  }
```

Replace with:

```js
      ctx.globalAlpha = d.opacity;
      ctx.lineWidth = (hovered ? 1.5 : 0.5) / transform.k;
      ctx.strokeStyle = hovered ? '#333' : '#fff';
      ctx.stroke();
    }
    for (const d of this._tokenOverlay) {
      const cx = this.baseXScale(d.x), cy = this.baseYScale(d.y);
      const size = 5 / transform.k;
      ctx.beginPath();
      ctx.moveTo(cx, cy - size);
      ctx.lineTo(cx + size, cy);
      ctx.lineTo(cx, cy + size);
      ctx.lineTo(cx - size, cy);
      ctx.closePath();
      ctx.globalAlpha = 0.9;
      ctx.fillStyle = '#e63946';
      ctx.fill();
      ctx.lineWidth = 0.75 / transform.k;
      ctx.strokeStyle = '#fff';
      ctx.stroke();
    }
    ctx.restore();
    ctx.globalAlpha = 1;
  }
```

- [ ] **Step 3: Make token overlay points hoverable (tooltip) and unclickable in `ProjectionView`**

In `client-js/projection-view.js`, find:

```js
  _onCanvasClick(event) {
    const d = this._pointAt(event);
    if (d) {
      this._onClickPoint?.(d);
    } else {
      this._onBackgroundClick?.();
    }
  }

  _onCanvasMouseMove(event) {
    const d = this._pointAt(event);
    const hoveredId = d ? d.id : null;
    if (hoveredId !== this._hoveredId) {
      this._hoveredId = hoveredId;
      this._draw();
    }
    if (d) {
      const rect = this.el.getBoundingClientRect();
      this.tooltip.show(event.clientX - rect.left, event.clientY - rect.top, `${d.lang}: ${d.text}`);
      this.canvas.style.cursor = 'pointer';
    } else {
      this._hideTooltip();
    }
  }
```

Replace with:

```js
  _overlayPointAt(event) {
    const rect = this.canvas.getBoundingClientRect();
    const screenX = event.clientX - rect.left - this.margin.left;
    const screenY = event.clientY - rect.top - this.margin.top;
    const [x, y] = this.axes.transform.invert([screenX, screenY]);
    return this.pointLayer.tokenOverlayPointAt(x, y, 8 / this.axes.transform.k);
  }

  _onCanvasClick(event) {
    if (this._overlayPointAt(event)) return; // token overlay points aren't clickable, just hoverable
    const d = this._pointAt(event);
    if (d) {
      this._onClickPoint?.(d);
    } else {
      this._onBackgroundClick?.();
    }
  }

  _onCanvasMouseMove(event) {
    const overlayHit = this._overlayPointAt(event);
    if (overlayHit) {
      const rect = this.el.getBoundingClientRect();
      this.tooltip.show(event.clientX - rect.left, event.clientY - rect.top, overlayHit.text);
      this.canvas.style.cursor = 'default';
      if (this._hoveredId !== null) {
        this._hoveredId = null;
        this._draw();
      }
      return;
    }

    const d = this._pointAt(event);
    const hoveredId = d ? d.id : null;
    if (hoveredId !== this._hoveredId) {
      this._hoveredId = hoveredId;
      this._draw();
    }
    if (d) {
      const rect = this.el.getBoundingClientRect();
      this.tooltip.show(event.clientX - rect.left, event.clientY - rect.top, `${d.lang}: ${d.text}`);
      this.canvas.style.cursor = 'pointer';
    } else {
      this._hideTooltip();
    }
  }
```

- [ ] **Step 4: Add the toggle button element and state to `DetailPanel`**

Replace the full contents of `client-js/detail-panel.js` with:

```js
// The sidebar detail panel shown when a point is selected alongside its
// translations and an optional per-point token-embedding toggle.

export class DetailPanel {
  constructor({ placeholderId, langId, textId, translationsId, tokensToggleId }) {
    this.placeholder = document.getElementById(placeholderId);
    this.langEl = document.getElementById(langId);
    this.textEl = document.getElementById(textId);
    this.transEl = document.getElementById(translationsId);
    this.tokensToggle = document.getElementById(tokensToggleId);
  }

  show(point, translations) {
    this.placeholder.style.display = 'none';

    this.langEl.style.display = 'block';
    this.langEl.textContent = point.lang;

    this.textEl.style.display = 'block';
    this.textEl.textContent = point.text;

    if (translations.length === 0) {
      this.transEl.style.display = 'none';
      this.transEl.innerHTML = '';
    } else {
      this.transEl.style.display = 'block';
      this.transEl.innerHTML = '<b>Translations</b>' +
        translations.map(p => `<div><i>${p.lang}</i>: ${p.text}</div>`).join('');
    }

    this.tokensToggle.style.display = point.hasTokens ? 'block' : 'none';
    this.setTokensShown(false);
  }

  clear() {
    this.placeholder.style.display = 'block';
    [this.langEl, this.textEl, this.transEl].forEach(el => {
      el.style.display = 'none';
      el.textContent = '';
    });
    this.tokensToggle.style.display = 'none';
  }

  setTokensShown(shown) {
    this.tokensToggle.textContent = shown ? 'Hide tokens' : 'Show tokens';
  }
}
```

- [ ] **Step 5: Wire the toggle in `App`**

In `client-js/app.js`, find:

```js
    this.detail = new DetailPanel({
      placeholderId: 'detail-placeholder',
      langId: 'detail-lang',
      textId: 'detail-text',
      translationsId: 'detail-translations',
    });
```

Replace with:

```js
    this.detail = new DetailPanel({
      placeholderId: 'detail-placeholder',
      langId: 'detail-lang',
      textId: 'detail-text',
      translationsId: 'detail-translations',
      tokensToggleId: 'detail-tokens-toggle',
    });
```

Find:

```js
    this.points = [];
    this._selectedGroupKey = null;
  }
```

Replace with:

```js
    this.points = [];
    this._selectedGroupKey = null;
    this._selectedPoint = null;
    this._tokenOverlayPointId = null;
  }
```

Find:

```js
    document.getElementById('format-help-toggle')
      .addEventListener('click', () => this._toggleFormatHelp());

    this._refresh();
```

Replace with:

```js
    document.getElementById('format-help-toggle')
      .addEventListener('click', () => this._toggleFormatHelp());
    document.getElementById('detail-tokens-toggle')
      .addEventListener('click', () => this.onToggleTokens());

    this._refresh();
```

Find:

```js
    this._selectedGroupKey = null;
    this.view.clearHighlight();
    this.detail.clear();

    const coordsById = new Map(proj.points.map(p => [p.id, p]));
```

Replace with:

```js
    this._selectedGroupKey = null;
    this._selectedPoint = null;
    this._tokenOverlayPointId = null;
    this.view.clearHighlight();
    this.detail.clear();

    const coordsById = new Map(proj.points.map(p => [p.id, p]));
```

Find:

```js
  onClickPoint(d) {
    if (this._selectedGroupKey === d.groupKey) {
      this.onBackgroundClick();
      return;
    }
    this._selectedGroupKey = d.groupKey;
    this.view.highlightParallel(d.groupKey);
    const translations = this.points.filter(p => p.groupKey === d.groupKey && p.id !== d.id);
    this.detail.show(d, translations);
  }

  onBackgroundClick() {
    if (this._selectedGroupKey === null) return;
    this._selectedGroupKey = null;
    this.view.clearHighlight();
    this.detail.clear();
  }
}
```

Replace with:

```js
  onClickPoint(d) {
    if (this._selectedGroupKey === d.groupKey) {
      this.onBackgroundClick();
      return;
    }
    this._clearTokenOverlay();
    this._selectedGroupKey = d.groupKey;
    this._selectedPoint = d;
    this.view.highlightParallel(d.groupKey);
    const translations = this.points.filter(p => p.groupKey === d.groupKey && p.id !== d.id);
    this.detail.show(d, translations);
  }

  onBackgroundClick() {
    if (this._selectedGroupKey === null) return;
    this._clearTokenOverlay();
    this._selectedGroupKey = null;
    this._selectedPoint = null;
    this.view.clearHighlight();
    this.detail.clear();
  }

  // Toggles the token overlay for the currently selected point. Token
  // projection only exists in the shared UMAP space (see server.py's
  // _get_umap_reducer), so if the main view is on t-SNE this switches it to
  // UMAP first — re-running _refresh() resets selection state, so the
  // point is re-selected afterward by id before fetching its tokens.
  async onToggleTokens() {
    if (!this._selectedPoint) return;
    const pointId = this._selectedPoint.id;

    if (this._tokenOverlayPointId === pointId) {
      this._clearTokenOverlay();
      return;
    }

    const methodSelect = document.getElementById('method-select');
    if (methodSelect.value !== 'umap') {
      methodSelect.value = 'umap';
      await this._refresh();
      const reselected = this.points.find(p => p.id === pointId);
      if (!reselected) return;
      this.onClickPoint(reselected);
      document.getElementById('status').textContent += ' · switched to UMAP to show tokens';
    }

    let proj;
    try {
      proj = await this.backend.fetchTokenProjection(pointId);
    } catch (err) {
      document.getElementById('status').textContent = err.message;
      return;
    }

    this._tokenOverlayPointId = pointId;
    this.detail.setTokensShown(true);
    this.view.setTokenOverlay(proj.tokens.map((t, i) => ({
      id: `${pointId}::${i}`,
      text: t.token,
      x: t.x,
      y: t.y,
    })));
  }

  _clearTokenOverlay() {
    if (this._tokenOverlayPointId === null) return;
    this._tokenOverlayPointId = null;
    this.detail.setTokensShown(false);
    this.view.clearTokenOverlay();
  }
}
```

- [ ] **Step 6: Add view methods App relies on**

In `client-js/projection-view.js`, find:

```js
  highlightParallel(groupKey) { this.pointLayer.highlightGroup(groupKey); }
  clearHighlight() { this.pointLayer.clear(); }
  setLangFilter(langs) { this.pointLayer.setLangFilter(langs); }
```

Replace with:

```js
  highlightParallel(groupKey) { this.pointLayer.highlightGroup(groupKey); }
  clearHighlight() { this.pointLayer.clear(); }
  setLangFilter(langs) { this.pointLayer.setLangFilter(langs); }
  setTokenOverlay(tokenPoints) { this.pointLayer.setTokenOverlay(tokenPoints); }
  clearTokenOverlay() { this.pointLayer.clearTokenOverlay(); }
```

- [ ] **Step 7: Manually verify end to end**

Use the `run` skill (or manually): start the backend and frontend per the README, open the app, click a point that has `tokenVectors` (its "Show tokens" button is visible), click "Show tokens", and confirm: small red diamonds appear near that point's language cluster, hovering one shows its token text in the tooltip, clicking one does nothing, clicking "Hide tokens" removes them, and selecting a different point clears the previous overlay. Then switch the projection method to t-SNE, select a point, click "Show tokens" again, and confirm the method dropdown auto-switches back to UMAP and the overlay still appears.

- [ ] **Step 8: Add the "Show tokens" usage bullet to the README**

In `README.md`, find:

```
- **Projection method** — switch between UMAP and t-SNE; re-runs server-side.
```

Replace with:

```
- **Projection method** — switch between UMAP and t-SNE; re-runs server-side.
- **Show/hide tokens** — with a point selected, click "Show tokens" in the
  sidebar to see that sentence's individual token embeddings projected into
  the same UMAP space (auto-switches the projection method to UMAP if
  needed, since t-SNE can't share axes with a second projection). Click
  again to hide them.
```

- [ ] **Step 9: Commit**

```bash
git add client-js/point-layer.js client-js/projection-view.js client-js/detail-panel.js client-js/app.js README.md
git commit -m "Overlay a selected point's tokens in the shared UMAP space"
```
