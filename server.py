"""
FastAPI backend for projection-app: holds one embeddings dataset in memory
and serves 2D UMAP/t-SNE projections of it on demand. Run alongside the
static frontend: uvicorn server:app --port 8500

Endpoints:
    POST /dataset                 load a dataset (JSON shape the frontend validates).
    GET  /points                  point metadata only, no vectors.
    GET  /project                 2D coords for all points (?method=umap|tsne&metric=cosine|euclidean|manhattan).
    GET  /project/tokens/{id}     2D coords for one point's tokens, in the same UMAP space (?metric= must match).
    GET  /neighbors/{id}          nearest neighbors by full-vector distance (?k=&metric=).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Callable, Literal

# umap-learn unconditionally imports its optional ParametricUMAP submodule,
# which unconditionally does `import tensorflow`. On machines where the
# installed tensorflow hangs on import (see embedding_pipeline.py for the
# same issue with transformers), this stub makes that import resolve
# instantly instead of hanging — umap/__init__.py already falls back to a
# dummy ParametricUMAP class when tensorflow isn't usable, we just need the
# import itself to not hang.
if "tensorflow" not in sys.modules:
    tf_stub = types.ModuleType("tensorflow")
    # Create a proper spec so torch's dynamic import doesn't fail
    spec = importlib.util.spec_from_loader("tensorflow", loader=None)
    tf_stub.__spec__ = spec
    sys.modules["tensorflow"] = tf_stub

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import umap
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

import embedding_pipeline

DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "flores_embeddings.json"
# The 40 languages FLORES-200 covers with reasonable coverage across major
# scripts/families; override with PROJECTION_LANGUAGES for a smaller/faster set.
DEFAULT_LANGUAGES = (
    "eng_Latn,zho_Hans,zho_Hant,spa_Latn,fra_Latn,deu_Latn,jpn_Jpan,kor_Hang,rus_Cyrl,por_Latn,"
    "ita_Latn,nld_Latn,pol_Latn,tur_Latn,vie_Latn,ind_Latn,tha_Thai,ukr_Cyrl,swe_Latn,ces_Latn,"
    "ron_Latn,hun_Latn,fin_Latn,dan_Latn,heb_Hebr,arb_Arab,pes_Arab,urd_Arab,hin_Deva,ben_Beng,"
    "tam_Taml,tel_Telu,mal_Mlym,kan_Knda,guj_Gujr,pan_Guru,sin_Sinh,mya_Mymr,khm_Khmr,amh_Ethi"
)
DEFAULT_N_SENTENCES = 200
DEFAULT_MODEL = "xlm-roberta-base"
DEFAULT_POOLING = "mean"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev tool, no auth/sensitive deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASET: dict = {"data": None, "generation": 0}
_REDUCER_CACHE: dict = {"generation": -1, "metric": None, "reducer": None}


def validate_and_normalize(raw: dict) -> dict:
    """Mirrors projection-app/client-js/app.client-js's validateAndNormalize exactly."""
    if not isinstance(raw, dict) or not isinstance(raw.get("points"), list) or len(raw["points"]) == 0:
        raise ValueError('expected an object with a non-empty "points" array')

    points = raw["points"]
    first_vector = points[0].get("vector")
    dim = raw.get("dim") or (len(first_vector) if isinstance(first_vector, list) else None)

    for i, p in enumerate(points):
        if not isinstance(p.get("id"), (str, int, float)):
            raise ValueError(f'points[{i}] is missing a valid "id"')
        if not isinstance(p.get("lang"), str):
            raise ValueError(f'points[{i}] is missing a valid "lang"')
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
        "model": raw.get("model") or "unknown model",
        "layer": raw.get("layer") or "unknown",
        "pooling": raw.get("pooling") or "mean",
        "dim": dim,
        "lang_codes": raw.get("lang_codes") or sorted({p["lang"] for p in points}),
        "points": points,
    }


def _pipeline_config() -> dict:
    langs = os.environ.get("PROJECTION_LANGUAGES", DEFAULT_LANGUAGES)
    return {
        "lang_codes": [c.strip() for c in langs.split(",") if c.strip()],
        "n_sentences": int(os.environ.get("PROJECTION_N_SENTENCES", DEFAULT_N_SENTENCES)),
        "model": os.environ.get("PROJECTION_MODEL", DEFAULT_MODEL),
        "pooling": os.environ.get("PROJECTION_POOLING", DEFAULT_POOLING),
    }


def _cache_matches(cached: dict, config: dict) -> bool:
    if cached.get("model") != config["model"]:
        return False
    if cached.get("pooling", "mean") != config["pooling"]:
        return False
    if sorted(cached.get("lang_codes") or []) != sorted(config["lang_codes"]):
        return False
    points = cached.get("points", [])
    if not points or points[0].get("tokenVectors") is None:
        return False
    counts = Counter(p["lang"] for p in points)
    return all(counts.get(code) == config["n_sentences"] for code in config["lang_codes"])


def _load_or_build_dataset(cache_path: Path, config: dict, builder: Callable) -> dict:
    """builder(lang_codes, n_sentences, model_name, pooling) -> dataset dict, same shape as the cache file."""
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if _cache_matches(cached, config):
            return cached

    built = builder(config["lang_codes"], config["n_sentences"], config["model"], config["pooling"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(built))
    return built


@app.on_event("startup")
def load_default_dataset() -> None:
    config = _pipeline_config()
    print(f"Preparing default dataset: {config}")
    try:
        raw = _load_or_build_dataset(DEFAULT_DATA_PATH, config, embedding_pipeline.build_dataset)
    except Exception as e:
        print(f"Could not prepare default dataset ({e}) — POST /dataset to load one.")
        return
    DATASET["data"] = validate_and_normalize(raw)
    DATASET["generation"] += 1
    print(f"Loaded default dataset: {len(DATASET['data']['points'])} points")


def _current_dataset() -> dict:
    if DATASET["data"] is None:
        raise HTTPException(status_code=404, detail="No dataset loaded yet — POST /dataset first")
    return DATASET["data"]


@app.post("/dataset")
async def set_dataset(raw: dict) -> dict:
    try:
        DATASET["data"] = validate_and_normalize(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    DATASET["generation"] += 1
    d = DATASET["data"]
    return {"n_points": len(d["points"]), "lang_codes": d["lang_codes"], "dim": d["dim"]}


@app.get("/points")
def get_points() -> dict:
    d = _current_dataset()
    return {
        "model": d["model"],
        "layer": d["layer"],
        "pooling": d.get("pooling", "mean"),
        "dim": d["dim"],
        "lang_codes": d["lang_codes"],
        "points": [
            {
                "id": p["id"], "lang": p["lang"], "pairId": p.get("pairId"), "text": p.get("text"),
                "hasTokens": p.get("tokenVectors") is not None,
                "tokenTexts": [t["token"] for t in p["tokenVectors"]] if p.get("tokenVectors") is not None else None,
            }
            for p in d["points"]
        ],
    }


def _fit_umap(vectors: np.ndarray, metric: str) -> umap.UMAP:
    n = len(vectors)
    n_neighbors = max(1, min(15, n - 1))
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42, metric=metric)
    reducer.fit(vectors)
    return reducer


def _get_umap_reducer(d: dict, metric: str) -> umap.UMAP:
    """Fits once per (dataset "generation", metric) and reuses the same
    reducer after that — both so repeated /project?method=umap calls with
    the same metric skip refitting, and so /project/tokens/{id} can
    .transform() token vectors into the exact same space the sentence
    points were fit into for that metric."""
    if _REDUCER_CACHE["generation"] != DATASET["generation"] or _REDUCER_CACHE["metric"] != metric:
        vectors = np.array([p["vector"] for p in d["points"]], dtype=np.float32)
        _REDUCER_CACHE["reducer"] = _fit_umap(vectors, metric)
        _REDUCER_CACHE["generation"] = DATASET["generation"]
        _REDUCER_CACHE["metric"] = metric
    return _REDUCER_CACHE["reducer"]


@app.get("/project")
def project(
    method: Literal["umap", "tsne"] = "umap",
    metric: Literal["cosine", "euclidean", "manhattan"] = "euclidean",
) -> dict:
    d = _current_dataset()
    points = d["points"]
    n = len(points)
    if n < 2:
        raise HTTPException(status_code=400, detail="need at least 2 points to project")

    if method == "umap":
        reducer = _get_umap_reducer(d, metric)
        coords = reducer.embedding_
    else:
        vectors = np.array([p["vector"] for p in points], dtype=np.float32)
        perplexity = max(1, min(30, n - 1))
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", metric=metric)
        coords = reducer.fit_transform(vectors)

    return {
        "method": method,
        "points": [{"id": p["id"], "x": float(coords[i][0]), "y": float(coords[i][1])} for i, p in enumerate(points)],
    }


@app.get("/neighbors/{point_id}")
def neighbors(
    point_id: str,
    k: int = Query(default=5, ge=1, le=100),
    metric: Literal["cosine", "euclidean", "manhattan"] = "euclidean",
) -> dict:
    """Nearest neighbors by distance in the full embedding space (not the 2D
    projection) — independent of whatever method/metric the plot is using."""
    d = _current_dataset()
    points = d["points"]
    idx = next((i for i, p in enumerate(points) if str(p["id"]) == point_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail=f"no point with id {point_id!r}")

    vectors = np.array([p["vector"] for p in points], dtype=np.float32)
    dists = pairwise_distances(vectors[idx:idx + 1], vectors, metric=metric)[0]
    order = [i for i in np.argsort(dists) if i != idx][:k]

    return {
        "id": points[idx]["id"],
        "neighbors": [
            {
                "id": points[i]["id"], "lang": points[i]["lang"], "text": points[i].get("text"),
                "distance": float(dists[i]),
            }
            for i in order
        ],
    }


@app.get("/project/tokens/{point_id}")
def project_tokens(point_id: str, metric: Literal["cosine", "euclidean", "manhattan"] = "euclidean") -> dict:
    d = _current_dataset()
    point = next((p for p in d["points"] if str(p["id"]) == point_id), None)
    if point is None:
        raise HTTPException(status_code=404, detail=f"no point with id {point_id!r}")

    token_vectors = point.get("tokenVectors")
    if not token_vectors:
        raise HTTPException(status_code=400, detail=f"point {point_id!r} has no token vectors")
    if len(token_vectors) < 2:
        raise HTTPException(status_code=400, detail="need at least 2 tokens to project")

    reducer = _get_umap_reducer(d, metric)
    vectors = np.array([t["vector"] for t in token_vectors], dtype=np.float32)
    coords = reducer.transform(vectors)

    return {
        "id": point["id"],
        "tokens": [
            {"token": t["token"], "x": float(coords[i][0]), "y": float(coords[i][1])}
            for i, t in enumerate(token_vectors)
        ],
    }
