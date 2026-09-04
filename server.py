"""
Projection backend for projection-app/.

Holds one dataset in memory (points with id/lang/pairId/text/vector) and
serves 2D projections of it on demand, computed server-side with umap-learn
or scikit-learn's TSNE. Run alongside the static frontend:

    uvicorn server:app --port 8500

TODO: dockerize
TODO: modularize and separate -> services layer? CRUD?
TODO: check edge cases, consider pagination...

Endpoints:
    POST /dataset            body = same JSON shape the frontend's file
                              picker validates client-side; stores it as the
                              current dataset.
    GET  /points             metadata only (no vectors) for rendering.
    GET  /project?method=... 2D coordinates, computed on the stored vectors.
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path
from typing import Literal

# umap-learn unconditionally imports its optional ParametricUMAP submodule,
# which unconditionally does `import tensorflow`. On machines where the
# installed tensorflow hangs on import (see scripts/prepare_flores_embeddings.py
# for the same issue with transformers), this stub makes that import resolve
# instantly instead of hanging — umap/__init__.py already falls back to a
# dummy ParametricUMAP class when tensorflow isn't usable, we just need the
# import itself to not hang.
sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import umap
from sklearn.manifold import TSNE

DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "flores_embeddings.json"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev tool, no auth/sensitive deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASET: dict = {"data": None}


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
        "dim": dim,
        "lang_codes": raw.get("lang_codes") or sorted({p["lang"] for p in points}),
        "points": points,
    }


@app.on_event("startup")
def load_default_dataset() -> None:
    if not DEFAULT_DATA_PATH.exists():
        print(f"No default dataset at {DEFAULT_DATA_PATH} — POST /dataset to load one.")
        return
    import json
    raw = json.loads(DEFAULT_DATA_PATH.read_text())
    DATASET["data"] = validate_and_normalize(raw)
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
    d = DATASET["data"]
    return {"n_points": len(d["points"]), "lang_codes": d["lang_codes"], "dim": d["dim"]}


@app.get("/points")
def get_points() -> dict:
    d = _current_dataset()
    return {
        "model": d["model"],
        "layer": d["layer"],
        "dim": d["dim"],
        "lang_codes": d["lang_codes"],
        "points": [
            {
                "id": p["id"], "lang": p["lang"], "pairId": p.get("pairId"), "text": p.get("text"),
                "hasTokens": p.get("tokenVectors") is not None,
            }
            for p in d["points"]
        ],
    }


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
