"""
FastAPI backend for projection-app: holds one embeddings dataset in memory
and serves 2D UMAP/t-SNE projections of it on demand. Run alongside the
static frontend: uvicorn server:app --port 8500

Endpoints:
    POST /dataset                 load a dataset (JSON shape the frontend validates).
    POST /dataset/upload          load a JSON, JSONL or CSV file (?format=).
    GET  /points                  point metadata only, no vectors (includes each point's language family).
    POST /dataset/flores          embed (or load from cache) a FLORES-200 subset; POST /dataset/flores/cancel stops it.
    GET  /project                 2D (or 3D, ?n_components=3) coords for all points (?method=umap|tsne&metric=&n_neighbors=&min_dist=&perplexity=&learning_rate=).
    GET  /project/tokens/{id}     coords for one point's tokens, in the same UMAP space (?metric=/n_neighbors=/min_dist=/n_components= must match).
    POST /stats                   aggregate cosine/alignment statistics for a set of selected point ids.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import logging
import math
import re
import sys
import threading
import types
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Literal, Optional, Union

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
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import umap
from sklearn.manifold import TSNE

import embedding_pipeline
from dataset_formats import parse_dataset_text

DATA_DIR = Path(__file__).parent / "data"
Metric = Literal["euclidean", "cosine", "manhattan"]
# FLORES-200 devtest has 1012 sentences per language and 204 language/script variants.
FLORES_DEVTEST_SIZE = 1012
MAX_FLORES_LANGUAGES = 204
FLORES_CODE_RE = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")
POOLINGS = ("mean", "max", "cls")
DEFAULT_MODEL = "xlm-roberta-base"
DEFAULT_POOLING = "mean"

FAMILIES_PATH = Path(__file__).parent / "resources" / "language_families.json"


@lru_cache(maxsize=1)
def _glottolog_data() -> dict:
    """The generated lookup: {"families": iso3 -> family, "aliases": 639-1
    code / English name -> iso3}. A missing file just means every language
    falls back to "Other" rather than failing the server."""
    try:
        data = json.loads(FAMILIES_PATH.read_text(encoding="utf-8"))
        return {"families": data["families"], "aliases": data.get("aliases", {})}
    except (OSError, ValueError, KeyError):
        return {"families": {}, "aliases": {}}


def _lookup_family(lang_code: str) -> Optional[str]:
    data = _glottolog_data()
    whole = " ".join(lang_code.lower().split())
    # "deu_Latn" / "de-DE" -> "deu" / "de"; names like "German" match as a whole.
    base = re.match(r"^([a-z]{2,3})[_-][a-z0-9]{2,4}$", whole)
    for key in (whole, base.group(1) if base else whole.split("_")[0]):
        iso = key if key in data["families"] else data["aliases"].get(key)
        if iso:
            return data["families"][iso]
    return None


def family_of(lang_code: str, overrides: Optional[dict] = None) -> str:
    """Linguistic family for the frontend's "color by language family" mode.
    `overrides` (a dataset's own lang_families map) win; otherwise the code is
    looked up in the Glottolog data: ISO 639-3 ("deu", "deu_Latn"), ISO 639-1
    ("de") or an English name ("German"). Anything unknown shares one
    "Other" bucket rather than going uncolored."""
    if overrides and lang_code in overrides:
        return overrides[lang_code]
    return _lookup_family(lang_code) or "Other"


def point_family(point: dict, dataset: dict) -> str:
    """A point's own `family` beats the dataset's lang_families, which beat Glottolog."""
    return point.get("family") or family_of(point["lang"], dataset.get("lang_families"))


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev tool, no auth/sensitive deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASET: dict = {"data": None, "generation": 0}
_REDUCER_CACHE: dict = {
    "generation": -1, "metric": None, "n_neighbors": None, "min_dist": None, "n_components": None, "reducer": None,
}
# Set by POST /dataset/flores/cancel to stop the running FLORES embedding job
# at its next sentence; every new load replaces (and first sets) the old one.
_FLORES_CANCEL: dict = {"event": threading.Event()}
UMAP_N_NEIGHBORS_DEFAULT = 15
UMAP_MIN_DIST_DEFAULT = 0.1
TSNE_PERPLEXITY_DEFAULT = 30
TSNE_LEARNING_RATE_DEFAULT = "auto"


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
        if "family" in p and (not isinstance(p["family"], str) or not p["family"].strip()):
            raise ValueError(f'points[{i}].family must be a non-empty string')
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

    lang_families = raw.get("lang_families")
    if lang_families is not None and (
        not isinstance(lang_families, dict)
        or not all(isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in lang_families.items())
    ):
        raise ValueError('"lang_families" must be an object mapping language codes to family names')

    return {
        "model": raw.get("model") or "unknown model",
        "layer": raw.get("layer") or "unknown",
        "pooling": raw.get("pooling") or "mean",
        "dim": dim,
        "lang_codes": raw.get("lang_codes") or sorted({p["lang"] for p in points}),
        "lang_families": lang_families or {},
        "points": points,
    }


def _flores_config(raw: dict) -> dict:
    """Validates a POST /dataset/flores body into the pipeline's config dict."""
    codes = raw.get("lang_codes")
    if not isinstance(codes, list) or not codes or not all(isinstance(c, str) for c in codes):
        raise ValueError("lang_codes must be a non-empty list of FLORES-200 codes like eng_Latn")
    codes = list(dict.fromkeys(c.strip() for c in codes))
    bad = [c for c in codes if not FLORES_CODE_RE.match(c)]
    if bad:
        raise ValueError(f"Not FLORES-200 language codes (expected e.g. eng_Latn): {', '.join(bad[:5])}")
    if len(codes) > MAX_FLORES_LANGUAGES:
        raise ValueError(f"At most {MAX_FLORES_LANGUAGES} languages")

    n = raw.get("n_sentences")
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= FLORES_DEVTEST_SIZE:
        raise ValueError(f"n_sentences must be an integer between 1 and {FLORES_DEVTEST_SIZE}")

    model = raw.get("model", DEFAULT_MODEL)
    if not isinstance(model, str) or not model.strip() or len(model) > 200:
        raise ValueError("model must be a non-empty Hugging Face model name")
    pooling = raw.get("pooling", DEFAULT_POOLING)
    if pooling not in POOLINGS:
        raise ValueError(f"pooling must be one of {', '.join(POOLINGS)}")
    return {"lang_codes": codes, "n_sentences": n, "model": model.strip(), "pooling": pooling}


def _language_cache_path(code: str, model: str, pooling: str) -> Path:
    key = json.dumps([model, pooling, code])
    return DATA_DIR / f"flores_lang_{hashlib.sha1(key.encode()).hexdigest()[:12]}.json"


def _read_language_cache(code: str, config: dict):
    """Cached points for one language (first n_sentences of them), or None.
    FLORES order is fixed, so a cache built for N sentences also serves any
    smaller request."""
    path = _language_cache_path(code, config["model"], config["pooling"])
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text())
    except ValueError:
        return None
    points = cached.get("points") or []
    n = config["n_sentences"]
    if len(points) < n or points[0].get("tokenVectors") is None:
        return None
    return points[:n]


def _load_or_build_dataset(config: dict, builder: Callable) -> dict:
    """Embeds only the languages missing from the per-language cache.
    builder(lang_codes, n_sentences, model_name, pooling) -> dataset dict."""
    by_lang = {}
    missing = []
    for code in config["lang_codes"]:
        points = _read_language_cache(code, config)
        if points is None:
            missing.append(code)
        else:
            by_lang[code] = points

    if missing:
        built = builder(missing, config["n_sentences"], config["model"], config["pooling"])
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for code in missing:
            points = [p for p in built["points"] if p["lang"] == code]
            by_lang[code] = points
            _language_cache_path(code, config["model"], config["pooling"]).write_text(
                json.dumps({"model": config["model"], "pooling": config["pooling"], "lang": code, "points": points})
            )

    points = [p for code in config["lang_codes"] for p in by_lang[code]]
    return {
        "model": config["model"], "layer": "last", "pooling": config["pooling"],
        "lang_codes": config["lang_codes"], "points": points,
    }


@app.post("/dataset/flores")
def load_flores_dataset(raw: dict) -> dict:
    """Embeds the requested FLORES-200 languages/sentences (or reads the
    per-config cache under data/) and makes that the current dataset. Runs
    in the threadpool since a first build can take minutes. On failure the
    current dataset is left untouched."""
    try:
        config = _flores_config(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    print(f"Preparing FLORES-200 dataset: {config}")
    _FLORES_CANCEL["event"].set()  # a newer load supersedes one still running
    cancel = _FLORES_CANCEL["event"] = threading.Event()

    def builder(*args):
        return embedding_pipeline.build_dataset(*args, should_cancel=cancel.is_set)

    try:
        built = _load_or_build_dataset(config, builder)
        if cancel.is_set():
            raise embedding_pipeline.EmbeddingCancelled()
        normalized = validate_and_normalize(built)
    except embedding_pipeline.EmbeddingCancelled as e:
        raise HTTPException(status_code=409, detail="FLORES-200 load cancelled.") from e
    except embedding_pipeline.UnknownLanguageError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logging.getLogger("uvicorn.error").exception("FLORES-200 load failed")
        raise HTTPException(status_code=500, detail=f"Could not load FLORES-200: {type(e).__name__}: {e}") from e
    DATASET["data"] = normalized
    DATASET["generation"] += 1
    return {"n_points": len(normalized["points"]), "lang_codes": normalized["lang_codes"], "dim": normalized["dim"]}


@app.post("/dataset/flores/cancel")
def cancel_flores_load() -> dict:
    """Asks the running FLORES embedding job (if any) to stop; the current
    dataset is left untouched. Harmless when nothing is running."""
    _FLORES_CANCEL["event"].set()
    return {"cancelled": True}


@app.get("/dataset/info")
def dataset_info() -> dict:
    """Never 404s, so the frontend can ask whether anything is loaded before
    requesting points/projections."""
    d = DATASET["data"]
    if d is None:
        return {"loaded": False}
    return {"loaded": True, "n_points": len(d["points"]), "lang_codes": d["lang_codes"], "dim": d["dim"]}


def _current_dataset() -> dict:
    if DATASET["data"] is None:
        raise HTTPException(status_code=404, detail="No dataset loaded yet — load a FLORES-200 subset or upload an embeddings JSON")
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


@app.post("/dataset/upload")
async def upload_dataset(request: Request, format: str = "json") -> dict:
    """Raw file text in the request body; `format` is json, jsonl or csv
    (see dataset_formats.py). Leaves the current dataset untouched on error."""
    try:
        text = (await request.body()).decode("utf-8-sig")
        normalized = validate_and_normalize(parse_dataset_text(text, format))
    except UnicodeDecodeError as e:
        raise HTTPException(status_code=400, detail="The file isn't valid UTF-8 text") from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    DATASET["data"] = normalized
    DATASET["generation"] += 1
    return {"n_points": len(normalized["points"]), "lang_codes": normalized["lang_codes"], "dim": normalized["dim"]}


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
                "family": point_family(p, d),
                "hasTokens": p.get("tokenVectors") is not None,
                "tokenTexts": [t["token"] for t in p["tokenVectors"]] if p.get("tokenVectors") is not None else None,
            }
            for p in d["points"]
        ],
    }


def _fit_umap(vectors: np.ndarray, metric: str, n_neighbors: int, min_dist: float, n_components: int = 2) -> umap.UMAP:
    n = len(vectors)
    n_neighbors = max(1, min(n_neighbors, n - 1))
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42, metric=metric)
    reducer.fit(vectors)
    return reducer


def _get_umap_reducer(d: dict, metric: str, n_neighbors: int, min_dist: float, n_components: int = 2) -> umap.UMAP:
    """Fits once per (dataset "generation", metric, n_neighbors, min_dist, n_components) and
    reuses the same reducer after that — both so repeated /project?method=umap
    calls with the same params skip refitting, and so /project/tokens/{id} can
    .transform() token vectors into the exact same space the sentence points
    were fit into for that combination (callers must pass the same metric/
    n_neighbors/min_dist the main projection is currently using)."""
    cache_key = (DATASET["generation"], metric, n_neighbors, min_dist, n_components)
    current_key = (
        _REDUCER_CACHE["generation"], _REDUCER_CACHE["metric"], _REDUCER_CACHE["n_neighbors"],
        _REDUCER_CACHE["min_dist"], _REDUCER_CACHE["n_components"],
    )
    if cache_key != current_key:
        vectors = np.array([p["vector"] for p in d["points"]], dtype=np.float32)
        _REDUCER_CACHE["reducer"] = _fit_umap(vectors, metric, n_neighbors, min_dist, n_components)
        _REDUCER_CACHE["generation"] = DATASET["generation"]
        _REDUCER_CACHE["metric"] = metric
        _REDUCER_CACHE["n_neighbors"] = n_neighbors
        _REDUCER_CACHE["min_dist"] = min_dist
        _REDUCER_CACHE["n_components"] = n_components
    return _REDUCER_CACHE["reducer"]


def _coords_dict(row) -> dict:
    """x, y and (for a 3D projection) z of one projected row."""
    return {axis: float(value) for axis, value in zip("xyz", row)}


@app.get("/project")
def project(
    method: Literal["umap", "tsne"] = "umap",
    metric: Metric = "euclidean",
    n_neighbors: Optional[int] = Query(default=None, ge=1, le=200),
    min_dist: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    perplexity: Optional[float] = Query(default=None, gt=0, le=500),
    learning_rate: Optional[float] = Query(default=None, gt=0),
    n_components: int = Query(default=2, ge=2, le=3),
) -> dict:
    d = _current_dataset()
    points = d["points"]
    n = len(points)
    if n < 2:
        raise HTTPException(status_code=400, detail="need at least 2 points to project")

    if method == "umap":
        nn = min(n_neighbors or UMAP_N_NEIGHBORS_DEFAULT, n - 1)
        md = UMAP_MIN_DIST_DEFAULT if min_dist is None else min_dist
        reducer = _get_umap_reducer(d, metric, nn, md, n_components)
        coords = reducer.embedding_
        params = {"n_neighbors": nn, "min_dist": md}
    else:
        vectors = np.array([p["vector"] for p in points], dtype=np.float32)
        perp = max(1, min(perplexity or TSNE_PERPLEXITY_DEFAULT, n - 1))
        lr = TSNE_LEARNING_RATE_DEFAULT if learning_rate is None else learning_rate
        reducer = TSNE(n_components=n_components, perplexity=perp, learning_rate=lr, random_state=42, init="pca", metric=metric)
        coords = reducer.fit_transform(vectors)
        params = {"perplexity": perp, "learning_rate": lr}

    return {
        "method": method,
        "params": params,
        "n_components": n_components,
        "points": [{"id": p["id"], **_coords_dict(coords[i])} for i, p in enumerate(points)],
    }


@app.get("/project/tokens/{point_id}")
def project_tokens(
    point_id: str,
    metric: Metric = "euclidean",
    n_neighbors: Optional[int] = Query(default=None, ge=1, le=200),
    min_dist: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    n_components: int = Query(default=2, ge=2, le=3),
) -> dict:
    d = _current_dataset()
    point = next((p for p in d["points"] if str(p["id"]) == point_id), None)
    if point is None:
        raise HTTPException(status_code=404, detail=f"no point with id {point_id!r}")

    token_vectors = point.get("tokenVectors")
    if not token_vectors:
        raise HTTPException(status_code=400, detail=f"point {point_id!r} has no token vectors")
    if len(token_vectors) < 2:
        raise HTTPException(status_code=400, detail="need at least 2 tokens to project")

    nn = min(n_neighbors or UMAP_N_NEIGHBORS_DEFAULT, len(d["points"]) - 1)
    md = UMAP_MIN_DIST_DEFAULT if min_dist is None else min_dist
    reducer = _get_umap_reducer(d, metric, nn, md, n_components)
    vectors = np.array([t["vector"] for t in token_vectors], dtype=np.float32)
    coords = reducer.transform(vectors)

    return {
        "id": point["id"],
        "tokens": [
            {"token": t["token"], **_coords_dict(coords[i])}
            for i, t in enumerate(token_vectors)
        ],
    }


class StatsRequest(BaseModel):
    ids: List[Union[str, int]]


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _mean_pairwise_cosine_sums(unit: np.ndarray) -> tuple[float, int]:
    """(sum of cosine over all unordered pairs, number of such pairs) for unit
    rows, via |sum u|^2 = n + 2 * sum_{i<j} u_i . u_j — O(n*d), no n x n matrix."""
    n = len(unit)
    total = unit.sum(axis=0)
    return (float(total @ total) - n) / 2, n * (n - 1) // 2


def _retrieval_accuracy(sim: np.ndarray) -> Optional[float]:
    """MEXA-style score for a square similarity matrix whose diagonal holds the
    true translation pairs: the fraction of diagonal entries that strictly beat
    every other entry in their row and in their column."""
    n = len(sim)
    if n < 2:
        return None
    diag = np.diag(sim)
    off = sim.copy()
    np.fill_diagonal(off, -np.inf)
    hits = (diag > off.max(axis=1)) & (diag > off.max(axis=0))
    return float(hits.mean())


def _mean_or_none(values: list[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


@app.post("/stats")
def stats(req: StatsRequest) -> dict:
    """Aggregate statistics over the selected points, always cosine in the full
    embedding space (not the 2D projection) — the usual convention for alignment.

    `overall` splits the mean pairwise cosine into same-language and
    cross-language pairs. `language_pairs` compares, per language pair, the
    mean cosine of true translations (same pairId) against all other
    cross-language pairs, plus retrieval accuracy over the translations."""
    if not req.ids:
        raise HTTPException(status_code=400, detail="select at least one point")
    d = _current_dataset()
    by_id = {str(p["id"]): p for p in d["points"]}
    missing = [i for i in req.ids if str(i) not in by_id]
    if missing:
        raise HTTPException(status_code=404, detail=f"no point with id {str(missing[0])!r}")

    selected = list({str(i): by_id[str(i)] for i in req.ids}.values())
    langs = sorted({p["lang"] for p in selected})
    unit_by_lang: dict[str, np.ndarray] = {}
    pair_ids_by_lang: dict[str, dict] = {}
    for lang in langs:
        members = [p for p in selected if p["lang"] == lang]
        unit_by_lang[lang] = _unit_rows(np.array([p["vector"] for p in members], dtype=np.float64))
        first_row: dict = {}
        for row, p in enumerate(members):
            if p.get("pairId") is not None:
                first_row.setdefault(p["pairId"], row)
        pair_ids_by_lang[lang] = first_row

    family_by_lang: dict = {}
    for p in selected:
        family_by_lang.setdefault(p["lang"], point_family(p, d))

    all_sum, all_pairs = _mean_pairwise_cosine_sums(np.vstack(list(unit_by_lang.values())))
    within_sum = within_pairs = 0
    for unit in unit_by_lang.values():
        s, n_pairs = _mean_pairwise_cosine_sums(unit)
        within_sum += s
        within_pairs += n_pairs
    cross_pairs = all_pairs - within_pairs

    language_pairs = []
    for ai, a in enumerate(langs):
        for b in langs[ai + 1:]:
            sim = unit_by_lang[a] @ unit_by_lang[b].T
            common = sorted(set(pair_ids_by_lang[a]) & set(pair_ids_by_lang[b]), key=str)
            rows = [pair_ids_by_lang[a][pid] for pid in common]
            cols = [pair_ids_by_lang[b][pid] for pid in common]
            parallel = sim[rows, cols]
            n_other = sim.size - len(common)
            language_pairs.append({
                "a": a, "b": b,
                "n_parallel": len(common),
                "parallel_cosine": float(parallel.mean()) if len(common) else None,
                "non_parallel_cosine": float((sim.sum() - parallel.sum()) / n_other) if n_other else None,
                "retrieval_accuracy": _retrieval_accuracy(sim[np.ix_(rows, cols)]) if len(common) else None,
            })

    return {
        "n_selected": len(selected),
        "n_languages": len(langs),
        "languages": [
            {"lang": lang, "family": family_by_lang[lang], "count": len(unit_by_lang[lang])} for lang in langs
        ],
        "overall": {
            "within_language": (within_sum / within_pairs) if within_pairs else None,
            "cross_language": ((all_sum - within_sum) / cross_pairs) if cross_pairs else None,
        },
        "summary": {
            "parallel_cosine": _mean_or_none([lp["parallel_cosine"] for lp in language_pairs if lp["parallel_cosine"] is not None]),
            "non_parallel_cosine": _mean_or_none([lp["non_parallel_cosine"] for lp in language_pairs if lp["non_parallel_cosine"] is not None]),
            "retrieval_accuracy": _mean_or_none([lp["retrieval_accuracy"] for lp in language_pairs if lp["retrieval_accuracy"] is not None]),
        },
        "language_pairs": language_pairs,
    }


# The frontend is served by this same app (one process, one port, no CORS
# needed). Only the static asset folders are mounted — never the project
# root, which holds source, caches under data/ and .git.
ROOT = Path(__file__).parent


@app.middleware("http")
async def no_cache_static(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache"  # revalidate, so edits show on a normal refresh
    return response


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(ROOT / "index.html")


for _folder in ("client-js", "css", "examples"):
    app.mount(f"/{_folder}", StaticFiles(directory=ROOT / _folder), name=_folder)
