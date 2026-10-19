"""
FastAPI backend for projection-app: holds one embeddings dataset in memory
and serves 2D UMAP/t-SNE projections of it on demand. Run alongside the
static frontend: uvicorn server:app --port 8500

Endpoints:
    POST /dataset                 load a dataset (JSON shape the frontend validates).
    POST /dataset/reset           reload the default FLORES-200 dataset, discarding whatever's loaded.
    GET  /points                  point metadata only, no vectors (includes each point's language family).
    GET  /project                 2D coords for all points (?method=umap|tsne&metric=&n_neighbors=&min_dist=&perplexity=&learning_rate=).
    GET  /project/tokens/{id}     2D coords for one point's tokens, in the same UMAP space (?metric=/n_neighbors=/min_dist= must match).
    GET  /neighbors/{id}          nearest neighbors by full-vector distance (?k=&metric=).
    POST /stats                   aggregate cosine/alignment statistics for a set of selected point ids.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import sys
import types
from collections import Counter
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
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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


def _glottolog_families() -> dict:
    return _glottolog_data()["families"]


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
_REDUCER_CACHE: dict = {"generation": -1, "metric": None, "n_neighbors": None, "min_dist": None, "reducer": None}
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


def _build_default_dataset() -> dict:
    config = _pipeline_config()
    print(f"Preparing default dataset: {config}")
    raw = _load_or_build_dataset(DEFAULT_DATA_PATH, config, embedding_pipeline.build_dataset)
    return validate_and_normalize(raw)


@app.on_event("startup")
def load_default_dataset() -> None:
    try:
        DATASET["data"] = _build_default_dataset()
    except Exception as e:
        print(f"Could not prepare default dataset ({e}) — POST /dataset to load one.")
        return
    DATASET["generation"] += 1
    print(f"Loaded default dataset: {len(DATASET['data']['points'])} points")


@app.post("/dataset/reset")
def reset_dataset() -> dict:
    """Reloads the default FLORES-200 dataset, discarding whatever is
    currently loaded (e.g. an uploaded file) — the same logic startup
    runs, callable on demand so the frontend's "Reset to default" button
    doesn't require restarting the server. Fails loudly (500) rather than
    startup's silent skip, since a user clicking this button expects
    either a reset or an explanation, not nothing."""
    try:
        DATASET["data"] = _build_default_dataset()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not reload the default dataset: {e}") from e
    DATASET["generation"] += 1
    d = DATASET["data"]
    return {"n_points": len(d["points"]), "lang_codes": d["lang_codes"], "dim": d["dim"]}


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
                "family": point_family(p, d),
                "hasTokens": p.get("tokenVectors") is not None,
                "tokenTexts": [t["token"] for t in p["tokenVectors"]] if p.get("tokenVectors") is not None else None,
            }
            for p in d["points"]
        ],
    }


def _fit_umap(vectors: np.ndarray, metric: str, n_neighbors: int, min_dist: float) -> umap.UMAP:
    n = len(vectors)
    n_neighbors = max(1, min(n_neighbors, n - 1))
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42, metric=metric)
    reducer.fit(vectors)
    return reducer


def _get_umap_reducer(d: dict, metric: str, n_neighbors: int, min_dist: float) -> umap.UMAP:
    """Fits once per (dataset "generation", metric, n_neighbors, min_dist) and
    reuses the same reducer after that — both so repeated /project?method=umap
    calls with the same params skip refitting, and so /project/tokens/{id} can
    .transform() token vectors into the exact same space the sentence points
    were fit into for that combination (callers must pass the same metric/
    n_neighbors/min_dist the main projection is currently using)."""
    cache_key = (DATASET["generation"], metric, n_neighbors, min_dist)
    current_key = (_REDUCER_CACHE["generation"], _REDUCER_CACHE["metric"], _REDUCER_CACHE["n_neighbors"], _REDUCER_CACHE["min_dist"])
    if cache_key != current_key:
        vectors = np.array([p["vector"] for p in d["points"]], dtype=np.float32)
        _REDUCER_CACHE["reducer"] = _fit_umap(vectors, metric, n_neighbors, min_dist)
        _REDUCER_CACHE["generation"] = DATASET["generation"]
        _REDUCER_CACHE["metric"] = metric
        _REDUCER_CACHE["n_neighbors"] = n_neighbors
        _REDUCER_CACHE["min_dist"] = min_dist
    return _REDUCER_CACHE["reducer"]


@app.get("/project")
def project(
    method: Literal["umap", "tsne"] = "umap",
    metric: Literal["cosine", "euclidean", "manhattan", "chebyshev", "correlation", "braycurtis", "canberra"] = "euclidean",
    n_neighbors: Optional[int] = Query(default=None, ge=1, le=200),
    min_dist: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    perplexity: Optional[float] = Query(default=None, gt=0, le=500),
    learning_rate: Optional[float] = Query(default=None, gt=0),
) -> dict:
    d = _current_dataset()
    points = d["points"]
    n = len(points)
    if n < 2:
        raise HTTPException(status_code=400, detail="need at least 2 points to project")

    if method == "umap":
        nn = min(n_neighbors or UMAP_N_NEIGHBORS_DEFAULT, n - 1)
        md = UMAP_MIN_DIST_DEFAULT if min_dist is None else min_dist
        reducer = _get_umap_reducer(d, metric, nn, md)
        coords = reducer.embedding_
        params = {"n_neighbors": nn, "min_dist": md}
    else:
        vectors = np.array([p["vector"] for p in points], dtype=np.float32)
        perp = max(1, min(perplexity or TSNE_PERPLEXITY_DEFAULT, n - 1))
        lr = TSNE_LEARNING_RATE_DEFAULT if learning_rate is None else learning_rate
        reducer = TSNE(n_components=2, perplexity=perp, learning_rate=lr, random_state=42, init="pca", metric=metric)
        coords = reducer.fit_transform(vectors)
        params = {"perplexity": perp, "learning_rate": lr}

    return {
        "method": method,
        "params": params,
        "points": [{"id": p["id"], "x": float(coords[i][0]), "y": float(coords[i][1])} for i, p in enumerate(points)],
    }


@app.get("/neighbors/{point_id}")
def neighbors(
    point_id: str,
    k: int = Query(default=5, ge=1, le=100),
    metric: Literal["cosine", "euclidean", "manhattan", "chebyshev", "correlation", "braycurtis", "canberra"] = "euclidean",
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
def project_tokens(
    point_id: str,
    metric: Literal["cosine", "euclidean", "manhattan", "chebyshev", "correlation", "braycurtis", "canberra"] = "euclidean",
    n_neighbors: Optional[int] = Query(default=None, ge=1, le=200),
    min_dist: Optional[float] = Query(default=None, ge=0.0, le=1.0),
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
    reducer = _get_umap_reducer(d, metric, nn, md)
    vectors = np.array([t["vector"] for t in token_vectors], dtype=np.float32)
    coords = reducer.transform(vectors)

    return {
        "id": point["id"],
        "tokens": [
            {"token": t["token"], "x": float(coords[i][0]), "y": float(coords[i][1])}
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
