"""
FastAPI backend for projection-app: holds one embeddings dataset in memory
and serves 2D/3D UMAP/t-SNE projections of it on demand. Run alongside the
static frontend: uvicorn server:app --port 8500

This module only maps HTTP to the backend/ package that does the work: dataset.py
(validation and the current dataset), flores.py (FLORES-200 loading),
dataset_formats.py (JSONL/CSV parsing), projection.py, alignment_stats.py
and languages.py.

Endpoints:
    POST /dataset                 load a dataset (JSON shape the frontend validates).
    POST /dataset/upload          load a JSON, JSONL or CSV file (?format=).
    POST /dataset/flores          embed (or load from cache) a FLORES-200 subset; POST /dataset/flores/cancel stops it.
    GET  /dataset/info            whether anything is loaded (never 404s).
    GET  /points                  point metadata only, no vectors (includes each point's language family).
    GET  /project                 2D (or 3D, ?n_components=3) coords for all points (?method=umap|tsne&metric=&n_neighbors=&min_dist=&perplexity=&learning_rate=).
    GET  /project/tokens/{id}     coords for one point's tokens, in the same UMAP space (?metric=/n_neighbors=/min_dist=/n_components= must match).
    POST /stats                   aggregate cosine/alignment statistics for a set of selected point ids.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Literal, Optional, Union

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import embedding_pipeline
from backend.alignment_stats import selection_stats
from backend.dataset import DatasetStore, NoDatasetError, validate_and_normalize
from backend.dataset_formats import parse_dataset_text
from backend.flores import FloresLoader
from backend.languages import point_family
from backend.projection import Projector

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
Metric = Literal["euclidean", "cosine", "manhattan"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev tool, no auth/sensitive deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

store = DatasetStore()
projector = Projector(store)
# Looks build_dataset up at call time, so tests can swap it out.
flores_loader = FloresLoader(DATA_DIR, lambda *args, **kwargs: embedding_pipeline.build_dataset(*args, **kwargs))


@app.exception_handler(NoDatasetError)
async def _no_dataset(request: Request, exc: NoDatasetError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


def _bad_request(e: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(e))


def _not_found(e: Exception) -> HTTPException:
    # str(LookupError(msg)) quotes the message; args[0] is the message itself.
    return HTTPException(status_code=404, detail=str(e.args[0]))


# ── Loading a dataset ────────────────────────────────────────────────────────

@app.post("/dataset")
async def set_dataset(raw: dict) -> dict:
    try:
        return store.replace(validate_and_normalize(raw))
    except ValueError as e:
        raise _bad_request(e) from e


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
        raise _bad_request(e) from e
    return store.replace(normalized)


@app.post("/dataset/flores")
def load_flores_dataset(raw: dict) -> dict:
    """Embeds the requested FLORES-200 languages/sentences (or reads the
    per-language cache under data/) and makes that the current dataset. Runs
    in the threadpool since a first build can take minutes. On failure the
    current dataset is left untouched."""
    try:
        normalized = flores_loader.load(raw)
    except embedding_pipeline.EmbeddingCancelled as e:
        raise HTTPException(status_code=409, detail="FLORES-200 load cancelled.") from e
    except ValueError as e:  # a bad request, or UnknownLanguageError
        raise _bad_request(e) from e
    except Exception as e:
        logging.getLogger("uvicorn.error").exception("FLORES-200 load failed")
        raise HTTPException(status_code=500, detail=f"Could not load FLORES-200: {type(e).__name__}: {e}") from e
    return store.replace(normalized)


@app.post("/dataset/flores/cancel")
def cancel_flores_load() -> dict:
    flores_loader.cancel()
    return {"cancelled": True}


@app.get("/dataset/info")
def dataset_info() -> dict:
    """Never 404s, so the frontend can ask whether anything is loaded before
    requesting points/projections."""
    if store.data is None:
        return {"loaded": False}
    return {"loaded": True, **store.summary()}


# ── Reading the dataset ──────────────────────────────────────────────────────

@app.get("/points")
def get_points() -> dict:
    d = store.current()
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
    try:
        return projector.project(method, metric, n_components, n_neighbors, min_dist, perplexity, learning_rate)
    except ValueError as e:
        raise _bad_request(e) from e


@app.get("/project/tokens/{point_id}")
def project_tokens(
    point_id: str,
    metric: Metric = "euclidean",
    n_neighbors: Optional[int] = Query(default=None, ge=1, le=200),
    min_dist: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    n_components: int = Query(default=2, ge=2, le=3),
) -> dict:
    try:
        return projector.project_tokens(point_id, metric, n_components, n_neighbors, min_dist)
    except NoDatasetError:
        raise
    except LookupError as e:
        raise _not_found(e) from e
    except ValueError as e:
        raise _bad_request(e) from e


class StatsRequest(BaseModel):
    ids: List[Union[str, int]]


@app.post("/stats")
def stats(req: StatsRequest) -> dict:
    try:
        return selection_stats(store.current(), req.ids)
    except NoDatasetError:
        raise
    except LookupError as e:
        raise _not_found(e) from e
    except ValueError as e:
        raise _bad_request(e) from e


# ── The frontend ─────────────────────────────────────────────────────────────
# Served by this same app (one process, one port, no CORS needed). Only the
# static asset folders are mounted — never the project root, which holds
# source, caches under data/ and .git.

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
