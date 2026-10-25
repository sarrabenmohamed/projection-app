"""Projecting embeddings to 2D or 3D with UMAP or t-SNE.

UMAP fits are cached so that repeated requests with the same parameters
skip refitting, and so token vectors can be placed with .transform() into
the exact space the sentence points were fitted into."""

from __future__ import annotations

import importlib.util
import sys
import types
from typing import Optional

# umap-learn unconditionally imports its optional ParametricUMAP submodule,
# which unconditionally does `import tensorflow`. On machines where the
# installed tensorflow hangs on import (see embedding_pipeline.py for the
# same issue with transformers), this stub makes that import resolve
# instantly instead of hanging — umap/__init__.py already falls back to a
# dummy ParametricUMAP class when tensorflow isn't usable, we just need the
# import itself to not hang. It must run before `import umap`.
if "tensorflow" not in sys.modules:
    tf_stub = types.ModuleType("tensorflow")
    tf_stub.__spec__ = importlib.util.spec_from_loader("tensorflow", loader=None)  # torch's dynamic import needs a spec
    sys.modules["tensorflow"] = tf_stub

import numpy as np
import umap
from sklearn.manifold import TSNE

from backend.dataset import DatasetStore

UMAP_N_NEIGHBORS_DEFAULT = 15
UMAP_MIN_DIST_DEFAULT = 0.1
TSNE_PERPLEXITY_DEFAULT = 30
TSNE_LEARNING_RATE_DEFAULT = "auto"


def _coords_dict(row) -> dict:
    """x, y and (for a 3D projection) z of one projected row."""
    return {axis: float(value) for axis, value in zip("xyz", row)}


def _vectors(items: list) -> np.ndarray:
    return np.array([item["vector"] for item in items], dtype=np.float32)


class Projector:
    """Projects the store's current dataset. Raises ValueError for a request
    that can't be satisfied and LookupError for an unknown point id."""

    def __init__(self, store: DatasetStore):
        self._store = store
        self._cached_key = None
        self._cached_reducer = None

    def project(
        self, method: str, metric: str, n_components: int = 2,
        n_neighbors: Optional[int] = None, min_dist: Optional[float] = None,
        perplexity: Optional[float] = None, learning_rate: Optional[float] = None,
    ) -> dict:
        """{"method", "params" (the values actually used, after clamping to the
        dataset size), "n_components", "points": [{"id", "x", "y"[, "z"]}]}."""
        points = self._store.current()["points"]
        n = len(points)
        if n < 2:
            raise ValueError("need at least 2 points to project")

        if method == "umap":
            nn = min(n_neighbors or UMAP_N_NEIGHBORS_DEFAULT, n - 1)
            md = UMAP_MIN_DIST_DEFAULT if min_dist is None else min_dist
            coords = self._umap_reducer(metric, nn, md, n_components).embedding_
            params = {"n_neighbors": nn, "min_dist": md}
        else:
            perp = max(1, min(perplexity or TSNE_PERPLEXITY_DEFAULT, n - 1))
            lr = TSNE_LEARNING_RATE_DEFAULT if learning_rate is None else learning_rate
            tsne = TSNE(n_components=n_components, perplexity=perp, learning_rate=lr, random_state=42, init="pca", metric=metric)
            coords = tsne.fit_transform(_vectors(points))
            params = {"perplexity": perp, "learning_rate": lr}

        return {
            "method": method,
            "params": params,
            "n_components": n_components,
            "points": [{"id": p["id"], **_coords_dict(coords[i])} for i, p in enumerate(points)],
        }

    def project_tokens(
        self, point_id: str, metric: str, n_components: int = 2,
        n_neighbors: Optional[int] = None, min_dist: Optional[float] = None,
    ) -> dict:
        """One point's tokens placed into the UMAP space of the main
        projection; metric, n_neighbors, min_dist and n_components must match
        what that projection used."""
        dataset = self._store.current()
        point = next((p for p in dataset["points"] if str(p["id"]) == point_id), None)
        if point is None:
            raise LookupError(f"no point with id {point_id!r}")
        token_vectors = point.get("tokenVectors")
        if not token_vectors:
            raise ValueError(f"point {point_id!r} has no token vectors")
        if len(token_vectors) < 2:
            raise ValueError("need at least 2 tokens to project")

        nn = min(n_neighbors or UMAP_N_NEIGHBORS_DEFAULT, len(dataset["points"]) - 1)
        md = UMAP_MIN_DIST_DEFAULT if min_dist is None else min_dist
        coords = self._umap_reducer(metric, nn, md, n_components).transform(_vectors(token_vectors))
        return {
            "id": point["id"],
            "tokens": [{"token": t["token"], **_coords_dict(coords[i])} for i, t in enumerate(token_vectors)],
        }

    def _umap_reducer(self, metric: str, n_neighbors: int, min_dist: float, n_components: int) -> umap.UMAP:
        """Fits once per (dataset generation, metric, n_neighbors, min_dist,
        n_components) and reuses that fit until one of them changes."""
        key = (self._store.generation, metric, n_neighbors, min_dist, n_components)
        if key != self._cached_key:
            self._cached_reducer = umap.UMAP(
                n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist,
                random_state=42, metric=metric,
            ).fit(_vectors(self._store.current()["points"]))
            self._cached_key = key
        return self._cached_reducer
