"""Projecting embeddings to 2D or 3D with UMAP or t-SNE.

A UMAP fit is kept between requests, so repeated requests with the same
settings skip refitting and the token vectors of a fitted sentence can be
placed with .transform() into the exact space the sentence points were
fitted into.

It also survives edits of the dataset: when points are added to the dataset
that was fitted, they are placed into the existing layout (overlay.py)
instead of refitting, so the points already on screen don't move. Removed
points leave the rest where it is. Only a different dataset, different
settings, or an explicit `refit` fits again."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

import numpy as np
from sklearn.manifold import TSNE

from backend.dataset import DatasetStore
from backend.overlay import place_new_points
from backend.umap_import import umap

UMAP_N_NEIGHBORS_DEFAULT = 15
UMAP_MIN_DIST_DEFAULT = 0.1
TSNE_PERPLEXITY_DEFAULT = 30
TSNE_LEARNING_RATE_DEFAULT = "auto"


def _coords_dict(row) -> dict:
    """x, y and (for a 3D projection) z of one projected row."""
    return {axis: float(value) for axis, value in zip("xyz", row)}


def _vectors(items: list) -> np.ndarray:
    return np.array([item["vector"] for item in items], dtype=np.float32)


@dataclass
class _UmapFit:
    """One UMAP fit and the layout it produced. `positions` also holds the
    points placed into the layout after the fit, and points removed from the
    dataset since — kept so they return to the same spot if they come back."""
    id: int
    reducer: umap.UMAP
    generation: int  # of the dataset it was fitted on
    metric: str
    n_components: int
    min_dist: float
    requested_neighbors: int
    n_neighbors: int  # what the fit used: the request, clamped to the dataset size
    positions: Dict[Any, np.ndarray]
    fitted: Set[Any]  # ids the fit itself was run on; the other positions were placed into it

    def answers(self, generation: int, metric: str, n_components: int, min_dist: float, n_neighbors: int) -> bool:
        """Whether a request would get this layout. n_neighbors may be what was
        asked for or what that was clamped to (the frontend echoes the latter back)."""
        return (
            (self.generation, self.metric, self.n_components, self.min_dist) == (generation, metric, n_components, min_dist)
            and n_neighbors in (self.requested_neighbors, self.n_neighbors)
        )


class Projector:
    """Projects the store's current dataset. Raises ValueError for a request
    that can't be satisfied and LookupError for an unknown point id."""

    def __init__(self, store: DatasetStore):
        self._store = store
        self._lock = threading.Lock()
        self._fit: Optional[_UmapFit] = None
        self._fits_made = 0

    def project(
        self, method: str, metric: str, n_components: int = 2,
        n_neighbors: Optional[int] = None, min_dist: Optional[float] = None,
        perplexity: Optional[float] = None, learning_rate: Optional[float] = None,
        refit: bool = False,
    ) -> dict:
        """{"method", "params" (the values actually used, after clamping to the
        dataset size), "n_components", "fit_id", "n_overlaid", "points":
        [{"id", "x", "y"[, "z"][, "overlaid"]}]}.

        UMAP only: a point has "overlaid" when it was placed into the layout
        rather than part of the fit, n_overlaid counts them, and fit_id
        identifies the layout (it changes when everything was fitted again).
        `refit` fits everything again, folding overlaid points in; t-SNE
        always does, and has no fit_id."""
        snapshot = self._store.snapshot()
        points, generation = snapshot.data["points"], snapshot.generation
        if len(points) < 2:
            raise ValueError("need at least 2 points to project")

        if method == "umap":
            settings = self._umap_settings(points, generation, metric, n_components, n_neighbors, min_dist)
            with self._lock:
                fit = self._umap_fit(**settings, refit=refit)
                if not self._place_unseen(fit, points, metric):
                    fit = self._umap_fit(**settings, refit=True)
                rows = [self._row(p["id"], fit) for p in points]
            return {
                "method": method,
                "params": {"n_neighbors": fit.n_neighbors, "min_dist": fit.min_dist},
                "n_components": n_components,
                "fit_id": fit.id,
                "n_overlaid": sum(1 for row in rows if row.get("overlaid")),
                "points": rows,
            }

        n = len(points)
        perp = max(1, min(perplexity or TSNE_PERPLEXITY_DEFAULT, n - 1))
        lr = TSNE_LEARNING_RATE_DEFAULT if learning_rate is None else learning_rate
        tsne = TSNE(n_components=n_components, perplexity=perp, learning_rate=lr, random_state=42, init="pca", metric=metric)
        coords = tsne.fit_transform(_vectors(points))
        return {
            "method": method,
            "params": {"perplexity": perp, "learning_rate": lr},
            "n_components": n_components,
            "fit_id": None,
            "n_overlaid": 0,
            "points": [{"id": p["id"], **_coords_dict(coords[i])} for i, p in enumerate(points)],
        }

    def project_tokens(
        self, point_id: str, metric: str, n_components: int = 2,
        n_neighbors: Optional[int] = None, min_dist: Optional[float] = None,
    ) -> dict:
        """One point's tokens placed into the UMAP space of the main
        projection; metric, n_neighbors, min_dist and n_components must match
        what that projection used. A sentence that was overlaid has its tokens
        overlaid too, so they land next to it."""
        snapshot = self._store.snapshot()
        point = next((p for p in snapshot.data["points"] if str(p["id"]) == point_id), None)
        if point is None:
            raise LookupError(f"no point with id {point_id!r}")
        token_vectors = point.get("tokenVectors")
        if not token_vectors:
            raise ValueError(f"point {point_id!r} has no token vectors")
        if len(token_vectors) < 2:
            raise ValueError("need at least 2 tokens to project")

        points = snapshot.data["points"]
        settings = self._umap_settings(points, snapshot.generation, metric, n_components, n_neighbors, min_dist)
        with self._lock:
            fit = self._umap_fit(**settings, refit=False)
            if not self._place_unseen(fit, points, metric):
                fit = self._umap_fit(**settings, refit=True)
            if point["id"] in fit.fitted:
                coords = fit.reducer.transform(_vectors(token_vectors))
            else:
                # .transform() would put them among the old points, where it
                # can't tell a new language from the ones it was fitted on.
                coords = self._place_into(fit, points, _vectors(token_vectors), metric)
        return {
            "id": point["id"],
            "tokens": [{"token": t["token"], **_coords_dict(coords[i])} for i, t in enumerate(token_vectors)],
        }

    @staticmethod
    def _umap_settings(points, generation, metric, n_components, n_neighbors, min_dist) -> dict:
        return {
            "points": points, "generation": generation, "metric": metric, "n_components": n_components,
            "n_neighbors": n_neighbors or UMAP_N_NEIGHBORS_DEFAULT,
            "min_dist": UMAP_MIN_DIST_DEFAULT if min_dist is None else min_dist,
        }

    def _umap_fit(self, points, generation, metric, n_components, n_neighbors, min_dist, refit) -> _UmapFit:
        """The kept fit if it still answers this request, otherwise a new fit
        of `points`. Call with the lock held."""
        fit = self._fit
        if refit or fit is None or not fit.answers(generation, metric, n_components, min_dist, n_neighbors):
            fit = self._fit = self._new_fit(points, generation, metric, n_components, n_neighbors, min_dist)
        return fit

    def _new_fit(self, points, generation, metric, n_components, n_neighbors, min_dist) -> _UmapFit:
        used_neighbors = min(n_neighbors, len(points) - 1)
        reducer = umap.UMAP(
            n_components=n_components, n_neighbors=used_neighbors, min_dist=min_dist,
            random_state=42, metric=metric,
        ).fit(_vectors(points))
        self._fits_made += 1
        return _UmapFit(
            id=self._fits_made, reducer=reducer, generation=generation, metric=metric,
            n_components=n_components, min_dist=min_dist, requested_neighbors=n_neighbors,
            n_neighbors=used_neighbors,
            positions={p["id"]: row for p, row in zip(points, reducer.embedding_)},
            fitted={p["id"] for p in points},
        )

    @staticmethod
    def _place_unseen(fit: _UmapFit, points: list, metric: str) -> bool:
        """Places the points the fit has no position for into its layout,
        anchored on the ones it has. False if too few are left to anchor on."""
        unseen = [p for p in points if p["id"] not in fit.positions]
        if not unseen:
            return True
        if sum(1 for p in points if p["id"] in fit.positions) < 2:
            return False
        placed = Projector._place_into(fit, points, _vectors(unseen), metric)
        for p, position in zip(unseen, placed):
            fit.positions[p["id"]] = position
        return True

    @staticmethod
    def _place_into(fit: _UmapFit, points: list, vectors: np.ndarray, metric: str) -> np.ndarray:
        """Positions for `vectors` in the fit's layout, anchored on those of
        `points` that have one."""
        anchors = [p for p in points if p["id"] in fit.positions]
        return place_new_points(
            _vectors(anchors), np.array([fit.positions[p["id"]] for p in anchors]), vectors,
            metric=metric, n_neighbors=fit.n_neighbors, min_dist=fit.min_dist,
        )

    @staticmethod
    def _row(point_id, fit: _UmapFit) -> dict:
        row = {"id": point_id, **_coords_dict(fit.positions[point_id])}
        if point_id not in fit.fitted:
            row["overlaid"] = True
        return row
