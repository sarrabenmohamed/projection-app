"""Placing new points into an existing, frozen 2D/3D layout.

UMAP.transform places every new point on its own, from its neighbours among
the *old* points only. A whole new language — whose nearest neighbours are
mostly its own sentences — therefore gets smeared over the old clusters
instead of forming a cluster of its own (on 7 FLORES languages the new
points kept 0.20 of their high-dimensional neighbourhood, against 0.45 for a
full refit). `place_new_points` instead runs UMAP's own layout optimiser
with the old points held fixed as anchors: each new point is attracted to
its neighbours among the anchors *and* the other new points, and repelled
from random points, as in a fit — but the anchors never move. Measured the
same way it recovers 0.40–0.45, in a fraction of a second."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.neighbors import NearestNeighbors

from backend.umap_import import (
    compute_membership_strengths,
    find_ab_params,
    make_epochs_per_sample,
    optimize_layout_euclidean,
    smooth_knn_dist,
)

DEFAULT_EPOCHS = 200
_INITIAL_ALPHA = 0.25  # what UMAP.transform uses (a quarter of the fit's learning rate)
_NEGATIVE_SAMPLE_RATE = 5.0
_INT32 = np.iinfo(np.int32)

logger = logging.getLogger(__name__)


def place_new_points(
    anchor_vectors, anchor_positions, new_vectors, *,
    metric: str, n_neighbors: int, min_dist: float, spread: float = 1.0,
    n_epochs: int = DEFAULT_EPOCHS, seed: int = 42,
) -> np.ndarray:
    """Layout positions (n_new x n_components) for `new_vectors`, given the
    high-dimensional `anchor_vectors` and their fixed `anchor_positions`.
    `n_epochs=0` skips the refinement and returns just the initial placement
    (the weighted mean of each point's nearest anchors)."""
    anchor_vectors = np.asarray(anchor_vectors, dtype=np.float32)
    anchor_positions = np.asarray(anchor_positions, dtype=np.float32)
    new_vectors = np.asarray(new_vectors, dtype=np.float32)
    n_anchors, n_components = anchor_positions.shape
    if n_anchors < 2:
        raise ValueError("need at least 2 anchor points to place new points")
    if len(new_vectors) == 0:
        return np.zeros((0, n_components), dtype=np.float32)

    initial = _initial_positions(anchor_vectors, anchor_positions, new_vectors, metric, n_neighbors)
    if n_epochs <= 0:
        return initial
    try:
        return _refine(anchor_vectors, anchor_positions, new_vectors, initial, metric, n_neighbors, min_dist, spread, n_epochs, seed)
    except Exception:  # the optimiser is umap-learn internals; an unexpected change must not break the app
        logger.warning("anchored refinement failed; using the initial placement", exc_info=True)
        return initial


def _initial_positions(anchor_vectors, anchor_positions, new_vectors, metric, n_neighbors) -> np.ndarray:
    """Each new point starts at the weighted mean position of its nearest
    anchors, weighted like UMAP's fuzzy memberships (what UMAP.transform does)."""
    k = min(n_neighbors, len(anchor_vectors))
    dists, inds = NearestNeighbors(n_neighbors=k, metric=metric).fit(anchor_vectors).kneighbors(new_vectors)
    dists = dists.astype(np.float32)
    sigmas, rhos = smooth_knn_dist(dists, float(k), local_connectivity=0.0)
    _, _, weights, _ = compute_membership_strengths(inds, dists, sigmas, rhos, bipartite=True)
    weights = weights.reshape(len(new_vectors), k)
    totals = weights.sum(axis=1, keepdims=True)
    weights = np.where(totals > 0, weights / np.where(totals > 0, totals, 1.0), 1.0 / k)
    return (weights[:, :, None] * anchor_positions[inds]).sum(axis=1).astype(np.float32)


def _refine(anchor_vectors, anchor_positions, new_vectors, initial, metric, n_neighbors, min_dist, spread, n_epochs, seed) -> np.ndarray:
    n_anchors = len(anchor_vectors)
    reference = np.vstack([anchor_vectors, new_vectors])
    k = min(n_neighbors, len(reference))  # counts the point itself, as in UMAP's fit
    dists, inds = NearestNeighbors(n_neighbors=k, metric=metric).fit(reference).kneighbors(new_vectors)
    dists = dists.astype(np.float32)
    sigmas, rhos = smooth_knn_dist(dists, float(k), local_connectivity=1.0)
    rows, cols, weights, _ = compute_membership_strengths(inds, dists, sigmas, rhos, bipartite=True)

    head = (n_anchors + rows).astype(np.int32)
    tail = cols.astype(np.int32)
    keep = (head != tail) & (weights > 0)  # a point is not its own neighbour
    head, tail, weights = head[keep], tail[keep], weights[keep]
    if len(weights) == 0:
        return initial
    weights = np.where(weights < weights.max() / n_epochs, 0.0, weights)  # drop negligible edges, as UMAP does
    keep = weights > 0
    head, tail, weights = head[keep], tail[keep], weights[keep]

    a, b = find_ab_params(spread, min_dist)
    embedding = np.ascontiguousarray(np.vstack([anchor_positions, initial]), dtype=np.float32)
    rng_state = np.random.RandomState(seed).randint(_INT32.min, _INT32.max, 3).astype(np.int64)
    # head and tail are the same array and move_other is off: only the new
    # points (the heads) move, the anchors are read but never written.
    optimize_layout_euclidean(
        embedding, embedding, head, tail, n_epochs, len(embedding),
        make_epochs_per_sample(weights, n_epochs), a, b, rng_state,
        1.0, _INITIAL_ALPHA, _NEGATIVE_SAMPLE_RATE, False, False, move_other=False,
    )
    return embedding[n_anchors:]
