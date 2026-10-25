import logging

import numpy as np
import pytest

from backend import overlay

DIM = 10
SEPARATION = 8.0


def _blob(rng, axis, n, spread=0.3):
    centre = np.zeros(DIM)
    centre[axis] = SEPARATION
    return centre + rng.normal(0, spread, (n, DIM))


@pytest.fixture
def world():
    """Two anchor clusters (A, B) with a 2D layout, plus a generator for new points."""
    rng = np.random.default_rng(0)
    vectors = np.vstack([_blob(rng, 0, 60), _blob(rng, 1, 60)]).astype(np.float32)
    positions = np.vstack([
        np.array([0.0, 0.0]) + rng.normal(0, 0.3, (60, 2)),
        np.array([10.0, 0.0]) + rng.normal(0, 0.3, (60, 2)),
    ]).astype(np.float32)
    return rng, vectors, positions


def _place(vectors, positions, new, **kwargs):
    params = {"metric": "euclidean", "n_neighbors": 15, "min_dist": 0.1, **kwargs}
    return overlay.place_new_points(vectors, positions, new, **params)


def _spread(points):
    return float(np.linalg.norm(points - points.mean(axis=0), axis=1).mean())


def test_new_points_from_an_existing_cluster_land_in_that_cluster(world):
    rng, vectors, positions = world
    placed = _place(vectors, positions, _blob(rng, 1, 20).astype(np.float32))
    assert np.linalg.norm(placed.mean(axis=0) - [10.0, 0.0]) < 1.5


def test_a_new_cluster_stays_together_instead_of_smearing_over_the_old_ones(world):
    rng, vectors, positions = world
    new = _blob(rng, 2, 40).astype(np.float32)  # a "language" whose nearest neighbours are mostly itself
    refined = _place(vectors, positions, new)
    only_initial = _place(vectors, positions, new, n_epochs=0)
    assert _spread(refined) < 1.5
    assert _spread(only_initial) > 2 * _spread(refined)


def test_the_optimiser_never_moves_the_anchors(world, monkeypatch):
    rng, vectors, positions = world
    real = overlay.optimize_layout_euclidean
    seen = {}

    def spy(embedding, *args, **kwargs):
        anchors_before = embedding[: len(positions)].copy()
        real(embedding, *args, **kwargs)
        seen["anchors_moved"] = not np.array_equal(anchors_before, embedding[: len(positions)])

    monkeypatch.setattr(overlay, "optimize_layout_euclidean", spy)
    _place(vectors, positions, _blob(rng, 2, 10).astype(np.float32))
    assert seen["anchors_moved"] is False


def test_inputs_are_left_untouched(world):
    rng, vectors, positions = world
    vectors_before, positions_before = vectors.copy(), positions.copy()
    _place(vectors, positions, _blob(rng, 2, 10).astype(np.float32))
    assert np.array_equal(vectors, vectors_before) and np.array_equal(positions, positions_before)


def test_placement_is_deterministic(world):
    rng, vectors, positions = world
    new = _blob(rng, 2, 10).astype(np.float32)
    assert np.array_equal(_place(vectors, positions, new), _place(vectors, positions, new))


def test_three_dimensional_layouts(world):
    rng, vectors, positions = world
    positions3 = np.hstack([positions, np.zeros((len(positions), 1), dtype=np.float32)])
    placed = _place(vectors, positions3, _blob(rng, 0, 5).astype(np.float32))
    assert placed.shape == (5, 3) and np.isfinite(placed).all()


@pytest.mark.parametrize("metric", ["euclidean", "cosine", "manhattan"])
def test_every_supported_metric_places_points(world, metric):
    rng, vectors, positions = world
    placed = _place(vectors, positions, _blob(rng, 1, 5).astype(np.float32), metric=metric)
    assert placed.shape == (5, 2) and np.isfinite(placed).all()


def test_no_new_points_gives_an_empty_result(world):
    _, vectors, positions = world
    assert _place(vectors, positions, np.zeros((0, DIM), dtype=np.float32)).shape == (0, 2)


def test_needs_at_least_two_anchors(world):
    rng, vectors, positions = world
    with pytest.raises(ValueError, match="at least 2"):
        _place(vectors[:1], positions[:1], _blob(rng, 0, 3).astype(np.float32))


def test_falls_back_to_the_initial_placement_if_the_optimiser_fails(world, monkeypatch, caplog):
    rng, vectors, positions = world
    new = _blob(rng, 2, 10).astype(np.float32)
    expected = _place(vectors, positions, new, n_epochs=0)

    def broken(*args, **kwargs):
        raise TypeError("signature changed")

    monkeypatch.setattr(overlay, "optimize_layout_euclidean", broken)
    with caplog.at_level(logging.WARNING):
        placed = _place(vectors, positions, new)
    assert np.array_equal(placed, expected)
    assert "refinement" in caplog.text
