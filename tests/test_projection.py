import numpy as np
import pytest

from backend.dataset import DatasetStore, validate_and_normalize
from backend.projection import Projector
from backend.umap_import import umap

DIM = 12
LANGS = ["eng_Latn", "deu_Latn", "fra_Latn", "spa_Latn"]
FLORES = {"kind": "flores", "n_sentences": 30}


def _vector(lang, i):
    centre = np.zeros(DIM)
    centre[LANGS.index(lang)] = 6.0
    return (centre + np.random.default_rng([LANGS.index(lang), i]).normal(0, 0.5, DIM)).tolist()


def _dataset(langs, n_sentences=30, tokens=False):
    points = []
    for lang in langs:
        for i in range(n_sentences):
            point = {"id": f"{lang}-{i}", "lang": lang, "pairId": i, "text": f"{lang} {i}", "vector": _vector(lang, i)}
            if tokens:
                point["tokenVectors"] = [{"token": f"t{j}", "vector": _vector(lang, i + 100 * j)} for j in range(3)]
            points.append(point)
    return validate_and_normalize({"model": "m", "lang_codes": list(langs), "points": points})


@pytest.fixture
def fits(monkeypatch):
    """The number of rows of every UMAP fit, so tests can tell a refit from a reuse."""
    rows = []
    real_fit = umap.UMAP.fit

    def counting_fit(self, X, *args, **kwargs):
        rows.append(len(X))
        return real_fit(self, X, *args, **kwargs)

    monkeypatch.setattr(umap.UMAP, "fit", counting_fit)
    return rows


@pytest.fixture
def store():
    store = DatasetStore()
    store.replace(_dataset(LANGS[:2]), FLORES)
    return store


@pytest.fixture
def projector(store):
    return Projector(store)


def _edit(store, langs, n_sentences=30, **kwargs):
    store.revise(_dataset(langs, n_sentences, **kwargs), FLORES, based_on_generation=store.generation)


def _by_id(result):
    return {p["id"]: p for p in result["points"]}


def _xy(point):
    return np.array([point["x"], point["y"]])


def test_the_first_projection_fits_every_point(projector, fits):
    result = projector.project("umap", "euclidean")
    assert fits == [60]
    assert len(result["points"]) == 60 and result["n_overlaid"] == 0
    assert not any("overlaid" in p for p in result["points"])


def test_the_same_request_reuses_the_fit(projector, fits):
    first = projector.project("umap", "euclidean")
    again = projector.project("umap", "euclidean")
    assert fits == [60] and again == first


def test_added_points_are_placed_without_refitting_the_others(projector, store, fits):
    before = projector.project("umap", "euclidean")
    _edit(store, LANGS[:3])
    after = projector.project("umap", "euclidean")

    assert fits == [60]
    assert after["fit_id"] == before["fit_id"]
    assert after["n_overlaid"] == 30
    old = _by_id(before)
    new = _by_id(after)
    assert all(new[i]["x"] == old[i]["x"] and new[i]["y"] == old[i]["y"] for i in old)
    assert {p["id"] for p in after["points"] if p.get("overlaid")} == {f"fra_Latn-{i}" for i in range(30)}


def test_an_added_language_forms_its_own_group_next_to_the_old_points(projector, store):
    projector.project("umap", "euclidean")
    _edit(store, LANGS[:3])
    points = _by_id(projector.project("umap", "euclidean"))
    new = np.array([_xy(points[f"fra_Latn-{i}"]) for i in range(30)])
    old = np.array([_xy(points[f"{lang}-{i}"]) for lang in LANGS[:2] for i in range(30)])
    within = np.linalg.norm(new - new.mean(axis=0), axis=1).mean()
    to_old = np.linalg.norm(new.mean(axis=0) - old, axis=1).min()
    assert within < to_old


def test_removing_points_leaves_the_rest_where_they_are(projector, store, fits):
    before = _by_id(projector.project("umap", "euclidean"))
    _edit(store, LANGS[:1])
    after = _by_id(projector.project("umap", "euclidean"))
    assert fits == [60]
    assert set(after) == {f"eng_Latn-{i}" for i in range(30)}
    assert all(after[i]["x"] == before[i]["x"] and after[i]["y"] == before[i]["y"] for i in after)


def test_points_that_come_back_return_to_their_old_position(projector, store, fits):
    projector.project("umap", "euclidean")
    _edit(store, LANGS[:3])
    with_overlay = _by_id(projector.project("umap", "euclidean"))
    _edit(store, LANGS[:2])
    projector.project("umap", "euclidean")
    _edit(store, LANGS[:3])
    again = _by_id(projector.project("umap", "euclidean"))
    assert fits == [60]
    assert all(again[i] == with_overlay[i] for i in with_overlay)


def test_more_sentences_per_language_are_placed_too(projector, store, fits):
    before = _by_id(projector.project("umap", "euclidean"))
    _edit(store, LANGS[:2], n_sentences=40)
    after = projector.project("umap", "euclidean")
    assert fits == [60] and after["n_overlaid"] == 20 and len(after["points"]) == 80
    assert all(_by_id(after)[i]["x"] == before[i]["x"] for i in before)


def test_replacing_the_dataset_fits_again(projector, store, fits):
    first = projector.project("umap", "euclidean")
    store.replace(_dataset(LANGS[:3]), FLORES)
    second = projector.project("umap", "euclidean")
    assert fits == [60, 90]
    assert second["fit_id"] != first["fit_id"] and second["n_overlaid"] == 0


def test_changing_a_setting_fits_every_point_again_including_the_added_ones(projector, store, fits):
    projector.project("umap", "euclidean")
    _edit(store, LANGS[:3])
    projector.project("umap", "euclidean")
    refitted = projector.project("umap", "cosine")
    assert fits == [60, 90]
    assert refitted["n_overlaid"] == 0 and not any(p.get("overlaid") for p in refitted["points"])


def test_asking_to_refit_folds_the_added_points_into_a_new_fit(projector, store, fits):
    first = projector.project("umap", "euclidean")
    _edit(store, LANGS[:3])
    projector.project("umap", "euclidean")
    folded = projector.project("umap", "euclidean", refit=True)
    assert fits == [60, 90]
    assert folded["n_overlaid"] == 0 and folded["fit_id"] != first["fit_id"]


def test_neighbour_counts_clamped_to_a_small_dataset_still_match_the_fit(fits):
    store = DatasetStore()
    store.replace(_dataset(LANGS[:2], 4), {"kind": "flores", "n_sentences": 4})
    projector = Projector(store)
    first = projector.project("umap", "euclidean")
    assert first["params"]["n_neighbors"] == 7
    store.revise(_dataset(LANGS[:3], 4), {"kind": "flores", "n_sentences": 4}, based_on_generation=1)
    echoed = projector.project("umap", "euclidean", n_neighbors=first["params"]["n_neighbors"])
    assert fits == [8] and echoed["n_overlaid"] == 4


def test_with_too_few_points_left_to_anchor_on_everything_is_fitted_again(projector, store, fits):
    projector.project("umap", "euclidean")
    store.revise(_dataset(LANGS[2:4], 3), FLORES, based_on_generation=store.generation)
    result = projector.project("umap", "euclidean")
    assert fits == [60, 6] and result["n_overlaid"] == 0


def test_three_dimensional_layouts_place_new_points_too(projector, store, fits):
    projector.project("umap", "euclidean", n_components=3)
    _edit(store, LANGS[:3])
    result = projector.project("umap", "euclidean", n_components=3)
    assert fits == [60] and all("z" in p for p in result["points"]) and result["n_overlaid"] == 30


def test_tsne_always_recomputes_and_never_overlays(projector, store):
    projector.project("tsne", "euclidean")
    _edit(store, LANGS[:3])
    result = projector.project("tsne", "euclidean")
    assert len(result["points"]) == 90
    assert result["fit_id"] is None and result["n_overlaid"] == 0


def test_tokens_are_placed_into_the_same_fit_as_the_points(store, fits):
    store.replace(_dataset(LANGS[:2], tokens=True), FLORES)
    projector = Projector(store)
    projector.project("umap", "euclidean")
    tokens = projector.project_tokens("eng_Latn-0", "euclidean")
    assert fits == [60] and [t["token"] for t in tokens["tokens"]] == ["t0", "t1", "t2"]


def test_tokens_still_use_the_original_fit_after_points_were_added(store, fits):
    store.replace(_dataset(LANGS[:2], tokens=True), FLORES)
    projector = Projector(store)
    before = projector.project("umap", "euclidean")
    _edit(store, LANGS[:3], tokens=True)
    tokens = projector.project_tokens("fra_Latn-0", "euclidean")
    assert fits == [60] and len(tokens["tokens"]) == 3
    assert projector.project("umap", "euclidean")["fit_id"] == before["fit_id"]


def test_projecting_needs_two_points(store):
    store.replace(_dataset(LANGS[:1], 1), FLORES)
    with pytest.raises(ValueError, match="at least 2"):
        Projector(store).project("umap", "euclidean")


def _cluster_of(position, layout):
    """The language whose points' centroid in `layout` is nearest to `position`."""
    centroids = {}
    for point_id, point in layout.items():
        centroids.setdefault(point_id.split("-")[0], []).append(_xy(point))
    return min(centroids, key=lambda lang: np.linalg.norm(np.mean(centroids[lang], axis=0) - position))


def test_tokens_of_an_added_sentence_land_with_the_sentence_not_among_the_old_points(store):
    store.replace(_dataset(LANGS[:2], tokens=True), FLORES)
    projector = Projector(store)
    projector.project("umap", "euclidean")
    _edit(store, LANGS[:3], tokens=True)
    layout = _by_id(projector.project("umap", "euclidean"))
    tokens = projector.project_tokens("fra_Latn-0", "euclidean")["tokens"]
    centroid = np.mean([[t["x"], t["y"]] for t in tokens], axis=0)
    assert _cluster_of(centroid, layout) == "fra_Latn"
