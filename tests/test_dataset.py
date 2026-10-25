import pytest

from backend.dataset import DatasetChangedError, DatasetStore, NoDatasetError, validate_and_normalize


def _dataset(*ids):
    return validate_and_normalize({
        "model": "m",
        "points": [{"id": i, "lang": "eng_Latn", "vector": [0.0, 1.0]} for i in ids],
    })


def test_a_new_store_is_empty():
    with pytest.raises(NoDatasetError):
        DatasetStore().current()


def test_replacing_bumps_the_generation_and_defaults_to_an_upload():
    store = DatasetStore()
    first = store.replace(_dataset("a", "b"))
    store.replace(_dataset("c", "d"))
    assert first["source"] == {"kind": "upload"}
    assert store.generation == 2 and store.source == {"kind": "upload"}


def test_replacing_records_where_the_dataset_came_from():
    store = DatasetStore()
    summary = store.replace(_dataset("a", "b"), {"kind": "flores", "n_sentences": 2})
    assert summary == {"n_points": 2, "lang_codes": ["eng_Latn"], "dim": 2, "source": {"kind": "flores", "n_sentences": 2}}


def test_revising_keeps_the_generation_so_cached_layouts_stay_valid():
    store = DatasetStore()
    store.replace(_dataset("a", "b"), {"kind": "flores", "n_sentences": 2})
    summary = store.revise(_dataset("a", "b", "c"), {"kind": "flores", "n_sentences": 3}, based_on_generation=1)
    assert store.generation == 1
    assert summary["n_points"] == 3 and store.source == {"kind": "flores", "n_sentences": 3}


def test_revising_a_dataset_that_was_replaced_meanwhile_is_refused():
    store = DatasetStore()
    store.replace(_dataset("a", "b"))
    store.replace(_dataset("x", "y"))
    with pytest.raises(DatasetChangedError):
        store.revise(_dataset("a", "b", "c"), {"kind": "flores", "n_sentences": 3}, based_on_generation=1)
    assert [p["id"] for p in store.current()["points"]] == ["x", "y"]


def test_revising_with_nothing_loaded_raises():
    with pytest.raises(NoDatasetError):
        DatasetStore().revise(_dataset("a", "b"), {"kind": "upload"}, based_on_generation=0)


def test_snapshot_pairs_the_dataset_with_its_source_and_generation():
    store = DatasetStore()
    store.replace(_dataset("a", "b"), {"kind": "flores", "n_sentences": 2})
    snapshot = store.snapshot()
    assert snapshot.data is store.data and snapshot.generation == 1
    assert snapshot.source == {"kind": "flores", "n_sentences": 2}


def test_snapshot_with_nothing_loaded_raises():
    with pytest.raises(NoDatasetError):
        DatasetStore().snapshot()


def test_duplicate_point_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        _dataset("a", "b", "a")
