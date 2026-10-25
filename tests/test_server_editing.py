"""Editing a loaded FLORES-200 dataset through the API: the new languages and
sentences are embedded, and projected into the layout that is already shown."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
from backend import embedding_pipeline, projection

CODES = ["eng_Latn", "deu_Latn", "fra_Latn", "spa_Latn"]


def _vector(code, i):
    return np.random.default_rng([CODES.index(code), i]).normal(CODES.index(code), 1.0, 6).tolist()


@pytest.fixture
def builds(tmp_path, monkeypatch):
    """Replaces the embedding pipeline; records (lang_codes, n_sentences, model, pooling, start) of every build."""
    calls = []

    def build(lang_codes, n_sentences, model_name, pooling, should_cancel=None, start=0):
        calls.append((list(lang_codes), n_sentences, model_name, pooling, start))
        return {
            "model": model_name, "layer": "last", "pooling": pooling, "lang_codes": lang_codes,
            "points": [
                {"id": f"{c}-{i}", "lang": c, "pairId": i, "text": f"{c} {i}", "vector": _vector(c, i),
                 "tokenVectors": [{"token": f"t{j}", "vector": _vector(c, i + 50 * (j + 1))} for j in range(3)]}
                for c in lang_codes for i in range(start, n_sentences)
            ],
        }

    monkeypatch.setattr(server.embedding_pipeline, "build_dataset", build)
    monkeypatch.setattr(server.flores_loader, "data_dir", tmp_path)
    return calls


@pytest.fixture
def client(builds):
    client = TestClient(server.app)
    assert client.post("/dataset/flores", json={"lang_codes": CODES[:2], "n_sentences": 10}).status_code == 200
    builds.clear()
    return client


@pytest.fixture
def fits(monkeypatch):
    rows = []
    real_fit = projection.umap.UMAP.fit

    def counting_fit(self, X, *args, **kwargs):
        rows.append(len(X))
        return real_fit(self, X, *args, **kwargs)

    monkeypatch.setattr(projection.umap.UMAP, "fit", counting_fit)
    return rows


def _edit(client, codes, n_sentences=10, **extra):
    return client.patch("/dataset/flores", json={"lang_codes": codes, "n_sentences": n_sentences, **extra})


def test_the_source_of_the_dataset_is_reported(client):
    assert client.get("/dataset/info").json()["source"] == {"kind": "flores", "n_sentences": 10}
    assert client.get("/points").json()["source"] == {"kind": "flores", "n_sentences": 10}

    client.post("/dataset/upload?format=csv", content="id,lang,v0,v1\na,en,1,2\nb,de,3,4\n")
    assert client.get("/dataset/info").json()["source"] == {"kind": "upload"}
    assert client.get("/points").json()["source"] == {"kind": "upload"}


def test_adding_a_language_embeds_only_that_language(client, builds):
    res = _edit(client, CODES[:3])
    assert res.status_code == 200
    assert res.json()["n_points"] == 30 and res.json()["lang_codes"] == CODES[:3]
    assert builds == [(["fra_Latn"], 10, "xlm-roberta-base", "mean", 0)]


def test_removing_a_language_embeds_nothing(client, builds):
    res = _edit(client, CODES[:1])
    assert res.status_code == 200 and res.json()["n_points"] == 10
    assert builds == []


def test_more_sentences_embed_only_the_new_ones(client, builds):
    res = _edit(client, CODES[:2], n_sentences=15)
    assert res.json()["n_points"] == 30 and res.json()["source"] == {"kind": "flores", "n_sentences": 15}
    assert builds == [(CODES[:2], 15, "xlm-roberta-base", "mean", 10)]


def test_fewer_sentences_are_served_from_the_cache(client, builds):
    res = _edit(client, CODES[:2], n_sentences=4)
    assert res.json()["n_points"] == 8 and builds == []


def test_the_model_and_pooling_stay_those_of_the_loaded_dataset(builds, tmp_path):
    client = TestClient(server.app)
    client.post("/dataset/flores", json={"lang_codes": CODES[:2], "n_sentences": 5, "model": "other-model", "pooling": "max"})
    builds.clear()
    assert _edit(client, CODES[:3], n_sentences=5).status_code == 200
    assert builds == [(["fra_Latn"], 5, "other-model", "max", 0)]


def test_asking_for_a_different_model_is_refused(client, builds):
    res = _edit(client, CODES[:3], model="bert-base-multilingual-cased")
    assert res.status_code == 400 and "model" in res.json()["detail"]
    assert _edit(client, CODES[:3], pooling="max").status_code == 400
    assert builds == []


def test_editing_with_nothing_loaded_is_a_404(client, monkeypatch):
    monkeypatch.setattr(server.store, "data", None)
    res = _edit(client, CODES[:3])
    assert res.status_code == 404 and "FLORES" in res.json()["detail"]


def test_an_uploaded_dataset_cannot_be_edited(client):
    client.post("/dataset/upload?format=csv", content="id,lang,v0,v1\na,en,1,2\nb,de,3,4\n")
    res = _edit(client, CODES[:3])
    assert res.status_code == 409 and "load a new dataset" in res.json()["detail"]


def test_a_bad_request_is_a_400_and_keeps_the_dataset(client):
    before = server.store.data
    assert _edit(client, ["nope"]).status_code == 400
    assert _edit(client, CODES[:2], n_sentences=0).status_code == 400
    assert server.store.data is before


def test_failures_while_embedding_keep_the_dataset_and_are_reported(client, monkeypatch):
    before = server.store.data

    def unknown(*args, **kwargs):
        raise embedding_pipeline.UnknownLanguageError('"fra_Latn" is not a FLORES-200 language')

    def boom(*args, **kwargs):
        raise RuntimeError("no network")

    def cancelled(*args, **kwargs):
        raise embedding_pipeline.EmbeddingCancelled()

    for failure, status, text in ((unknown, 400, "fra_Latn"), (boom, 500, "no network"), (cancelled, 409, "cancelled")):
        monkeypatch.setattr(server.embedding_pipeline, "build_dataset", failure)
        res = _edit(client, CODES[:3])
        assert res.status_code == status and text in res.json()["detail"]
        assert server.store.data is before


def test_an_edit_of_a_dataset_that_was_replaced_meanwhile_is_refused(client, builds, monkeypatch):
    real_build = server.embedding_pipeline.build_dataset

    def build_while_someone_loads_another_dataset(*args, **kwargs):
        client.post("/dataset/upload?format=csv", content="id,lang,v0,v1\na,en,1,2\nb,de,3,4\n")
        return real_build(*args, **kwargs)

    monkeypatch.setattr(server.embedding_pipeline, "build_dataset", build_while_someone_loads_another_dataset)
    res = _edit(client, CODES[:3])
    assert res.status_code == 409 and "replaced" in res.json()["detail"]
    assert server.store.source == {"kind": "upload"}


def test_an_edit_keeps_the_generation_and_loading_a_new_dataset_does_not(client):
    generation = server.store.generation
    _edit(client, CODES[:3])
    assert server.store.generation == generation
    client.post("/dataset/flores", json={"lang_codes": CODES[:2], "n_sentences": 10})
    assert server.store.generation == generation + 1


def test_the_projection_after_an_edit_keeps_the_old_points_where_they_are(client, fits):
    before = client.get("/project?method=umap").json()
    _edit(client, CODES[:3])
    after = client.get("/project?method=umap").json()

    assert fits == [20]
    assert after["fit_id"] == before["fit_id"] and after["n_overlaid"] == 10
    old = {p["id"]: (p["x"], p["y"]) for p in before["points"]}
    assert all((p["x"], p["y"]) == old[p["id"]] for p in after["points"] if p["id"] in old)
    assert {p["id"] for p in after["points"] if p.get("overlaid")} == {f"fra_Latn-{i}" for i in range(10)}


def test_refit_folds_the_overlaid_points_into_a_new_fit(client, fits):
    client.get("/project?method=umap")
    _edit(client, CODES[:3])
    client.get("/project?method=umap")
    res = client.get("/project?method=umap&refit=true").json()
    assert fits == [20, 30] and res["n_overlaid"] == 0


def test_tokens_of_an_added_point_are_placed_in_the_same_layout(client, fits):
    client.get("/project?method=umap")
    _edit(client, CODES[:3])
    res = client.get("/project/tokens/fra_Latn-0?metric=euclidean")
    assert res.status_code == 200 and [t["token"] for t in res.json()["tokens"]] == ["t0", "t1", "t2"]
    assert fits == [20]
