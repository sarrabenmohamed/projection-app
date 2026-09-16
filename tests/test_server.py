import copy
import json

import pytest
from fastapi.testclient import TestClient

import server

SAMPLE_DATASET = {
    "model": "test-model",
    "layer": "last",
    "dim": 3,
    "lang_codes": ["en", "de"],
    "points": [
        {"id": "en-0", "lang": "en", "pairId": 0, "text": "hi", "vector": [0.1, 0.2, 0.3]},
        {"id": "de-0", "lang": "de", "pairId": 0, "text": "hallo", "vector": [0.15, 0.25, 0.35]},
        {
            "id": "en-1", "lang": "en", "pairId": 1, "text": "bye", "vector": [0.9, 0.8, 0.7],
            "tokenVectors": [
                {"token": "bye", "vector": [0.9, 0.8, 0.7]},
                {"token": "!", "vector": [0.5, 0.5, 0.5]},
            ],
        },
        {"id": "en-2", "lang": "en", "pairId": 2, "text": "thanks", "vector": [0.2, 0.3, 0.4]},
        {"id": "de-2", "lang": "de", "pairId": 2, "text": "danke", "vector": [0.25, 0.35, 0.45]},
        {"id": "en-3", "lang": "en", "pairId": 3, "text": "please", "vector": [0.3, 0.4, 0.5]},
        {"id": "de-3", "lang": "de", "pairId": 3, "text": "bitte", "vector": [0.35, 0.45, 0.55]},
    ],
}


@pytest.fixture
def client():
    server.DATASET["data"] = server.validate_and_normalize(copy.deepcopy(SAMPLE_DATASET))
    server.DATASET["generation"] += 1
    return TestClient(server.app)


def test_validate_and_normalize_accepts_dataset_without_token_vectors():
    normalized = server.validate_and_normalize(copy.deepcopy(SAMPLE_DATASET))
    assert normalized["points"][0].get("tokenVectors") is None


def test_validate_and_normalize_accepts_token_vectors():
    normalized = server.validate_and_normalize(copy.deepcopy(SAMPLE_DATASET))
    assert normalized["points"][2]["tokenVectors"][0] == {"token": "bye", "vector": [0.9, 0.8, 0.7]}


def test_validate_and_normalize_rejects_wrong_length_token_vector():
    bad = copy.deepcopy(SAMPLE_DATASET)
    bad["points"][2]["tokenVectors"] = [{"token": "bye", "vector": [0.9, 0.8]}]  # dim 2, dataset dim is 3
    with pytest.raises(ValueError):
        server.validate_and_normalize(bad)


def test_validate_and_normalize_rejects_token_vector_missing_token_string():
    bad = copy.deepcopy(SAMPLE_DATASET)
    bad["points"][2]["tokenVectors"] = [{"vector": [0.9, 0.8, 0.7]}]
    with pytest.raises(ValueError):
        server.validate_and_normalize(bad)


def test_points_endpoint_reports_has_tokens(client):
    res = client.get("/points")
    body = res.json()
    by_id = {p["id"]: p for p in body["points"]}
    assert by_id["en-0"]["hasTokens"] is False
    assert by_id["en-1"]["hasTokens"] is True


def test_points_endpoint_reports_language_family(client):
    res = client.get("/points")
    body = res.json()
    by_id = {p["id"]: p for p in body["points"]}
    # SAMPLE_DATASET's "en"/"de" codes aren't real FLORES-200 codes, so they
    # fall back to the shared "Other" bucket rather than a real family.
    assert by_id["en-0"]["family"] == "Other"


def test_family_of_maps_known_flores_codes():
    assert server.family_of("eng_Latn") == "Germanic"
    assert server.family_of("fra_Latn") == "Romance"
    assert server.family_of("not-a-real-code") == "Other"


def test_points_endpoint_reports_token_texts(client):
    res = client.get("/points")
    body = res.json()
    by_id = {p["id"]: p for p in body["points"]}
    assert by_id["en-0"]["tokenTexts"] is None
    assert by_id["en-1"]["tokenTexts"] == ["bye", "!"]


def test_pipeline_config_reads_env_vars(monkeypatch):
    monkeypatch.setenv("PROJECTION_LANGUAGES", "eng_Latn,deu_Latn")
    monkeypatch.setenv("PROJECTION_N_SENTENCES", "50")
    monkeypatch.setenv("PROJECTION_MODEL", "some-model")
    monkeypatch.setenv("PROJECTION_POOLING", "max")
    assert server._pipeline_config() == {
        "lang_codes": ["eng_Latn", "deu_Latn"], "n_sentences": 50, "model": "some-model", "pooling": "max",
    }


def test_cache_matches_true_when_config_matches():
    cached = {
        "model": "m", "lang_codes": ["a", "b"],
        "points": [
            {"lang": "a", "tokenVectors": []}, {"lang": "a", "tokenVectors": []},
            {"lang": "b", "tokenVectors": []}, {"lang": "b", "tokenVectors": []},
        ],
    }
    config = {"model": "m", "lang_codes": ["a", "b"], "n_sentences": 2, "pooling": "mean"}
    assert server._cache_matches(cached, config) is True


def test_cache_matches_false_when_points_missing_token_vectors():
    cached = {"model": "m", "lang_codes": ["a", "b"], "points": [{"lang": "a"}, {"lang": "a"}, {"lang": "b"}, {"lang": "b"}]}
    config = {"model": "m", "lang_codes": ["a", "b"], "n_sentences": 2, "pooling": "mean"}
    assert server._cache_matches(cached, config) is False


def test_cache_matches_false_on_model_mismatch():
    cached = {"model": "old", "lang_codes": ["a"], "points": [{"lang": "a"}]}
    config = {"model": "new", "lang_codes": ["a"], "n_sentences": 1, "pooling": "mean"}
    assert server._cache_matches(cached, config) is False


def test_cache_matches_false_on_sentence_count_mismatch():
    cached = {"model": "m", "lang_codes": ["a"], "points": [{"lang": "a"}]}
    config = {"model": "m", "lang_codes": ["a"], "n_sentences": 2, "pooling": "mean"}
    assert server._cache_matches(cached, config) is False


def test_cache_matches_false_on_pooling_mismatch():
    cached = {
        "model": "m", "lang_codes": ["a"], "pooling": "mean",
        "points": [{"lang": "a", "tokenVectors": []}],
    }
    config = {"model": "m", "lang_codes": ["a"], "n_sentences": 1, "pooling": "max"}
    assert server._cache_matches(cached, config) is False


def test_load_or_build_dataset_uses_cache_on_hit(tmp_path):
    cache_path = tmp_path / "cache.json"
    cached = {
        "model": "m", "lang_codes": ["a"],
        "points": [{"lang": "a", "id": "a-0", "vector": [0.0], "tokenVectors": []}],
    }
    cache_path.write_text(json.dumps(cached))
    config = {"model": "m", "lang_codes": ["a"], "n_sentences": 1, "pooling": "mean"}

    def fake_builder(*args):
        raise AssertionError("builder should not be called on a cache hit")

    result = server._load_or_build_dataset(cache_path, config, fake_builder)
    assert result == cached


def test_load_or_build_dataset_rebuilds_on_cache_miss(tmp_path):
    cache_path = tmp_path / "cache.json"
    config = {"model": "m", "lang_codes": ["a"], "n_sentences": 1, "pooling": "mean"}
    built = {"model": "m", "lang_codes": ["a"], "pooling": "mean", "points": [{"lang": "a", "id": "a-0", "vector": [0.0]}]}

    def fake_builder(lang_codes, n_sentences, model_name, pooling):
        assert (lang_codes, n_sentences, model_name, pooling) == (["a"], 1, "m", "mean")
        return built

    result = server._load_or_build_dataset(cache_path, config, fake_builder)
    assert result == built
    assert json.loads(cache_path.read_text()) == built


def test_load_default_dataset_degrades_gracefully_when_build_fails(tmp_path, monkeypatch):
    def fake_build_dataset(*args, **kwargs):
        raise RuntimeError("no HuggingFace auth/network available")

    monkeypatch.setattr(server.embedding_pipeline, "build_dataset", fake_build_dataset)
    monkeypatch.setattr(server, "DEFAULT_DATA_PATH", tmp_path / "does-not-exist.json")

    saved_dataset = copy.deepcopy(server.DATASET)
    try:
        server.DATASET["data"] = None
        server.DATASET["generation"] = 0

        server.load_default_dataset()  # must not raise

        assert server.DATASET["data"] is None
    finally:
        server.DATASET["data"] = saved_dataset["data"]
        server.DATASET["generation"] = saved_dataset["generation"]


def test_project_umap_refits_only_when_dataset_changes(client, monkeypatch):
    fit_calls = []
    orig_fit = server.umap.UMAP.fit

    def spy_fit(self, X, *a, **kw):
        fit_calls.append(1)
        return orig_fit(self, X, *a, **kw)

    monkeypatch.setattr(server.umap.UMAP, "fit", spy_fit)

    client.get("/project?method=umap")
    client.get("/project?method=umap")
    assert len(fit_calls) == 1  # second call reuses the cached reducer

    client.post("/dataset", json=copy.deepcopy(SAMPLE_DATASET))
    client.get("/project?method=umap")
    assert len(fit_calls) == 2  # dataset change invalidates the cache


def test_project_umap_refits_when_metric_changes(client, monkeypatch):
    fit_calls = []
    orig_fit = server.umap.UMAP.fit

    def spy_fit(self, X, *a, **kw):
        fit_calls.append(1)
        return orig_fit(self, X, *a, **kw)

    monkeypatch.setattr(server.umap.UMAP, "fit", spy_fit)

    client.get("/project?method=umap&metric=euclidean")
    client.get("/project?method=umap&metric=euclidean")
    assert len(fit_calls) == 1  # same metric reuses the cached reducer

    client.get("/project?method=umap&metric=cosine")
    assert len(fit_calls) == 2  # metric change invalidates the cache


def test_project_tsne_accepts_metric(client):
    res = client.get("/project?method=tsne&metric=cosine")
    assert res.status_code == 200
    assert len(res.json()["points"]) == len(SAMPLE_DATASET["points"])


def test_project_tokens_success(client):
    res = client.get("/project/tokens/en-1")
    assert res.status_code == 200
    body = res.json()
    assert body["id"] == "en-1"
    assert {t["token"] for t in body["tokens"]} == {"bye", "!"}
    assert all(isinstance(t["x"], float) and isinstance(t["y"], float) for t in body["tokens"])


def test_project_tokens_404_for_unknown_id(client):
    res = client.get("/project/tokens/does-not-exist")
    assert res.status_code == 404


def test_project_tokens_400_when_point_has_no_token_vectors(client):
    res = client.get("/project/tokens/en-0")
    assert res.status_code == 400


def test_neighbors_excludes_self_and_respects_k(client):
    res = client.get("/neighbors/en-0?k=2")
    assert res.status_code == 200
    body = res.json()
    assert body["id"] == "en-0"
    assert len(body["neighbors"]) == 2
    assert all(n["id"] != "en-0" for n in body["neighbors"])


def test_neighbors_sorted_by_ascending_distance(client):
    res = client.get("/neighbors/en-0?k=5")
    dists = [n["distance"] for n in res.json()["neighbors"]]
    assert dists == sorted(dists)


def test_neighbors_404_for_unknown_id(client):
    res = client.get("/neighbors/does-not-exist")
    assert res.status_code == 404


def test_neighbors_rejects_k_out_of_range(client):
    assert client.get("/neighbors/en-0?k=0").status_code == 422
    assert client.get("/neighbors/en-0?k=101").status_code == 422
