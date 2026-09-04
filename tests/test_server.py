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
    ],
}


@pytest.fixture
def client():
    server.DATASET["data"] = server.validate_and_normalize(copy.deepcopy(SAMPLE_DATASET))
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


def test_pipeline_config_reads_env_vars(monkeypatch):
    monkeypatch.setenv("PROJECTION_LANGUAGES", "eng_Latn,deu_Latn")
    monkeypatch.setenv("PROJECTION_N_SENTENCES", "50")
    monkeypatch.setenv("PROJECTION_MODEL", "some-model")
    assert server._pipeline_config() == {
        "lang_codes": ["eng_Latn", "deu_Latn"], "n_sentences": 50, "model": "some-model",
    }


def test_cache_matches_true_when_config_matches():
    cached = {"model": "m", "lang_codes": ["a", "b"], "points": [{"lang": "a"}, {"lang": "a"}, {"lang": "b"}, {"lang": "b"}]}
    config = {"model": "m", "lang_codes": ["a", "b"], "n_sentences": 2}
    assert server._cache_matches(cached, config) is True


def test_cache_matches_false_on_model_mismatch():
    cached = {"model": "old", "lang_codes": ["a"], "points": [{"lang": "a"}]}
    config = {"model": "new", "lang_codes": ["a"], "n_sentences": 1}
    assert server._cache_matches(cached, config) is False


def test_cache_matches_false_on_sentence_count_mismatch():
    cached = {"model": "m", "lang_codes": ["a"], "points": [{"lang": "a"}]}
    config = {"model": "m", "lang_codes": ["a"], "n_sentences": 2}
    assert server._cache_matches(cached, config) is False


def test_load_or_build_dataset_uses_cache_on_hit(tmp_path):
    cache_path = tmp_path / "cache.json"
    cached = {"model": "m", "lang_codes": ["a"], "points": [{"lang": "a", "id": "a-0", "vector": [0.0]}]}
    cache_path.write_text(json.dumps(cached))
    config = {"model": "m", "lang_codes": ["a"], "n_sentences": 1}

    def fake_builder(*args):
        raise AssertionError("builder should not be called on a cache hit")

    result = server._load_or_build_dataset(cache_path, config, fake_builder)
    assert result == cached


def test_load_or_build_dataset_rebuilds_on_cache_miss(tmp_path):
    cache_path = tmp_path / "cache.json"
    config = {"model": "m", "lang_codes": ["a"], "n_sentences": 1}
    built = {"model": "m", "lang_codes": ["a"], "points": [{"lang": "a", "id": "a-0", "vector": [0.0]}]}

    def fake_builder(lang_codes, n_sentences, model_name):
        assert (lang_codes, n_sentences, model_name) == (["a"], 1, "m")
        return built

    result = server._load_or_build_dataset(cache_path, config, fake_builder)
    assert result == built
    assert json.loads(cache_path.read_text()) == built
