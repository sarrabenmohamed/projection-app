import copy
import json

import numpy as np
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
    assert by_id["en-0"]["family"] == "Germanic"


def test_family_of_maps_known_flores_codes():
    assert server.family_of("eng_Latn") == "Germanic"
    assert server.family_of("fra_Latn") == "Romance"
    assert server.family_of("not-a-real-code") == "Other"
    assert server.family_of("zzz_Latn") == "Other"


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


def test_reset_dataset_endpoint_reloads_default(client, tmp_path, monkeypatch):
    default_built = {
        "model": "default-model", "layer": "last", "pooling": "mean", "lang_codes": ["a"],
        "points": [{"id": "a-0", "lang": "a", "vector": [0.0, 0.0, 0.0]}],
    }

    def fake_build_dataset(lang_codes, n_sentences, model_name, pooling):
        return default_built

    monkeypatch.setattr(server.embedding_pipeline, "build_dataset", fake_build_dataset)
    monkeypatch.setattr(server, "DEFAULT_DATA_PATH", tmp_path / "cache.json")

    res = client.post("/dataset/reset")
    assert res.status_code == 200
    body = res.json()
    assert body["n_points"] == 1
    assert body["lang_codes"] == ["a"]
    assert server.DATASET["data"]["model"] == "default-model"


def test_reset_dataset_endpoint_500_and_keeps_current_dataset_on_failure(client, tmp_path, monkeypatch):
    def fake_build_dataset(*args, **kwargs):
        raise RuntimeError("no HuggingFace auth/network available")

    monkeypatch.setattr(server.embedding_pipeline, "build_dataset", fake_build_dataset)
    monkeypatch.setattr(server, "DEFAULT_DATA_PATH", tmp_path / "does-not-exist.json")

    previous_model = server.DATASET["data"]["model"]
    res = client.post("/dataset/reset")
    assert res.status_code == 500
    assert "Could not reload the default dataset" in res.json()["detail"]
    assert server.DATASET["data"]["model"] == previous_model  # untouched on failure


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


def test_project_umap_accepts_n_neighbors_and_min_dist(client):
    res = client.get("/project?method=umap&n_neighbors=2&min_dist=0.5")
    assert res.status_code == 200
    assert res.json()["params"] == {"n_neighbors": 2, "min_dist": 0.5}


def test_project_umap_clamps_n_neighbors_to_available_points(client):
    # SAMPLE_DATASET has 7 points, so n_neighbors must clamp to 6
    res = client.get("/project?method=umap&n_neighbors=100")
    assert res.status_code == 200
    assert res.json()["params"]["n_neighbors"] == len(SAMPLE_DATASET["points"]) - 1


def test_project_umap_refits_when_n_neighbors_or_min_dist_changes(client, monkeypatch):
    fit_calls = []
    orig_fit = server.umap.UMAP.fit

    def spy_fit(self, X, *a, **kw):
        fit_calls.append(1)
        return orig_fit(self, X, *a, **kw)

    monkeypatch.setattr(server.umap.UMAP, "fit", spy_fit)

    client.get("/project?method=umap&n_neighbors=2&min_dist=0.1")
    client.get("/project?method=umap&n_neighbors=2&min_dist=0.1")
    assert len(fit_calls) == 1  # same params reuse the cached reducer

    client.get("/project?method=umap&n_neighbors=3&min_dist=0.1")
    assert len(fit_calls) == 2  # n_neighbors change invalidates the cache

    client.get("/project?method=umap&n_neighbors=3&min_dist=0.9")
    assert len(fit_calls) == 3  # min_dist change invalidates the cache


def test_project_umap_rejects_out_of_range_hyperparams(client):
    assert client.get("/project?method=umap&n_neighbors=0").status_code == 422
    assert client.get("/project?method=umap&min_dist=-0.1").status_code == 422
    assert client.get("/project?method=umap&min_dist=1.1").status_code == 422


def test_project_tsne_accepts_perplexity_and_learning_rate(client):
    res = client.get("/project?method=tsne&perplexity=3&learning_rate=100")
    assert res.status_code == 200
    assert res.json()["params"] == {"perplexity": 3.0, "learning_rate": 100.0}


def test_project_tsne_clamps_perplexity_to_available_points(client):
    res = client.get("/project?method=tsne&perplexity=500")
    assert res.status_code == 200
    assert res.json()["params"]["perplexity"] == len(SAMPLE_DATASET["points"]) - 1


def test_project_tsne_rejects_out_of_range_hyperparams(client):
    assert client.get("/project?method=tsne&perplexity=0").status_code == 422
    assert client.get("/project?method=tsne&learning_rate=0").status_code == 422


def test_project_tokens_uses_matching_n_neighbors_and_min_dist(client, monkeypatch):
    fit_calls = []
    orig_fit = server.umap.UMAP.fit

    def spy_fit(self, X, *a, **kw):
        fit_calls.append(1)
        return orig_fit(self, X, *a, **kw)

    monkeypatch.setattr(server.umap.UMAP, "fit", spy_fit)

    client.get("/project?method=umap&n_neighbors=2&min_dist=0.2")
    client.get("/project/tokens/en-1?n_neighbors=2&min_dist=0.2")
    assert len(fit_calls) == 1  # matching params reuse the same cached reducer

    client.get("/project/tokens/en-1?n_neighbors=3&min_dist=0.2")
    assert len(fit_calls) == 2  # mismatched params force a refit


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


def test_stats_summarizes_selection_by_language(client):
    res = client.post("/stats", json={"ids": ["en-0", "en-2", "de-0"]})
    assert res.status_code == 200
    body = res.json()
    assert body["n_selected"] == 3
    assert body["n_languages"] == 2
    assert {l["lang"]: l["count"] for l in body["languages"]} == {"en": 2, "de": 1}


def test_stats_overall_cosine_splits_within_and_cross_language(client):
    body = client.post("/stats", json={"ids": ["en-0", "en-2", "de-0"]}).json()
    vec = {p["id"]: np.array(p["vector"]) for p in SAMPLE_DATASET["points"]}

    def cos(a, b):
        return float(vec[a] @ vec[b] / np.linalg.norm(vec[a]) / np.linalg.norm(vec[b]))

    assert body["overall"]["within_language"] == pytest.approx(cos("en-0", "en-2"))
    assert body["overall"]["cross_language"] == pytest.approx((cos("en-0", "de-0") + cos("en-2", "de-0")) / 2)


def test_stats_language_pair_compares_parallel_to_non_parallel(client):
    body = client.post("/stats", json={"ids": ["en-0", "de-0", "en-2", "de-2"]}).json()
    (pair,) = body["language_pairs"]
    assert (pair["a"], pair["b"]) == ("de", "en")
    assert pair["n_parallel"] == 2
    vec = {p["id"]: np.array(p["vector"]) for p in SAMPLE_DATASET["points"]}

    def cos(a, b):
        return float(vec[a] @ vec[b] / np.linalg.norm(vec[a]) / np.linalg.norm(vec[b]))

    assert pair["parallel_cosine"] == pytest.approx((cos("en-0", "de-0") + cos("en-2", "de-2")) / 2)
    assert pair["non_parallel_cosine"] == pytest.approx((cos("en-0", "de-2") + cos("en-2", "de-0")) / 2)


def test_stats_retrieval_accuracy_counts_mutual_nearest_translations():
    en = np.array([[1.0, 0.0], [0.0, 1.0]])
    de_aligned = np.array([[0.9, 0.1], [0.1, 0.9]])
    de_swapped = np.array([[0.1, 0.9], [0.9, 0.1]])
    assert server._retrieval_accuracy(en @ de_aligned.T) == 1.0
    assert server._retrieval_accuracy(en @ de_swapped.T) == 0.0


def test_stats_single_pair_has_no_retrieval_accuracy(client):
    body = client.post("/stats", json={"ids": ["en-0", "de-0"]}).json()
    assert body["language_pairs"][0]["n_parallel"] == 1
    assert body["language_pairs"][0]["retrieval_accuracy"] is None


def test_stats_one_language_has_no_pairs_or_cross_language_cosine(client):
    body = client.post("/stats", json={"ids": ["en-0", "en-2"]}).json()
    assert body["language_pairs"] == []
    assert body["overall"]["cross_language"] is None


def test_stats_single_point_has_no_pairwise_values(client):
    body = client.post("/stats", json={"ids": ["en-0"]}).json()
    assert body["n_selected"] == 1
    assert body["overall"]["within_language"] is None


def test_stats_empty_selection_is_400(client):
    assert client.post("/stats", json={"ids": []}).status_code == 400


def test_stats_unknown_id_is_404(client):
    assert client.post("/stats", json={"ids": ["en-0", "nope"]}).status_code == 404


def test_family_of_uses_glottolog_data_for_flores_codes():
    assert server.family_of("rus_Cyrl") == "Slavic"
    assert server.family_of("hin_Deva") == "Indo-Iranian"
    assert server.family_of("jpn_Jpan") == "Japonic"
    assert server.family_of("lvs_Latn") == "Baltic"
    assert server.family_of("zho_Hans") == server.family_of("zho_Hant") == "Sino-Tibetan"


def test_family_of_covers_languages_beyond_the_old_hardcoded_forty():
    assert server.family_of("swh_Latn") != "Other"
    assert server.family_of("quy_Latn") != "Other"


def test_family_of_override_wins_over_glottolog():
    assert server.family_of("eng_Latn", {"eng_Latn": "Custom"}) == "Custom"
    assert server.family_of("fra_Latn", {"eng_Latn": "Custom"}) == "Romance"


def test_points_endpoint_uses_dataset_lang_families():
    raw = copy.deepcopy(SAMPLE_DATASET)
    raw["lang_families"] = {"en": "Germanic", "de": "Germanic"}
    server.DATASET["data"] = server.validate_and_normalize(raw)
    server.DATASET["generation"] += 1
    by_id = {p["id"]: p for p in TestClient(server.app).get("/points").json()["points"]}
    assert by_id["en-0"]["family"] == "Germanic"
    assert by_id["de-0"]["family"] == "Germanic"


def test_points_endpoint_per_point_family_beats_lang_families():
    raw = copy.deepcopy(SAMPLE_DATASET)
    raw["lang_families"] = {"en": "Germanic"}
    raw["points"][0]["family"] = "Special"
    server.DATASET["data"] = server.validate_and_normalize(raw)
    server.DATASET["generation"] += 1
    by_id = {p["id"]: p for p in TestClient(server.app).get("/points").json()["points"]}
    assert by_id["en-0"]["family"] == "Special"
    assert by_id["en-2"]["family"] == "Germanic"


def test_stats_languages_report_dataset_families():
    raw = copy.deepcopy(SAMPLE_DATASET)
    raw["lang_families"] = {"en": "Special"}
    server.DATASET["data"] = server.validate_and_normalize(raw)
    server.DATASET["generation"] += 1
    body = TestClient(server.app).post("/stats", json={"ids": ["en-0", "de-0"]}).json()
    assert {l["lang"]: l["family"] for l in body["languages"]} == {"en": "Special", "de": "Germanic"}


def test_validate_and_normalize_rejects_non_object_lang_families():
    bad = copy.deepcopy(SAMPLE_DATASET)
    bad["lang_families"] = ["Germanic"]
    with pytest.raises(ValueError):
        server.validate_and_normalize(bad)


def test_validate_and_normalize_rejects_empty_point_family():
    bad = copy.deepcopy(SAMPLE_DATASET)
    bad["points"][0]["family"] = ""
    with pytest.raises(ValueError):
        server.validate_and_normalize(bad)


def test_family_of_resolves_iso1_codes_and_english_names():
    assert server.family_of("en") == "Germanic"
    assert server.family_of("fr") == "Romance"
    assert server.family_of("German") == "Germanic"
    assert server.family_of("english") == "Germanic"
    assert server.family_of("de-DE") == "Germanic"
    assert server.family_of("  Eng_Latn ") == "Germanic"
    assert server.family_of("zh") == "Sino-Tibetan"


def test_family_of_still_falls_back_to_other_for_nonsense():
    assert server.family_of("xx") == "Other"
    assert server.family_of("") == "Other"
