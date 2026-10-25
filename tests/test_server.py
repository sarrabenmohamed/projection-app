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


def _fake_builder(calls):
    def build(lang_codes, n_sentences, model_name, pooling):
        calls.append((list(lang_codes), n_sentences, model_name, pooling))
        return {"points": [
            {"id": f"{c}-{i}", "lang": c, "pairId": i, "vector": [float(i), 1.0], "tokenVectors": []}
            for c in lang_codes for i in range(n_sentences)
        ]}
    return build


CACHE_CONFIG = {"model": "m", "lang_codes": ["a", "b"], "n_sentences": 3, "pooling": "mean"}


def test_load_or_build_dataset_embeds_only_new_languages(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    calls = []
    server._load_or_build_dataset(CACHE_CONFIG, _fake_builder(calls))
    assert calls == [(["a", "b"], 3, "m", "mean")]

    result = server._load_or_build_dataset({**CACHE_CONFIG, "lang_codes": ["a", "b", "c"]}, _fake_builder(calls))
    assert calls[1] == (["c"], 3, "m", "mean")  # a and b came from the cache
    assert [p["id"] for p in result["points"]][:1] == ["a-0"]
    assert result["lang_codes"] == ["a", "b", "c"] and len(result["points"]) == 9


def test_load_or_build_dataset_full_cache_hit_never_builds(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    server._load_or_build_dataset(CACHE_CONFIG, _fake_builder([]))

    def explode(*args):
        raise AssertionError("builder should not run on a cache hit")

    server._load_or_build_dataset(CACHE_CONFIG, explode)


def test_load_or_build_dataset_serves_smaller_n_from_larger_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    server._load_or_build_dataset(CACHE_CONFIG, _fake_builder([]))
    result = server._load_or_build_dataset({**CACHE_CONFIG, "n_sentences": 2}, lambda *a: 1 / 0)
    assert len(result["points"]) == 4


def test_load_or_build_dataset_rebuilds_language_when_more_sentences_needed(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    calls = []
    server._load_or_build_dataset({**CACHE_CONFIG, "n_sentences": 2}, _fake_builder(calls))
    server._load_or_build_dataset(CACHE_CONFIG, _fake_builder(calls))
    assert calls[1] == (["a", "b"], 3, "m", "mean")


def test_load_or_build_dataset_cache_is_per_model_and_pooling(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    calls = []
    server._load_or_build_dataset(CACHE_CONFIG, _fake_builder(calls))
    server._load_or_build_dataset({**CACHE_CONFIG, "model": "other"}, _fake_builder(calls))
    server._load_or_build_dataset({**CACHE_CONFIG, "pooling": "max"}, _fake_builder(calls))
    assert len(calls) == 3


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


def test_project_umap_3d_returns_z_and_refits_per_dimensionality(client, monkeypatch):
    fit_calls = []
    orig_fit = server.umap.UMAP.fit

    def spy_fit(self, X, *a, **kw):
        fit_calls.append(1)
        return orig_fit(self, X, *a, **kw)

    monkeypatch.setattr(server.umap.UMAP, "fit", spy_fit)

    flat = client.get("/project?method=umap").json()
    assert "z" not in flat["points"][0]

    res = client.get("/project?method=umap&n_components=3")
    assert res.status_code == 200
    body = res.json()
    assert body["n_components"] == 3
    assert all(isinstance(p["z"], float) for p in body["points"])
    assert len(fit_calls) == 2  # 2D and 3D are different fits


def test_project_tsne_3d_returns_z(client):
    res = client.get("/project?method=tsne&n_components=3")
    assert res.status_code == 200
    assert "z" in res.json()["points"][0]


def test_project_rejects_unsupported_n_components(client):
    assert client.get("/project?n_components=4").status_code == 422
    assert client.get("/project?n_components=1").status_code == 422


def test_project_tokens_3d_returns_z(client):
    client.get("/project?method=umap&n_components=3")
    res = client.get("/project/tokens/en-1?n_components=3")
    assert res.status_code == 200
    assert all("z" in t for t in res.json()["tokens"])


def test_project_tokens_404_for_unknown_id(client):
    res = client.get("/project/tokens/does-not-exist")
    assert res.status_code == 404


def test_project_tokens_400_when_point_has_no_token_vectors(client):
    res = client.get("/project/tokens/en-0")
    assert res.status_code == 400


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


def test_server_does_not_build_a_dataset_on_startup():
    assert not hasattr(server, "load_default_dataset")
    assert not server.app.router.on_startup


def test_points_404_with_helpful_message_when_empty():
    saved = dict(server.DATASET)
    try:
        server.DATASET["data"] = None
        res = TestClient(server.app).get("/points")
        assert res.status_code == 404
        assert "FLORES" in res.json()["detail"]
    finally:
        server.DATASET.update(saved)


def test_flores_config_validates_and_dedupes():
    cfg = server._flores_config({"lang_codes": ["eng_Latn", "deu_Latn", "eng_Latn"], "n_sentences": 25})
    assert cfg == {"lang_codes": ["eng_Latn", "deu_Latn"], "n_sentences": 25, "model": "xlm-roberta-base", "pooling": "mean"}


@pytest.mark.parametrize("raw", [
    {"lang_codes": [], "n_sentences": 10},
    {"lang_codes": ["english"], "n_sentences": 10},
    {"lang_codes": ["eng_Latn"], "n_sentences": 0},
    {"lang_codes": ["eng_Latn"], "n_sentences": 5000},
    {"lang_codes": ["eng_Latn"], "n_sentences": "ten"},
    {"lang_codes": "eng_Latn", "n_sentences": 10},
    {"lang_codes": ["eng_Latn"], "n_sentences": 10, "pooling": "median"},
    {"lang_codes": ["eng_Latn"], "n_sentences": 10, "model": ""},
])
def test_flores_config_rejects_bad_input(raw):
    with pytest.raises(ValueError):
        server._flores_config(raw)


def test_flores_endpoint_builds_loads_and_caches(client, tmp_path, monkeypatch):
    calls = []

    def fake_build_dataset(lang_codes, n_sentences, model_name, pooling, **kwargs):
        calls.append((lang_codes, n_sentences, model_name, pooling))
        return {
            "model": model_name, "layer": "last", "pooling": pooling, "lang_codes": lang_codes,
            "points": [
                {"id": f"{c}-{i}", "lang": c, "pairId": i, "vector": [float(i), 1.0], "tokenVectors": [{"token": "t", "vector": [0.0, 1.0]}]}
                for c in lang_codes for i in range(n_sentences)
            ],
        }

    monkeypatch.setattr(server.embedding_pipeline, "build_dataset", fake_build_dataset)
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)

    body = {"lang_codes": ["eng_Latn", "deu_Latn"], "n_sentences": 3}
    res = client.post("/dataset/flores", json=body)
    assert res.status_code == 200
    assert res.json()["n_points"] == 6
    assert calls == [(["eng_Latn", "deu_Latn"], 3, "xlm-roberta-base", "mean")]
    assert len(list(tmp_path.glob("flores_lang_*.json"))) == 2

    res = client.post("/dataset/flores", json=body)  # same config -> cache hit
    assert res.status_code == 200
    assert len(calls) == 1

    res = client.post("/dataset/flores", json={**body, "n_sentences": 2})  # smaller n -> served from the same cache
    assert res.json()["n_points"] == 4
    assert len(calls) == 1

    res = client.post("/dataset/flores", json={**body, "lang_codes": ["eng_Latn", "deu_Latn", "fra_Latn"]})
    assert res.json()["n_points"] == 9
    assert calls[1][0] == ["fra_Latn"]  # only the new language is embedded


def test_flores_cancel_stops_the_build_and_keeps_the_dataset(client, tmp_path, monkeypatch):
    def build_until_cancelled(lang_codes, n_sentences, model_name, pooling, should_cancel=None):
        assert client.post("/dataset/flores/cancel").json() == {"cancelled": True}
        assert should_cancel()
        raise server.embedding_pipeline.EmbeddingCancelled()

    monkeypatch.setattr(server.embedding_pipeline, "build_dataset", build_until_cancelled)
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    before = server.DATASET["data"]
    res = client.post("/dataset/flores", json={"lang_codes": ["eng_Latn"], "n_sentences": 3})
    assert res.status_code == 409
    assert "cancelled" in res.json()["detail"]
    assert server.DATASET["data"] is before
    assert not list(tmp_path.glob("flores_lang_*.json"))  # nothing half-built is cached


def test_flores_load_after_a_cancel_is_not_affected_by_it(client, tmp_path, monkeypatch):
    def build(lang_codes, n_sentences, model_name, pooling, should_cancel=None):
        assert not should_cancel()
        return {
            "model": model_name, "layer": "last", "pooling": pooling, "lang_codes": lang_codes,
            "points": [
                {"id": f"{c}-{i}", "lang": c, "pairId": i, "vector": [float(i), 1.0], "tokenVectors": [{"token": "t", "vector": [0.0, 1.0]}]}
                for c in lang_codes for i in range(n_sentences)
            ],
        }

    monkeypatch.setattr(server.embedding_pipeline, "build_dataset", build)
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    client.post("/dataset/flores/cancel")  # nothing running
    assert client.post("/dataset/flores", json={"lang_codes": ["eng_Latn"], "n_sentences": 3}).status_code == 200


def test_flores_endpoint_400_on_unknown_language(client, tmp_path, monkeypatch):
    def unknown(*args, **kwargs):
        raise server.embedding_pipeline.UnknownLanguageError('"zzz_Xxxx" is not a FLORES-200 language')

    monkeypatch.setattr(server.embedding_pipeline, "build_dataset", unknown)
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    res = client.post("/dataset/flores", json={"lang_codes": ["zzz_Xxxx"], "n_sentences": 3})
    assert res.status_code == 400
    assert "zzz_Xxxx" in res.json()["detail"]


def test_flores_endpoint_400_on_bad_config(client):
    res = client.post("/dataset/flores", json={"lang_codes": ["nope"], "n_sentences": 10})
    assert res.status_code == 400


def test_flores_endpoint_500_and_keeps_dataset_when_build_fails(client, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("no HuggingFace auth/network available")

    monkeypatch.setattr(server.embedding_pipeline, "build_dataset", boom)
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    before = server.DATASET["data"]
    res = client.post("/dataset/flores", json={"lang_codes": ["eng_Latn"], "n_sentences": 3})
    assert res.status_code == 500
    assert "no HuggingFace auth" in res.json()["detail"]
    assert server.DATASET["data"] is before


def test_server_serves_the_frontend(client):
    index = client.get("/")
    assert index.status_code == 200
    assert "Embedding Projection" in index.text
    assert client.get("/client-js/app.js").status_code == 200
    assert client.get("/css/style.css").status_code == 200
    assert client.get("/examples/sample_embeddings.json").status_code == 200


def test_server_does_not_expose_source_or_data(client):
    for path in ("/server.py", "/data/", "/tests/test_server.py", "/.git/config", "/requirements.txt"):
        assert client.get(path).status_code == 404


def test_upload_endpoint_accepts_jsonl(client):
    body = "\n".join(
        json.dumps({"id": f"x{i}", "lang": "en" if i % 2 else "de", "pairId": i // 2, "vector": [float(i), 1.0, 0.5]})
        for i in range(6)
    )
    res = client.post("/dataset/upload?format=jsonl", content=body)
    assert res.status_code == 200
    assert res.json()["n_points"] == 6 and res.json()["dim"] == 3


def test_upload_endpoint_accepts_csv(client):
    csv_text = "id,lang,text,v0,v1\na,en,hello,1,2\nb,de,hallo,3,4\n"
    res = client.post("/dataset/upload?format=csv", content=csv_text.encode("utf-8"))
    assert res.status_code == 200
    assert sorted(res.json()["lang_codes"]) == ["de", "en"]
    assert {p["text"] for p in server.DATASET["data"]["points"]} == {"hello", "hallo"}


def test_upload_endpoint_400_on_bad_file_and_keeps_dataset(client):
    before = server.DATASET["data"]
    res = client.post("/dataset/upload?format=csv", content="id,lang\na,en\n")
    assert res.status_code == 400
    assert "vector" in res.json()["detail"]
    assert client.post("/dataset/upload?format=nope", content="x").status_code == 400
    assert client.post("/dataset/upload?format=json", content=b"\xff\xfe").status_code == 400
    assert server.DATASET["data"] is before


def test_dataset_info_reports_loaded_and_empty(client, monkeypatch):
    info = client.get("/dataset/info").json()
    assert info["loaded"] is True and info["n_points"] == len(SAMPLE_DATASET["points"])

    monkeypatch.setitem(server.DATASET, "data", None)
    res = client.get("/dataset/info")
    assert res.status_code == 200 and res.json() == {"loaded": False}
