import json

import pytest

from backend import embedding_pipeline, flores

CONFIG = {"model": "m", "lang_codes": ["a", "b"], "n_sentences": 3, "pooling": "mean"}


def _fake_builder(calls):
    def build(lang_codes, n_sentences, model_name, pooling, should_cancel=None, start=0):
        calls.append((list(lang_codes), n_sentences, model_name, pooling, start))
        return {"points": [
            {"id": f"{c}-{i}", "lang": c, "pairId": i, "vector": [float(i), 1.0], "tokenVectors": []}
            for c in lang_codes for i in range(start, n_sentences)
        ]}
    return build


def _ids(dataset):
    return [p["id"] for p in dataset["points"]]


def _cache_file(tmp_path, code, config=CONFIG):
    return flores._language_cache_path(tmp_path, code, config["model"], config["pooling"])


def test_flores_config_validates_and_dedupes():
    cfg = flores.flores_config({"lang_codes": ["eng_Latn", "deu_Latn", "eng_Latn"], "n_sentences": 25})
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
        flores.flores_config(raw)


def test_embeds_only_new_languages(tmp_path):
    calls = []
    flores.load_or_build_dataset(CONFIG, _fake_builder(calls), tmp_path)
    assert calls == [(["a", "b"], 3, "m", "mean", 0)]

    result = flores.load_or_build_dataset({**CONFIG, "lang_codes": ["a", "b", "c"]}, _fake_builder(calls), tmp_path)
    assert calls[1] == (["c"], 3, "m", "mean", 0)  # a and b came from the cache
    assert _ids(result)[:1] == ["a-0"]
    assert result["lang_codes"] == ["a", "b", "c"] and len(result["points"]) == 9


def test_a_full_cache_hit_never_builds(tmp_path):
    flores.load_or_build_dataset(CONFIG, _fake_builder([]), tmp_path)

    def explode(*args, **kwargs):
        raise AssertionError("builder should not run on a cache hit")

    flores.load_or_build_dataset(CONFIG, explode, tmp_path)


def test_a_smaller_request_is_served_from_a_larger_cache(tmp_path):
    flores.load_or_build_dataset(CONFIG, _fake_builder([]), tmp_path)
    result = flores.load_or_build_dataset({**CONFIG, "n_sentences": 2}, lambda *a, **k: 1 / 0, tmp_path)
    assert _ids(result) == ["a-0", "a-1", "b-0", "b-1"]


def test_more_sentences_embed_only_the_missing_ones(tmp_path):
    calls = []
    flores.load_or_build_dataset({**CONFIG, "n_sentences": 2}, _fake_builder(calls), tmp_path)
    result = flores.load_or_build_dataset(CONFIG, _fake_builder(calls), tmp_path)

    assert calls[1] == (["a", "b"], 3, "m", "mean", 2)  # sentences 0-1 were cached
    assert _ids(result) == ["a-0", "a-1", "a-2", "b-0", "b-1", "b-2"]
    assert len(json.loads(_cache_file(tmp_path, "a").read_text())["points"]) == 3


def test_a_new_language_and_more_sentences_are_built_in_one_call_per_group(tmp_path):
    calls = []
    flores.load_or_build_dataset({**CONFIG, "lang_codes": ["a"], "n_sentences": 2}, _fake_builder(calls), tmp_path)
    result = flores.load_or_build_dataset(CONFIG, _fake_builder(calls), tmp_path)

    assert calls[1:] == [(["b"], 3, "m", "mean", 0), (["a"], 3, "m", "mean", 2)]
    assert _ids(result) == ["a-0", "a-1", "a-2", "b-0", "b-1", "b-2"]


def test_the_cache_is_per_model_and_pooling(tmp_path):
    calls = []
    flores.load_or_build_dataset(CONFIG, _fake_builder(calls), tmp_path)
    flores.load_or_build_dataset({**CONFIG, "model": "other"}, _fake_builder(calls), tmp_path)
    flores.load_or_build_dataset({**CONFIG, "pooling": "max"}, _fake_builder(calls), tmp_path)
    assert len(calls) == 3


def test_a_failed_extension_leaves_the_cache_as_it_was(tmp_path):
    flores.load_or_build_dataset({**CONFIG, "n_sentences": 2}, _fake_builder([]), tmp_path)
    before = _cache_file(tmp_path, "a").read_text()

    def cancelled(*args, **kwargs):
        raise embedding_pipeline.EmbeddingCancelled()

    with pytest.raises(embedding_pipeline.EmbeddingCancelled):
        flores.load_or_build_dataset(CONFIG, cancelled, tmp_path)
    assert _cache_file(tmp_path, "a").read_text() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_no_temporary_files_are_left_behind(tmp_path):
    flores.load_or_build_dataset(CONFIG, _fake_builder([]), tmp_path)
    assert sorted(p.suffix for p in tmp_path.iterdir()) == [".json", ".json"]


@pytest.mark.parametrize("content", [
    "not json",
    "[]",
    json.dumps({"points": []}),
    json.dumps({"points": [{"id": "a-0", "lang": "a", "pairId": 0, "vector": [0.0], "tokenVectors": None}]}),  # built before tokens were kept
    json.dumps({"points": [{"id": "a-1", "lang": "a", "pairId": 1, "vector": [0.0], "tokenVectors": []}]}),  # not a prefix of the sentences
])
def test_an_unusable_cache_file_is_rebuilt_from_the_start(tmp_path, content):
    _cache_file(tmp_path, "a").write_text(content)
    calls = []
    result = flores.load_or_build_dataset({**CONFIG, "lang_codes": ["a"]}, _fake_builder(calls), tmp_path)
    assert calls == [(["a"], 3, "m", "mean", 0)]
    assert _ids(result) == ["a-0", "a-1", "a-2"]


def test_the_loader_builds_through_the_injected_builder_and_cancel_stops_the_next_build(tmp_path):
    seen = []

    def build(codes, n, model, pooling, should_cancel=None, start=0):
        seen.append(should_cancel())
        data = _fake_builder([])(codes, n, model, pooling)
        for point in data["points"]:
            point["tokenVectors"] = [{"token": "t", "vector": point["vector"]}]
        return data

    loader = flores.FloresLoader(tmp_path, build)
    loader.load({"lang_codes": ["eng_Latn", "deu_Latn"], "n_sentences": 2})
    assert seen == [False]
    loader.cancel()
    loader.load({"lang_codes": ["fra_Latn"], "n_sentences": 2})  # a fresh load starts uncancelled
    assert seen == [False, False]


def test_the_loader_reports_where_the_dataset_came_from(tmp_path):
    def build(codes, n, model, pooling, should_cancel=None, start=0):
        data = _fake_builder([])(codes, n, model, pooling, start=start)
        for point in data["points"]:
            point["tokenVectors"] = [{"token": "t", "vector": point["vector"]}]
        return data

    loaded = flores.FloresLoader(tmp_path, build).load({"lang_codes": ["eng_Latn"], "n_sentences": 2})
    assert loaded.source == {"kind": "flores", "n_sentences": 2}
    assert _ids(loaded.dataset) == ["eng_Latn-0", "eng_Latn-1"]
