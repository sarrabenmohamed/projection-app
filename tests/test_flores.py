import json

import pytest

from backend import embedding_pipeline, flores
from backend.embedding_store import EmbeddingStore

def _store(tmp_path):
    return EmbeddingStore(tmp_path)


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
    return EmbeddingStore(tmp_path).path(code, config["model"], config["pooling"])


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
    flores.load_or_build_dataset(CONFIG, _fake_builder(calls), _store(tmp_path))
    assert calls == [(["a", "b"], 3, "m", "mean", 0)]

    result = flores.load_or_build_dataset({**CONFIG, "lang_codes": ["a", "b", "c"]}, _fake_builder(calls), _store(tmp_path))
    assert calls[1] == (["c"], 3, "m", "mean", 0)  # a and b came from the cache
    assert _ids(result)[:1] == ["a-0"]
    assert result["lang_codes"] == ["a", "b", "c"] and len(result["points"]) == 9


def test_a_full_cache_hit_never_builds(tmp_path):
    flores.load_or_build_dataset(CONFIG, _fake_builder([]), _store(tmp_path))

    def explode(*args, **kwargs):
        raise AssertionError("builder should not run on a cache hit")

    flores.load_or_build_dataset(CONFIG, explode, _store(tmp_path))


def test_a_smaller_request_is_served_from_a_larger_cache(tmp_path):
    flores.load_or_build_dataset(CONFIG, _fake_builder([]), _store(tmp_path))
    result = flores.load_or_build_dataset({**CONFIG, "n_sentences": 2}, lambda *a, **k: 1 / 0, _store(tmp_path))
    assert _ids(result) == ["a-0", "a-1", "b-0", "b-1"]


def test_more_sentences_embed_only_the_missing_ones(tmp_path):
    calls = []
    flores.load_or_build_dataset({**CONFIG, "n_sentences": 2}, _fake_builder(calls), _store(tmp_path))
    result = flores.load_or_build_dataset(CONFIG, _fake_builder(calls), _store(tmp_path))

    assert calls[1] == (["a", "b"], 3, "m", "mean", 2)  # sentences 0-1 were cached
    assert _ids(result) == ["a-0", "a-1", "a-2", "b-0", "b-1", "b-2"]
    assert len(_store(tmp_path).read("a", "m", "mean")) == 3


def test_a_new_language_and_more_sentences_are_built_in_one_call_per_group(tmp_path):
    calls = []
    flores.load_or_build_dataset({**CONFIG, "lang_codes": ["a"], "n_sentences": 2}, _fake_builder(calls), _store(tmp_path))
    result = flores.load_or_build_dataset(CONFIG, _fake_builder(calls), _store(tmp_path))

    assert calls[1:] == [(["b"], 3, "m", "mean", 0), (["a"], 3, "m", "mean", 2)]
    assert _ids(result) == ["a-0", "a-1", "a-2", "b-0", "b-1", "b-2"]


def test_the_cache_is_per_model_and_pooling(tmp_path):
    calls = []
    flores.load_or_build_dataset(CONFIG, _fake_builder(calls), _store(tmp_path))
    flores.load_or_build_dataset({**CONFIG, "model": "other"}, _fake_builder(calls), _store(tmp_path))
    flores.load_or_build_dataset({**CONFIG, "pooling": "max"}, _fake_builder(calls), _store(tmp_path))
    assert len(calls) == 3


def test_a_failed_extension_leaves_the_cache_as_it_was(tmp_path):
    flores.load_or_build_dataset({**CONFIG, "n_sentences": 2}, _fake_builder([]), _store(tmp_path))
    before = (_cache_file(tmp_path, "a") / "meta.json").read_text()

    def cancelled(*args, **kwargs):
        raise embedding_pipeline.EmbeddingCancelled()

    with pytest.raises(embedding_pipeline.EmbeddingCancelled):
        flores.load_or_build_dataset(CONFIG, cancelled, _store(tmp_path))
    assert (_cache_file(tmp_path, "a") / "meta.json").read_text() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_no_temporary_files_are_left_behind(tmp_path):
    flores.load_or_build_dataset(CONFIG, _fake_builder([]), _store(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(p.name for p in [_cache_file(tmp_path, "a"), _cache_file(tmp_path, "b")])


def _damage(folder, how):
    meta_path = folder / "meta.json"
    meta = json.loads(meta_path.read_text())
    if how == "not json":
        meta_path.write_text("not json")
    elif how == "missing array":
        (folder / "vectors.npy").unlink()
    elif how == "not a prefix":
        meta["pair_ids"] = [1, 2, 3]
        meta_path.write_text(json.dumps(meta))
    elif how == "wrong token count":
        meta["token_counts"] = [5, 5, 5]
        meta_path.write_text(json.dumps(meta))
    elif how == "other format":
        meta["format"] = 99
        meta_path.write_text(json.dumps(meta))


@pytest.mark.parametrize("how", ["not json", "missing array", "not a prefix", "wrong token count", "other format"])
def test_an_unusable_cache_is_rebuilt_from_the_start(tmp_path, how):
    flores.load_or_build_dataset({**CONFIG, "lang_codes": ["a"]}, _fake_builder([]), _store(tmp_path))
    _damage(_cache_file(tmp_path, "a"), how)
    calls = []
    result = flores.load_or_build_dataset({**CONFIG, "lang_codes": ["a"]}, _fake_builder(calls), _store(tmp_path))
    assert calls == [(["a"], 3, "m", "mean", 0)]
    assert _ids(result) == ["a-0", "a-1", "a-2"]


def test_cached_vectors_come_back_exactly_with_their_tokens(tmp_path):
    def build(lang_codes, n_sentences, model_name, pooling, should_cancel=None, start=0):
        return {"points": [
            {"id": f"a-{i}", "lang": "a", "pairId": i, "text": f"s{i}", "vector": [i * 0.5, 0.25],
             "tokenVectors": [{"token": f"t{j}", "vector": [j, i + 0.125]} for j in range(i + 1)]}
            for i in range(start, n_sentences)]}

    first = flores.load_or_build_dataset({**CONFIG, "lang_codes": ["a"]}, build, _store(tmp_path))
    again = flores.load_or_build_dataset({**CONFIG, "lang_codes": ["a"]}, lambda *a, **k: 1 / 0, _store(tmp_path))
    for result in (first, again):
        assert result["dim"] == 2
        point = result["points"][2]
        assert point["text"] == "s2" and list(point["vector"]) == [1.0, 0.25]
        assert [t["token"] for t in point["tokenVectors"]] == ["t0", "t1", "t2"]
        assert [list(t["vector"]) for t in point["tokenVectors"]] == [[0.0, 2.125], [1.0, 2.125], [2.0, 2.125]]


def test_an_old_json_cache_is_converted_on_first_read_and_kept(tmp_path):
    legacy = _cache_file(tmp_path, "a").with_suffix(".json")
    legacy.write_text(json.dumps({"model": "m", "pooling": "mean", "lang": "a", "points": [
        {"id": f"a-{i}", "lang": "a", "pairId": i, "text": f"s{i}", "vector": [i * 0.5, 1.0],
         "tokenVectors": [{"token": "t", "vector": [0.25, i]}]} for i in range(3)]}))
    result = flores.load_or_build_dataset({**CONFIG, "lang_codes": ["a"]}, lambda *a, **k: 1 / 0, _store(tmp_path))
    assert _ids(result) == ["a-0", "a-1", "a-2"]
    assert legacy.exists()


def test_a_legacy_cache_without_token_vectors_is_not_converted(tmp_path):
    legacy = _cache_file(tmp_path, "a").with_suffix(".json")
    legacy.write_text(json.dumps({"model": "m", "pooling": "mean", "lang": "a", "points": [
        {"id": "a-0", "lang": "a", "pairId": 0, "text": "s", "vector": [0.0], "tokenVectors": None}]}))
    assert _store(tmp_path).read("a", "m", "mean") == []


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


def test_a_converted_json_is_covered_even_after_the_folder_was_extended_but_not_if_it_differs(tmp_path):
    store = _store(tmp_path)
    legacy = _cache_file(tmp_path, "a").with_suffix(".json")
    points = [{"id": f"a-{i}", "lang": "a", "pairId": i, "text": f"s{i}", "vector": [i * 0.5, 1.0],
               "tokenVectors": [{"token": "t", "vector": [0.25, i]}]} for i in range(3)]
    legacy.write_text(json.dumps({"model": "m", "pooling": "mean", "lang": "a", "points": points}))
    store.import_json(legacy)
    assert store.covers_json(legacy)

    extra = {"id": "a-3", "lang": "a", "pairId": 3, "text": "s3", "vector": [9.0, 1.0], "tokenVectors": [{"token": "t", "vector": [1.0, 1.0]}]}
    store.write("a", "m", "mean", points + [extra])
    assert store.covers_json(legacy)

    store.write("a", "m", "mean", [{**points[0], "vector": [7.0, 1.0]}] + points[1:])
    assert not store.covers_json(legacy)
    assert len(store.read("a", "m", "mean")) == 3  # a failed check never touches the folder
