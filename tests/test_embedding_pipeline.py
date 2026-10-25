import pytest
import torch

from backend import embedding_pipeline as ep


def test_mean_pool_averages_only_unmasked_tokens():
    hidden = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [99.0, 99.0]]])  # 1 batch, 3 tokens, dim 2
    mask = torch.tensor([[1, 1, 0]])  # third token is padding
    pooled = ep._mean_pool(hidden, mask)
    assert pooled == pytest.approx([2.0, 0.0])


def test_token_vectors_excludes_special_tokens():
    hidden = torch.tensor([[[9.0, 9.0], [1.0, 2.0], [3.0, 4.0], [9.0, 9.0]]])
    tokens_str = ["<s>", "▁cat", "▁sat", "</s>"]
    special_mask = [1, 0, 0, 1]
    tvecs = ep._token_vectors(hidden, tokens_str, special_mask)
    assert [t["token"] for t in tvecs] == ["▁cat", "▁sat"]
    assert tvecs[0]["vector"] == pytest.approx([1.0, 2.0])
    assert tvecs[1]["vector"] == pytest.approx([3.0, 4.0])


def test_token_vectors_same_dim_as_hidden_size():
    hidden = torch.zeros((1, 2, 768))
    tvecs = ep._token_vectors(hidden, ["a", "b"], [0, 0])
    assert all(len(t["vector"]) == 768 for t in tvecs)


class _FakeDataset:
    def select(self, rows):
        return [{"sentence": f"s{i}"} for i in rows]


def _stub_pipeline(monkeypatch):
    class Model:
        def eval(self):
            pass

    monkeypatch.setattr(ep, "load_dataset", lambda *a, **kw: _FakeDataset())
    monkeypatch.setattr(ep.AutoTokenizer, "from_pretrained", lambda *a, **kw: object())
    monkeypatch.setattr(ep.AutoModel, "from_pretrained", lambda *a, **kw: Model())
    embedded = []

    def embed(sent, tokenizer, model, pooling):
        embedded.append(sent)
        return [0.0, 1.0], []

    monkeypatch.setattr(ep, "embed_sentence", embed)
    return embedded


def test_build_dataset_stops_when_cancelled(monkeypatch):
    embedded = _stub_pipeline(monkeypatch)
    with pytest.raises(ep.EmbeddingCancelled):
        ep.build_dataset(["eng_Latn"], 5, "m", should_cancel=lambda: len(embedded) >= 2)
    assert embedded == ["s0", "s1"]


def test_build_dataset_runs_to_the_end_when_not_cancelled(monkeypatch):
    embedded = _stub_pipeline(monkeypatch)
    result = ep.build_dataset(["eng_Latn"], 3, "m", should_cancel=lambda: False)
    assert len(result["points"]) == 3 and len(embedded) == 3
