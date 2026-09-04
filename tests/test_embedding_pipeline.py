import pytest
import torch

import embedding_pipeline as ep


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
