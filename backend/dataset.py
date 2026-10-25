"""The one embeddings dataset the server holds, and the rules a dataset must
satisfy before it is accepted."""

from __future__ import annotations

import math


class NoDatasetError(LookupError):
    """Nothing has been loaded yet."""


def _is_finite_vector(value, dim) -> bool:
    return (
        isinstance(value, list) and len(value) == dim
        and all(isinstance(v, (int, float)) and math.isfinite(v) for v in value)
    )


def validate_and_normalize(raw: dict) -> dict:
    """Checks a dataset dict (the JSON shape documented in the README; JSONL
    and CSV uploads are converted to it first) and fills in the optional
    fields. Raises ValueError with a message fit to show the user."""
    if not isinstance(raw, dict) or not isinstance(raw.get("points"), list) or len(raw["points"]) == 0:
        raise ValueError('expected an object with a non-empty "points" array')

    points = raw["points"]
    first_vector = points[0].get("vector")
    dim = raw.get("dim") or (len(first_vector) if isinstance(first_vector, list) else None)

    for i, p in enumerate(points):
        if not isinstance(p.get("id"), (str, int, float)):
            raise ValueError(f'points[{i}] is missing a valid "id"')
        if not isinstance(p.get("lang"), str):
            raise ValueError(f'points[{i}] is missing a valid "lang"')
        if "family" in p and (not isinstance(p["family"], str) or not p["family"].strip()):
            raise ValueError(f'points[{i}].family must be a non-empty string')
        if not _is_finite_vector(p.get("vector"), dim):
            raise ValueError(f'points[{i}].vector must be an array of {dim} finite numbers')

        token_vectors = p.get("tokenVectors")
        if token_vectors is not None:
            if not isinstance(token_vectors, list) or len(token_vectors) == 0:
                raise ValueError(f'points[{i}].tokenVectors must be a non-empty array')
            for j, t in enumerate(token_vectors):
                if not (isinstance(t, dict) and isinstance(t.get("token"), str) and _is_finite_vector(t.get("vector"), dim)):
                    raise ValueError(
                        f'points[{i}].tokenVectors[{j}] must be {{"token": str, "vector": array of {dim} finite numbers}}'
                    )

    lang_families = raw.get("lang_families")
    if lang_families is not None and (
        not isinstance(lang_families, dict)
        or not all(isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in lang_families.items())
    ):
        raise ValueError('"lang_families" must be an object mapping language codes to family names')

    return {
        "model": raw.get("model") or "unknown model",
        "layer": raw.get("layer") or "unknown",
        "pooling": raw.get("pooling") or "mean",
        "dim": dim,
        "lang_codes": raw.get("lang_codes") or sorted({p["lang"] for p in points}),
        "lang_families": lang_families or {},
        "points": points,
    }


class DatasetStore:
    """Holds the current dataset. `generation` increases on every replacement,
    so anything cached from an older dataset can tell it is stale."""

    def __init__(self):
        self.data = None
        self.generation = 0

    def replace(self, data: dict) -> dict:
        self.data = data
        self.generation += 1
        return self.summary()

    def current(self) -> dict:
        if self.data is None:
            raise NoDatasetError("No dataset loaded yet — load a FLORES-200 subset or upload an embeddings JSON")
        return self.data

    def summary(self) -> dict:
        d = self.current()
        return {"n_points": len(d["points"]), "lang_codes": d["lang_codes"], "dim": d["dim"]}
