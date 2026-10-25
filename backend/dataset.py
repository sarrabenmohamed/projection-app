"""The one embeddings dataset the server holds, and the rules a dataset must
satisfy before it is accepted."""

from __future__ import annotations

import math
import threading
from typing import NamedTuple


class NoDatasetError(LookupError):
    """Nothing has been loaded yet."""


class DatasetChangedError(Exception):
    """The dataset was replaced while an edit of the previous one was running."""


UPLOADED = {"kind": "upload"}


class Snapshot(NamedTuple):
    """The dataset, where it came from, and its generation, as they were at one moment."""
    data: dict
    source: dict
    generation: int


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

    seen_ids = set()
    for i, p in enumerate(points):
        if not isinstance(p.get("id"), (str, int, float)):
            raise ValueError(f'points[{i}] is missing a valid "id"')
        if p["id"] in seen_ids:
            raise ValueError(f'points[{i}].id {p["id"]!r} is a duplicate; every point needs its own id')
        seen_ids.add(p["id"])
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
    """Holds the current dataset and where it came from (`source`: an upload,
    or a FLORES-200 subset with its sentence count).

    `generation` increases whenever the dataset is replaced, so anything
    cached from an older one (a projection layout) can tell it is stale.
    Revising — adding or removing points of the same dataset — keeps it, so
    that layout stays valid and the points can be placed into it."""

    def __init__(self):
        self._lock = threading.Lock()
        self.data = None
        self.source = None
        self.generation = 0

    def replace(self, data: dict, source: dict = UPLOADED) -> dict:
        with self._lock:
            self.data, self.source = data, source
            self.generation += 1
        return self.summary()

    def revise(self, data: dict, source: dict, *, based_on_generation: int) -> dict:
        """Swaps in an edited version of the dataset that was current at
        `based_on_generation`; DatasetChangedError if it was replaced since."""
        with self._lock:
            self.current()
            if self.generation != based_on_generation:
                raise DatasetChangedError("The dataset was replaced while it was being edited")
            self.data, self.source = data, source
        return self.summary()

    def snapshot(self) -> Snapshot:
        """The current dataset with its source and generation, consistent with each other."""
        with self._lock:
            return Snapshot(self.current(), self.source, self.generation)

    def current(self) -> dict:
        if self.data is None:
            raise NoDatasetError("No dataset loaded yet — load a FLORES-200 subset or upload an embeddings JSON")
        return self.data

    def summary(self) -> dict:
        d = self.current()
        return {"n_points": len(d["points"]), "lang_codes": d["lang_codes"], "dim": d["dim"], "source": self.source}
