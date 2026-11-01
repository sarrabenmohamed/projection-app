"""The on-disk cache of embedded FLORES-200 sentences, one folder per
(model, pooling, language).

Each folder holds `vectors.npy` (one sentence vector per row),
`token_vectors.npy` (every sentence's token vectors, one after another) and
`meta.json` (ids, texts, token strings). Reading memory-maps the two arrays,
so a dataset costs only the pages that are actually touched; JSON text for the
same vectors was ~7x larger on disk and had to be parsed into Python floats.

Sentences 0..k-1 are stored in FLORES order, so a cache built for k sentences
also serves any smaller request and can be extended.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Sequence
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
_POINT_KEYS = {"id", "lang", "pairId", "text", "vector", "tokenVectors"}


class TokenVectors(Sequence):
    """One sentence's tokens, as a read-only sequence of {"token", "vector"}
    (the shape uploaded datasets use) backed by a slice of the token matrix."""

    def __init__(self, texts: list, rows: np.ndarray):
        self.texts = texts
        self._rows = rows

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        return {"token": self.texts[i], "vector": self._rows[i]}

    def matrix(self) -> np.ndarray:
        return np.asarray(self._rows, dtype=np.float32)


class EmbeddingStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def path(self, code: str, model: str, pooling: str) -> Path:
        key = json.dumps([model, pooling, code])
        return self.data_dir / f"flores_lang_{hashlib.sha1(key.encode()).hexdigest()[:12]}"

    def read(self, code: str, model: str, pooling: str) -> list:
        """Everything cached for one language, as point dicts whose vectors are
        views into the memory-mapped files. Empty if there is no usable cache.
        A cache still in the old JSON format is converted (and kept) first."""
        folder = self.path(code, model, pooling)
        if not folder.exists() and folder.with_suffix(".json").exists():
            try:
                self.import_json(folder.with_suffix(".json"))
            except (OSError, ValueError, KeyError, TypeError):
                return []
        try:
            return self._open(folder, code)
        except (OSError, ValueError, KeyError, TypeError):
            return []

    def _open(self, folder: Path, code: str) -> list:
        meta = json.loads((folder / "meta.json").read_text())
        if meta["format"] != FORMAT_VERSION:
            return []
        n = len(meta["ids"])
        vectors = np.load(folder / "vectors.npy", mmap_mode="r")
        token_rows = np.load(folder / "token_vectors.npy", mmap_mode="r")
        counts = meta["token_counts"]
        if n == 0 or vectors.shape[0] != n or len(meta["texts"]) != n or len(counts) != n or len(meta["tokens"]) != n:
            return []
        if meta["pair_ids"] != list(range(n)):  # extending something that isn't a prefix would corrupt it
            return []
        if token_rows.shape[0] != sum(counts):
            return []
        ends = np.cumsum(counts)
        points = []
        for i in range(n):
            start = int(ends[i] - counts[i])
            points.append({
                "id": meta["ids"][i], "lang": code, "pairId": i, "text": meta["texts"][i],
                "vector": vectors[i],
                "tokenVectors": TokenVectors(meta["tokens"][i], token_rows[start:start + counts[i]]),
            })
        return points

    def write(self, code: str, model: str, pooling: str, points: list) -> None:
        """Replaces the language's cache in one step, so an interrupted write
        can't leave a half-written folder behind. The old folder's files stay
        valid for anyone still holding them open."""
        folder = self.path(code, model, pooling)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        scratch = folder.with_name(f"{folder.name}.{uuid.uuid4().hex}.tmp")
        scratch.mkdir()
        retired = None
        try:
            self._write_files(scratch, code, points)
            if folder.exists():
                retired = folder.with_name(f"{folder.name}.{uuid.uuid4().hex}.old")
                os.replace(folder, retired)
            os.replace(scratch, folder)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
            if retired is not None:
                shutil.rmtree(retired, ignore_errors=True)

    @staticmethod
    def _write_files(folder: Path, code: str, points: list) -> None:
        dim = len(points[0]["vector"])
        vectors = np.asarray([p["vector"] for p in points], dtype=np.float32).reshape(len(points), dim)
        tokens = [p["tokenVectors"] for p in points]
        token_rows = np.asarray(
            [t["vector"] for sentence in tokens for t in sentence], dtype=np.float32).reshape(-1, dim)
        np.save(folder / "vectors.npy", vectors)
        np.save(folder / "token_vectors.npy", token_rows)
        meta = {
            "format": FORMAT_VERSION, "lang": code, "dim": dim,
            "ids": [p["id"] for p in points],
            "pair_ids": [p["pairId"] for p in points],
            "texts": [p.get("text") for p in points],
            "tokens": [[t["token"] for t in sentence] for sentence in tokens],
            "token_counts": [len(sentence) for sentence in tokens],
        }
        (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))

    def import_json(self, json_path: Path) -> Path:
        """Converts one old-format per-language JSON cache into the folder
        format, checks the result holds exactly the same numbers, and returns
        the folder. The JSON file is left in place."""
        cached = json.loads(json_path.read_text())
        points = cached["points"]
        if not points or any(p.get("tokenVectors") is None for p in points):
            raise ValueError("built before token vectors were kept")
        if any(set(p) - _POINT_KEYS for p in points):
            raise ValueError("has fields the folder format doesn't store")
        if any(p["pairId"] != i for i, p in enumerate(points)):
            raise ValueError("isn't a prefix of the FLORES sentences")

        code, model, pooling = cached["lang"], cached["model"], cached["pooling"]
        folder = self.path(code, model, pooling)
        if folder.with_suffix(".json") != json_path:
            raise ValueError("the file name doesn't match its model, pooling and language")
        self.write(code, model, pooling, points)
        try:
            self._verify(folder, code, points)
        except ValueError:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        return folder

    def covers_json(self, json_path: Path) -> bool:
        """Whether the folder already converted from this JSON file holds every
        number in it exactly (it may hold more, if it was extended since)."""
        cached = json.loads(json_path.read_text())
        folder = self.path(cached["lang"], cached["model"], cached["pooling"])
        if folder.with_suffix(".json") != json_path:
            return False
        try:
            self._verify(folder, cached["lang"], cached["points"])
        except (OSError, ValueError, KeyError, TypeError):
            return False
        return True

    def _verify(self, folder: Path, code: str, original: list) -> None:
        """Every number of `original` must be in the folder exactly as it was
        in the JSON (the folder may hold further sentences after them)."""
        stored = self._open(folder, code)
        same = len(stored) >= len(original) and len(original) > 0 and all(
            s["id"] == o["id"] and s["text"] == o["text"]
            and np.array_equal(np.asarray(s["vector"], dtype=np.float64), np.asarray(o["vector"], dtype=np.float64))
            and [t["token"] for t in s["tokenVectors"]] == [t["token"] for t in o["tokenVectors"]]
            and np.array_equal(
                s["tokenVectors"].matrix().astype(np.float64),
                np.asarray([t["vector"] for t in o["tokenVectors"]], dtype=np.float64).reshape(len(o["tokenVectors"]), s["vector"].shape[0]))
            for s, o in zip(stored, original)
        )
        if not same:
            raise ValueError(f"{folder.name} did not read back identical")
