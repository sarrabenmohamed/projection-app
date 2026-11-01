"""Loading FLORES-200 subsets: request validation, a per-language cache on
disk, and a loader that can be cancelled while it embeds."""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Callable, NamedTuple

from backend import embedding_pipeline
from backend.embedding_store import EmbeddingStore

# FLORES-200 devtest has 1012 sentences per language and 204 language/script variants.
FLORES_DEVTEST_SIZE = 1012
MAX_FLORES_LANGUAGES = 204
FLORES_CODE_RE = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")
POOLINGS = ("mean", "max", "cls")
DEFAULT_MODEL = "xlm-roberta-base"
DEFAULT_POOLING = "mean"


def flores_config(raw: dict) -> dict:
    """Validates a load request into the pipeline's config dict (ValueError on bad input)."""
    codes = raw.get("lang_codes")
    if not isinstance(codes, list) or not codes or not all(isinstance(c, str) for c in codes):
        raise ValueError("lang_codes must be a non-empty list of FLORES-200 codes like eng_Latn")
    codes = list(dict.fromkeys(c.strip() for c in codes))
    bad = [c for c in codes if not FLORES_CODE_RE.match(c)]
    if bad:
        raise ValueError(f"Not FLORES-200 language codes (expected e.g. eng_Latn): {', '.join(bad[:5])}")
    if len(codes) > MAX_FLORES_LANGUAGES:
        raise ValueError(f"At most {MAX_FLORES_LANGUAGES} languages")

    n = raw.get("n_sentences")
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= FLORES_DEVTEST_SIZE:
        raise ValueError(f"n_sentences must be an integer between 1 and {FLORES_DEVTEST_SIZE}")

    model = raw.get("model", DEFAULT_MODEL)
    if not isinstance(model, str) or not model.strip() or len(model) > 200:
        raise ValueError("model must be a non-empty Hugging Face model name")
    pooling = raw.get("pooling", DEFAULT_POOLING)
    if pooling not in POOLINGS:
        raise ValueError(f"pooling must be one of {', '.join(POOLINGS)}")
    return {"lang_codes": codes, "n_sentences": n, "model": model.strip(), "pooling": pooling}


def load_or_build_dataset(config: dict, builder: Callable, store: EmbeddingStore) -> dict:
    """Embeds only what the per-language cache lacks: whole languages that
    were never embedded, and the later sentences of languages cached for
    fewer sentences than requested. The result's vectors are views into the
    cache files, not copies.
    builder(lang_codes, n_sentences, model_name, pooling, start=0) -> dataset
    dict with sentences start..n_sentences-1 of each language."""
    n = config["n_sentences"]
    model, pooling = config["model"], config["pooling"]
    cached = {code: store.read(code, model, pooling) for code in config["lang_codes"]}

    # Languages needing the same sentences are built together, loading the model once per group.
    groups = {}
    for code, points in cached.items():
        if len(points) < n:
            groups.setdefault(len(points), []).append(code)

    for have, codes in sorted(groups.items()):
        built = builder(codes, n, config["model"], config["pooling"], start=have)
        for code in codes:
            store.write(code, model, pooling, cached[code] + [p for p in built["points"] if p["lang"] == code])
            cached[code] = store.read(code, model, pooling)

    points = [p for code in config["lang_codes"] for p in cached[code][:n]]
    return {
        "model": model, "layer": "last", "pooling": pooling, "dim": len(points[0]["vector"]),
        "lang_codes": config["lang_codes"], "lang_families": {}, "points": points,
    }


class LoadedFlores(NamedTuple):
    dataset: dict
    source: dict  # what DatasetStore.replace/revise record: {"kind": "flores", "n_sentences": n}


class FloresLoader:
    """Turns a load request into a validated dataset, embedding whatever the
    cache lacks. `build` is the embedding function (see
    embedding_pipeline.build_dataset); it is called with the languages to
    embed and a should_cancel callback."""

    def __init__(self, data_dir: Path, build: Callable):
        self.store = EmbeddingStore(data_dir)
        self._build = build
        self._cancel = threading.Event()

    def load(self, raw: dict) -> LoadedFlores:
        """Raises ValueError for a bad request, UnknownLanguageError for a code
        FLORES doesn't have, EmbeddingCancelled if cancel() (or a newer load)
        stopped it."""
        config = flores_config(raw)
        self._cancel.set()  # a newer load supersedes one still running
        cancel = self._cancel = threading.Event()

        def builder(*args, **kwargs):
            return self._build(*args, should_cancel=cancel.is_set, **kwargs)

        built = load_or_build_dataset(config, builder, self.store)
        if cancel.is_set():
            raise embedding_pipeline.EmbeddingCancelled()
        return LoadedFlores(built, {"kind": "flores", "n_sentences": config["n_sentences"]})

    def cancel(self) -> None:
        """Stops the running load at its next sentence; harmless when idle."""
        self._cancel.set()
