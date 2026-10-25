"""Loading FLORES-200 subsets: request validation, a per-language cache on
disk, and a loader that can be cancelled while it embeds."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Callable

from backend import embedding_pipeline
from backend.dataset import validate_and_normalize

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


def _language_cache_path(data_dir: Path, code: str, model: str, pooling: str) -> Path:
    key = json.dumps([model, pooling, code])
    return data_dir / f"flores_lang_{hashlib.sha1(key.encode()).hexdigest()[:12]}.json"


def _read_language_cache(data_dir: Path, code: str, config: dict):
    """Cached points for one language (first n_sentences of them), or None.
    FLORES order is fixed, so a cache built for N sentences also serves any
    smaller request."""
    path = _language_cache_path(data_dir, code, config["model"], config["pooling"])
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text())
    except ValueError:
        return None
    points = cached.get("points") or []
    n = config["n_sentences"]
    if len(points) < n or points[0].get("tokenVectors") is None:
        return None
    return points[:n]


def load_or_build_dataset(config: dict, builder: Callable, data_dir: Path) -> dict:
    """Embeds only the languages missing from the per-language cache.
    builder(lang_codes, n_sentences, model_name, pooling) -> dataset dict."""
    by_lang = {}
    missing = []
    for code in config["lang_codes"]:
        points = _read_language_cache(data_dir, code, config)
        if points is None:
            missing.append(code)
        else:
            by_lang[code] = points

    if missing:
        built = builder(missing, config["n_sentences"], config["model"], config["pooling"])
        data_dir.mkdir(parents=True, exist_ok=True)
        for code in missing:
            points = [p for p in built["points"] if p["lang"] == code]
            by_lang[code] = points
            _language_cache_path(data_dir, code, config["model"], config["pooling"]).write_text(
                json.dumps({"model": config["model"], "pooling": config["pooling"], "lang": code, "points": points})
            )

    points = [p for code in config["lang_codes"] for p in by_lang[code]]
    return {
        "model": config["model"], "layer": "last", "pooling": config["pooling"],
        "lang_codes": config["lang_codes"], "points": points,
    }


class FloresLoader:
    """Turns a load request into a validated dataset, embedding whatever the
    cache lacks. `build` is the embedding function (see
    embedding_pipeline.build_dataset); it is called with the languages to
    embed and a should_cancel callback."""

    def __init__(self, data_dir: Path, build: Callable):
        self.data_dir = data_dir
        self._build = build
        self._cancel = threading.Event()

    def load(self, raw: dict) -> dict:
        """Raises ValueError for a bad request, UnknownLanguageError for a code
        FLORES doesn't have, EmbeddingCancelled if cancel() (or a newer load)
        stopped it."""
        config = flores_config(raw)
        self._cancel.set()  # a newer load supersedes one still running
        cancel = self._cancel = threading.Event()

        def builder(*args):
            return self._build(*args, should_cancel=cancel.is_set)

        built = load_or_build_dataset(config, builder, self.data_dir)
        if cancel.is_set():
            raise embedding_pipeline.EmbeddingCancelled()
        return validate_and_normalize(built)

    def cancel(self) -> None:
        """Stops the running load at its next sentence; harmless when idle."""
        self._cancel.set()
