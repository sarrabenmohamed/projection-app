"""Alignment statistics for a selection of points: how similar are the
embeddings within a language, across languages, and between true translations.
Always cosine similarity in the full embedding space (not the 2D/3D
projection), the usual convention for alignment."""

from __future__ import annotations

from typing import Optional

import numpy as np

from backend.languages import point_family


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _mean_pairwise_cosine_sums(unit: np.ndarray) -> tuple[float, int]:
    """(sum of cosine over all unordered pairs, number of such pairs) for unit
    rows, via |sum u|^2 = n + 2 * sum_{i<j} u_i . u_j — O(n*d), no n x n matrix."""
    n = len(unit)
    total = unit.sum(axis=0)
    return (float(total @ total) - n) / 2, n * (n - 1) // 2


def _retrieval_accuracy(sim: np.ndarray) -> Optional[float]:
    """MEXA-style score for a square similarity matrix whose diagonal holds the
    true translation pairs: the fraction of diagonal entries that strictly beat
    every other entry in their row and in their column."""
    n = len(sim)
    if n < 2:
        return None
    diag = np.diag(sim)
    off = sim.copy()
    np.fill_diagonal(off, -np.inf)
    hits = (diag > off.max(axis=1)) & (diag > off.max(axis=0))
    return float(hits.mean())


def _mean_or_none(values: list[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _overall_cosines(unit_by_lang: dict) -> dict:
    """Mean pairwise cosine within a language and across languages."""
    all_sum, all_pairs = _mean_pairwise_cosine_sums(np.vstack(list(unit_by_lang.values())))
    within_sum = within_pairs = 0
    for unit in unit_by_lang.values():
        s, n_pairs = _mean_pairwise_cosine_sums(unit)
        within_sum += s
        within_pairs += n_pairs
    cross_pairs = all_pairs - within_pairs
    return {
        "within_language": (within_sum / within_pairs) if within_pairs else None,
        "cross_language": ((all_sum - within_sum) / cross_pairs) if cross_pairs else None,
    }


def _language_pair(a: str, b: str, unit_by_lang: dict, first_row_by_lang: dict) -> dict:
    """Translations (same pairId) of languages a and b against all other cross pairs."""
    sim = unit_by_lang[a] @ unit_by_lang[b].T
    common = sorted(set(first_row_by_lang[a]) & set(first_row_by_lang[b]), key=str)
    rows = [first_row_by_lang[a][pid] for pid in common]
    cols = [first_row_by_lang[b][pid] for pid in common]
    parallel = sim[rows, cols]
    n_other = sim.size - len(common)
    return {
        "a": a, "b": b,
        "n_parallel": len(common),
        "parallel_cosine": float(parallel.mean()) if len(common) else None,
        "non_parallel_cosine": float((sim.sum() - parallel.sum()) / n_other) if n_other else None,
        "retrieval_accuracy": _retrieval_accuracy(sim[np.ix_(rows, cols)]) if len(common) else None,
    }


def selection_stats(dataset: dict, ids: list) -> dict:
    """Statistics over the points with these ids. `overall` splits the mean
    pairwise cosine into same-language and cross-language pairs;
    `language_pairs` compares, per language pair, the mean cosine of true
    translations against all other cross-language pairs, plus retrieval
    accuracy over the translations. Raises ValueError for an empty selection
    and LookupError for an unknown id."""
    if not ids:
        raise ValueError("select at least one point")
    by_id = {str(p["id"]): p for p in dataset["points"]}
    missing = [i for i in ids if str(i) not in by_id]
    if missing:
        raise LookupError(f"no point with id {str(missing[0])!r}")

    selected = list({str(i): by_id[str(i)] for i in ids}.values())
    langs = sorted({p["lang"] for p in selected})
    unit_by_lang: dict[str, np.ndarray] = {}
    first_row_by_lang: dict[str, dict] = {}  # lang -> pairId -> row in that language's matrix
    for lang in langs:
        members = [p for p in selected if p["lang"] == lang]
        unit_by_lang[lang] = _unit_rows(np.array([p["vector"] for p in members], dtype=np.float64))
        first_row: dict = {}
        for row, p in enumerate(members):
            if p.get("pairId") is not None:
                first_row.setdefault(p["pairId"], row)
        first_row_by_lang[lang] = first_row

    family_by_lang: dict = {}
    for p in selected:
        family_by_lang.setdefault(p["lang"], point_family(p, dataset))

    language_pairs = [
        _language_pair(a, b, unit_by_lang, first_row_by_lang)
        for ai, a in enumerate(langs) for b in langs[ai + 1:]
    ]

    def mean_of(key: str) -> Optional[float]:
        return _mean_or_none([lp[key] for lp in language_pairs if lp[key] is not None])

    return {
        "n_selected": len(selected),
        "n_languages": len(langs),
        "languages": [
            {"lang": lang, "family": family_by_lang[lang], "count": len(unit_by_lang[lang])} for lang in langs
        ],
        "overall": _overall_cosines(unit_by_lang),
        "summary": {
            "parallel_cosine": mean_of("parallel_cosine"),
            "non_parallel_cosine": mean_of("non_parallel_cosine"),
            "retrieval_accuracy": mean_of("retrieval_accuracy"),
        },
        "language_pairs": language_pairs,
    }
