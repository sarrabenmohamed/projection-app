"""Experiment: how well can a new language be overlaid on a finished layout?

Takes cached FLORES-200 embeddings (the data/flores_lang_* folders), lays out the base
languages, then places one held-out language into that layout with each
method, and compares against laying everything out again from scratch.

UMAP family:   refit | UMAP.transform | anchored (backend/overlay.py) | parametric UMAP
t-SNE family:  refit | initial placement only | joint t-SNE (Wang et al. 2022)

Metrics, all on the held-out points:
  old shift     mean move of the base points, relative to the layout's radius
  kNN overlap   share of a held-out point's 10 nearest neighbours in the full
                space that are also among its 10 nearest in 2D
  translation   share of held-out sentences whose 2D-nearest point in a base
                language is that sentence's translation (chance is 1 / N)
  new-new       share of a held-out point's 2D neighbours that are held out too
                (compare with the same number measured in the full space)

Parametric UMAP is a small torch MLP trained with UMAP's cross-entropy loss
on the fitted graph (umap.parametric_umap needs TensorFlow, which this
project does not install). Joint t-SNE comes from the vendored code in the
embeddings-vis-tool project; pass --joint-tsne-path if it lives elsewhere.

Usage:
    python scripts/overlay_experiment.py --family umap --n 200
    python scripts/overlay_experiment.py --family tsne --n 60
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.embedding_store import EmbeddingStore  # noqa: E402
from backend.overlay import place_new_points  # noqa: E402
from backend.umap_import import umap  # noqa: E402

K = 10
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
BASE = ["eng_Latn", "fra_Latn", "deu_Latn", "rus_Cyrl", "hin_Deva", "jpn_Jpan"]
HELD_OUT = ["spa_Latn", "tha_Thai"]  # a close relative of the base set, and an isolate


def load_language(lang: str, n: int, data_dir: Path):
    points = EmbeddingStore(data_dir).read(lang, "xlm-roberta-base", "mean")[:n]
    if not points:
        raise SystemExit(f"no cached xlm-roberta-base/mean embeddings for {lang} in {data_dir}")
    return np.array([p["vector"] for p in points], dtype=np.float32), [p["pairId"] for p in points]


def knn(points: np.ndarray, k: int) -> np.ndarray:
    sq = (points ** 2).sum(1)
    dist = sq[:, None] + sq[None, :] - 2 * points @ points.T
    np.fill_diagonal(dist, np.inf)
    return np.argsort(dist, axis=1)[:, :k]


def evaluate(x_all, y_all, n_old, labels, pair_ids, y_old_before, y_old_after):
    """Metrics for the held-out rows (those from n_old on) of a full layout."""
    new = np.arange(n_old, len(x_all))
    hi, lo = knn(x_all, K)[new], knn(y_all, K)[new]
    overlap = float(np.mean([len(set(h) & set(l)) / K for h, l in zip(hi, lo)]))
    hit, tried = 0, 0
    for i in new:
        for lang in set(labels[:n_old]):
            members = [j for j in range(n_old) if labels[j] == lang]
            truth = [j for j in members if pair_ids[j] == pair_ids[i]]
            if not truth:
                continue
            nearest = members[int(np.argmin(((y_all[members] - y_all[i]) ** 2).sum(1)))]
            hit += nearest == truth[0]
            tried += 1
    radius = np.sqrt(((y_old_before - y_old_before.mean(0)) ** 2).sum(1).mean())
    shift = float(np.linalg.norm(y_old_after - y_old_before, axis=1).mean() / radius)
    is_new = np.zeros(len(x_all), dtype=bool)
    is_new[new] = True
    return {
        "old shift": shift,
        "kNN overlap": overlap,
        "translation": hit / max(tried, 1),
        "new-new 2D": float(is_new[lo].mean()),
        "new-new full": float(is_new[hi].mean()),
    }


# ── parametric UMAP ─────────────────────────────────────────────────────────

def fit_parametric_umap(x, reducer, steps=4000, batch=512, negatives=5, seed=0):
    """A small MLP trained to reproduce UMAP's layout of `x` from its fuzzy graph."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean, std = x.mean(0), x.std(0) + 1e-6
    data = torch.tensor((x - mean) / std, dtype=torch.float32)
    graph = reducer.graph_.tocoo()
    probability = graph.data / graph.data.sum()
    a, b = reducer._a, reducer._b
    net = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 2),
    )
    optimiser = torch.optim.Adam(net.parameters(), lr=1e-3)

    def similarity(p, q):
        return 1.0 / (1.0 + a * ((p - q) ** 2).sum(-1).clamp(min=1e-12) ** b)

    for _ in range(steps):
        edge = rng.choice(len(probability), size=batch, p=probability)
        head = torch.tensor(graph.row[edge], dtype=torch.long)
        tail = torch.tensor(graph.col[edge], dtype=torch.long)
        other = torch.tensor(rng.integers(0, len(x), size=(batch, negatives)), dtype=torch.long)
        zh, zt, zn = net(data[head]), net(data[tail]), net(data[other])
        attract = -torch.log(similarity(zh, zt) + 1e-4).mean()
        repel = -torch.log(1.0 - similarity(zh[:, None, :], zn) + 1e-4).sum(1).mean()
        optimiser.zero_grad()
        (attract + repel).backward()
        optimiser.step()

    def encode(points):
        with torch.no_grad():
            return net(torch.tensor((points - mean) / std, dtype=torch.float32)).numpy()

    return encode


# ── runners ─────────────────────────────────────────────────────────────────

def timed(function):
    start = time.time()
    result = function()
    return result, time.time() - start


def run_umap_family(x_old, x_new, steps):
    x_all = np.vstack([x_old, x_new])
    make = lambda: umap.UMAP(n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST, random_state=42)
    reducer, fit_time = timed(lambda: make().fit(x_old))
    base = reducer.embedding_.astype(np.float32)
    out = {}
    refit, took = timed(lambda: make().fit_transform(x_all))
    out["umap refit (all points)"] = (refit[: len(x_old)], refit[len(x_old):], took, base)
    placed, took = timed(lambda: reducer.transform(x_new))
    out["umap .transform"] = (base, placed, took, base)
    placed, took = timed(lambda: place_new_points(
        x_old, base, x_new, metric="euclidean", n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST))
    out["anchored (ours)"] = (base, placed, took, base)
    encode, train_time = timed(lambda: fit_parametric_umap(x_old, reducer, steps=steps))
    old_encoded = encode(x_old)
    placed, took = timed(lambda: encode(x_new))
    out["parametric UMAP"] = (old_encoded, placed, took, old_encoded)
    out["_parametric_train_time"] = train_time
    return out


def run_tsne_family(x_old, x_new, gammas, joint_path):
    sys.path.insert(0, str(joint_path))
    from joint_tsne import common_neighbor_ratio, find_match_edges, joint_tsne, tsne

    np.random.seed(0)
    x_all = np.vstack([x_old, x_new])
    out = {}
    base = tsne(x_old, np.array([]), no_dims=2, initial_dims=50, perplexity=30)[0]
    np.random.seed(0)
    refit, took = timed(lambda: tsne(x_all, np.array([]), no_dims=2, initial_dims=50, perplexity=30)[0])
    out["t-SNE refit (all points)"] = (refit[: len(x_old)], refit[len(x_old):], took, base)
    start, took = timed(lambda: place_new_points(
        x_old, base, x_new, metric="euclidean", n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST, n_epochs=0))
    out["t-SNE initial placement only"] = (base, start, took, base)

    n_old = len(x_old)
    edges_old = [(i, int(j)) for i, row in enumerate(knn(x_old, K)) for j in row]
    edges_all = [(i, int(j)) for i, row in enumerate(knn(x_all, K)) for j in row]
    ratios = common_neighbor_ratio(n_old, edges_old, edges_all, K)
    edges = find_match_edges(edges_old, edges_all, {i: ratios[i] for i in range(n_old)})
    values = list(edges.values())
    if values and min(values) != max(values):
        low, high = min(values), max(values)
        edges = {e: (v - low) / (high - low) for e, v in edges.items()}
    for gamma in gammas:
        joint, took = timed(lambda: joint_tsne(
            gamma, base, x_all, np.vstack([base, start]), edges, no_dims=2, initial_dims_1=50, perplexity=30))
        out[f"joint t-SNE (gamma={gamma:g})"] = (joint[:n_old], joint[n_old:], took, base)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", choices=["umap", "tsne"], required=True)
    parser.add_argument("--n", type=int, default=100, help="sentences per language")
    parser.add_argument("--held-out", nargs="+", default=HELD_OUT)
    parser.add_argument("--parametric-steps", type=int, default=4000, help="training steps of the parametric UMAP network")
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.003, 0.01, 0.03, 0.1])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--joint-tsne-path", type=Path, default=ROOT.parent / "embeddings-vis-tool" / "vendor")
    args = parser.parse_args()

    for held_out in args.held_out:
        vectors, labels, pair_ids = [], [], []
        for lang in BASE + [held_out]:
            x, pairs = load_language(lang, args.n, args.data_dir)
            vectors.append(x)
            labels += [lang] * len(x)
            pair_ids += pairs
        n_old = sum(len(v) for v in vectors[:-1])
        x_old, x_new = np.vstack(vectors[:-1]), vectors[-1]
        x_all = np.vstack([x_old, x_new])
        print(f"\n=== {args.family}: base {len(BASE)} languages ({n_old} points), held out {held_out} ({len(x_new)} points) ===")

        results = run_umap_family(x_old, x_new, args.parametric_steps) if args.family == "umap" else run_tsne_family(
            x_old, x_new, args.gammas, args.joint_tsne_path)
        extra = results.pop("_parametric_train_time", None)
        print(f"{'method':34s} {'old shift':>9s} {'kNN ovl':>8s} {'transl.':>8s} {'new-new':>8s} {'(full)':>7s} {'time s':>7s}")
        for name, row in results.items():
            y_old, y_new, took, before = row
            m = evaluate(x_all, np.vstack([y_old, y_new]), n_old, labels, pair_ids, before, y_old)
            print(f"{name:34s} {m['old shift']:9.3f} {m['kNN overlap']:8.3f} {m['translation']:8.3f} "
                  f"{m['new-new 2D']:8.3f} {m['new-new full']:7.3f} {took:7.1f}")
        if extra is not None:
            print(f"(parametric UMAP was trained once for {extra:.0f}s; the time above is only the placement)")


if __name__ == "__main__":
    main()
