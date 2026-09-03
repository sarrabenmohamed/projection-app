# Token embeddings + backend/script merge — design

## Goal

Two changes, designed together because they share the same data pipeline:

1. Fold `scripts/prepare_flores_embeddings.py` into `server.py` so the
   backend computes (or loads a cache of) the default dataset itself on
   startup, instead of requiring a manual script run first.
2. Save per-token embeddings alongside each sentence's pooled embedding, and
   let the frontend show a selected point's own tokens projected into the
   *same* 2D axes as the sentence scatter, toggleable independently of the
   existing "isolate translations" click behavior.

## 1. Architecture: merging the script into the backend

`scripts/prepare_flores_embeddings.py` is deleted. Its logic (FLORES-200
loading, XLM-R tokenize + forward pass, mean-pooling) moves into a new
module, `embedding_pipeline.py`, imported by `server.py`.

On startup, `server.py`:

1. Reads config from env vars: `PROJECTION_LANGUAGES` (default
   `eng_Latn,deu_Latn,fra_Latn,spa_Latn,zho_Hans`), `PROJECTION_N_SENTENCES`
   (default `200`), `PROJECTION_MODEL` (default `xlm-roberta-base`) — same
   defaults the script's CLI flags had.
2. Checks the cache file (`data/flores_embeddings.json`, same path as
   today). If it exists and its stored `model`, `lang_codes`, and per-language
   sentence count match the requested config, loads it directly (fast path).
3. Otherwise calls `embedding_pipeline.build_dataset(...)`, which runs the
   full pipeline and returns the same JSON-shaped dict `prepare_flores_embeddings.py`
   used to write to disk (now including `tokenVectors`, see below). `server.py`
   writes this to the cache path, then loads it.

This keeps `/points`, `/project`, `POST /dataset` byte-compatible with today
— only the *provenance* of the default dataset changes. `requirements.txt`
merges into one list (torch, transformers, datasets, huggingface_hub added
to the existing fastapi/umap-learn/scikit-learn/numpy set), since it's one
process now.

`README.md`'s setup steps collapse: no more "run the prep script, then
start the backend" — just start the backend; first startup with no cache
takes a few minutes (same cost as running the script did today), subsequent
startups are fast.

## 2. Data model: token vectors

Extend the point schema with an optional field:

```json
{
  "id": "eng_Latn-0",
  "lang": "eng_Latn",
  "pairId": 0,
  "text": "The cat sat.",
  "vector": [0.37, 0.23, "... 768 floats"],
  "tokenVectors": [
    {"token": "▁The", "vector": [0.30, 0.10, "... 768 floats"]},
    {"token": "▁cat", "vector": [0.90, 0.20, "... 768 floats"]},
    {"token": "▁sat", "vector": [0.70, -0.10, "... 768 floats"]},
    {"token": ".",        "vector": [0.00, 0.30, "... 768 floats"]}
  ]
}
```

- `<s>`/`</s>` special tokens are excluded from `tokenVectors` (not useful to
  visualize) but still included in the mean-pool that produces `vector`,
  unchanged from today.
- `tokenVectors` is optional everywhere. `validate_and_normalize` in
  `server.py` accepts it if present and validates each entry's vector length
  against `dim`; datasets (uploaded or backend-generated) that omit it keep
  working exactly as today — token projection is simply unavailable (400)
  for those points.
- `embedding_pipeline.py`'s FLORES pipeline always populates it, so the
  default dataset gets token-level data automatically once regenerated.
- Uploaded JSON via `POST /dataset` may optionally include `tokenVectors`
  per point using the same shape; this is the general contract, not
  FLORES-specific.

Storage cost: for ~1000 sentences (5 languages x 200) this raises the
default dataset JSON from a few MB to tens of MB (short sentences, ~10-30
tokens each). Fine for local dev; no change needed to how it's loaded/served.

## 3. Shared 2D space for token overlay

Token-level and sentence-level projections come from unrelated input sets
(a whole corpus of 768-dim sentence vectors vs. a handful of token vectors
for one sentence). Projecting them independently would place them
arbitrarily relative to each other on the same axes. To make them
comparable, the backend fits **one UMAP reducer** on the corpus of sentence
vectors and reuses it (via `.transform()`, not a fresh fit) to embed token
vectors into that same fixed space — this is a form of the
"shared reducer, fit once, transform many" pattern used in `layer_explorer.py`'s
`shared_parametric_umap` mode, but using UMAP's built-in (non-parametric)
`.transform()` support rather than `ParametricUMAP`. `ParametricUMAP`
requires a real TensorFlow/Keras backend; `server.py` already stubs
`sys.modules["tensorflow"]` (line 37) specifically because a real import
hangs on hardware this app has been run on, and reintroducing a working TF
dependency would undo that workaround for no benefit at this dataset size
(~1000 points).

Concretely:

- `server.py` keeps a fitted UMAP reducer in memory next to `DATASET`,
  keyed to the current dataset. It's built lazily the first time a UMAP
  projection or a token projection is requested, and invalidated (cleared)
  whenever the dataset changes (`POST /dataset`, or a startup recompute).
- `GET /project?method=umap` uses (fitting-if-missing, then reusing) this
  cached reducer instead of calling `fit_transform` fresh every request.
  Output is unchanged (still `random_state=42`, same numbers) — this is a
  caching change, not a behavior change, and it's what makes the reducer
  available for token transforms.
- `method=tsne` is untouched: still fits fresh per call, and cannot back
  token overlay (sklearn's `TSNE` has no out-of-sample `transform`).

## 4. New endpoint

```
GET /project/tokens/{id}
```

- 404 if `id` isn't in the current dataset.
- 400 if the point has no `tokenVectors`, or fewer than 2 tokens (mirrors
  the existing "need >= 2 points" guard in `/project`).
- Otherwise: ensure the shared UMAP reducer is fitted (fit lazily on the
  full sentence-vector corpus if not already), then
  `reducer.transform(token_vectors)`.
- Response:
  ```json
  {"id": "eng_Latn-0", "tokens": [
    {"token": "▁The", "x": 1.2, "y": -0.3},
    {"token": "▁cat", "x": 1.5, "y": 0.1}
  ]}
  ```
- No `method` query param — the shared space is always UMAP-based. If the
  frontend's main view is currently `tsne`, it switches the method to
  `umap` (see below) before calling this endpoint, so the request is always
  satisfiable.

`_reduce` (the umap/tsne branching logic currently inlined in `project()`)
is pulled into a small shared helper used by both `/project` and the
reducer-fit path behind `/project/tokens/{id}`, since both need the same
`n_neighbors = min(15, n-1)`-style clamping logic.

## 5. Frontend wiring

Both interactions are per-point and coexist — clicking a point still
isolates its translations in the main view exactly as today; token display
is a separate, independently toggleable action on the selected point.

- **`backend-client.js`**: add `fetchTokenProjection(id)` -> `GET
  /project/tokens/{id}`.
- **`point-layer.js`**: add `setTokenOverlay(tokenPoints)` /
  `clearTokenOverlay()`. Overlay points are drawn in the same `draw()` pass
  on the same `baseXScale`/`baseYScale` (they're in the same coordinate
  space now, by construction), styled distinctly from sentence points (e.g.
  small diamonds vs. circles) so they read as "this point's tokens," always
  at full opacity — independent of the isolate-fade animation applied to
  sentence points.
- **`detail-panel.js`**: add a "Show tokens" / "Hide tokens" toggle,
  enabled once a point is selected. Disabled (or hidden) if the currently
  selected point has no token data, per the `hasTokens` flag below.
- **`server.py`**: `GET /points` gains a per-point `hasTokens: bool` field
  (`p.get("tokenVectors") is not None`) alongside the existing `id`/`lang`/
  `pairId`/`text`, so the frontend can disable the toggle upfront rather
  than discovering via a failed request.
- **`app.js`**:
  - Tracks at most one overlay at a time: `this._tokenOverlayPointId`.
    Toggling on a different point replaces the previous overlay; toggling
    the same point off clears it; selecting a new point (click elsewhere)
    also clears any active overlay.
  - On toggle-on: if `method-select`'s value isn't `umap`, set it to `umap`
    and await `_refresh()` first (status line notes the switch, e.g. "switched
    to UMAP to show tokens"), then `backend.fetchTokenProjection(id)`, map
    each `{token, x, y}` to `{id: "<pointId>::<i>", text: token, x, y}`, call
    `view.setTokenOverlay(...)`.
  - On toggle-off: `view.clearTokenOverlay()`.
- **`tooltip.js`**: no changes needed if it already keys off a point's
  `text` field for hover display; token overlay points reuse that field for
  their token string.

## Testing

- `embedding_pipeline.py`: unit-test `_mean_pool`, unit-test that
  `tokenVectors` excludes special tokens and has the same `dim` as `vector`.
- `server.py`: test cache hit/miss logic (matching vs. mismatched
  `model`/`lang_codes`/count triggers reuse vs. rebuild); test
  `validate_and_normalize` accepts datasets with and without `tokenVectors`,
  and rejects malformed ones (wrong length, non-finite values); test
  `/project/tokens/{id}` 404/400/success paths; test that repeated
  `/project?method=umap` calls reuse the cached reducer (e.g. via a call
  counter/mock) while `/dataset` invalidates it.
- Frontend: manual verification (documented in README/PR) — toggle tokens
  on a point, confirm they render near their sentence's language cluster
  and clear correctly on toggle-off/point-switch; confirm auto-switch from
  t-SNE to UMAP works.

## Out of scope

- Configurable overlay marker styling beyond a sensible default.
- Showing multiple points' token overlays simultaneously.
- Persisting the fitted UMAP reducer to disk across server restarts (refit
  lazily in memory is fast enough at this dataset size).
