# Embedding Projection

A D3/canvas viewer for exploring high-dimensional embeddings (UMAP or
t-SNE) in 2D — built around FLORES-200 parallel sentence embeddings, but
accepts any embeddings JSON.

Two processes: a static frontend (plain JS, ES modules, no build step) and
a FastAPI backend that computes projections server-side.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -c "from huggingface_hub import login; login()"
```
Then accept the terms at https://huggingface.co/datasets/facebook/flores
(skip this if you only plan to upload your own embeddings JSON).

## Run

```bash
python3 -m uvicorn server:app --port 8500   # backend
python3 -m http.server 8000                 # frontend, separate terminal
```
Open **http://localhost:8000**. First backend startup computes and caches
the default 40-language dataset (a few minutes); later startups load the
cache instantly. Customize it via env vars: `PROJECTION_LANGUAGES`,
`PROJECTION_N_SENTENCES`, `PROJECTION_MODEL`.

## Using it

- **Select a point** (click on the plot or a data-table row) to highlight
  it — the table row reveals **Show translations** / **Show tokens**.
- **Show translations** isolates its parallel sentences on the plot;
  **Show tokens** overlays that sentence's token embeddings in the same
  UMAP space (auto-switches method to UMAP if needed).
- **Method / metric** — switch UMAP ↔ t-SNE and cosine/euclidean/manhattan;
  **Refit** forces a manual re-run.
- **Label on hover** — choose what the tooltip shows (text/ID/language).
- **Filter languages** — selected languages get real colors; the rest stay
  neutral gray.
- **Load embeddings JSON** — upload your own dataset; click "?" for the
  expected format.

## Troubleshooting

- **"Can't reach the backend"** — start/check `uvicorn server:app --port 8500`.
- **Address already in use** — `lsof -ti:8000 -sTCP:LISTEN` (or `:8500`), `kill <pid>`.
- **No default dataset** — `data/flores_embeddings.json` missing; run the
  HF login step above, or upload your own file.

## Layout

```
index.html, css/style.css   page shell + styling
client-js/                  app.js orchestrates; data-table.js, projection-view.js
                             (point-layer.js, zoomable-axes.js, tooltip.js),
                             lang-filter.js, backend-client.js
server.py                   FastAPI backend
embedding_pipeline.py        FLORES-200 + XLM-R embedding generation
tests/                      pytest suite
```
