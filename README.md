# Embedding Projection

A D3/canvas viewer for exploring high-dimensional embeddings (UMAP or t-SNE)
in 2D — built around FLORES-200 parallel sentence embeddings, but accepts any
embeddings JSON via the file picker.

Two processes: a static frontend (plain HTML/CSS/JS, ES modules, no build
step) and a small FastAPI backend that computes projections server-side.

## 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. One-time Hugging Face setup

The backend computes the default FLORES-200 dataset itself on first
startup, but FLORES-200 requires accepting its terms once per account:

```bash
python3 -c "from huggingface_hub import login; login()"
```
then accept the terms at https://huggingface.co/datasets/facebook/flores

(Skip this if you only plan to upload your own embeddings — see step 4.)

## 3. Start the backend

```bash
python3 -m uvicorn server:app --port 8500
```
Leave this running. On first startup (or whenever the requested config
changes), it computes the default dataset — 40 languages x 200 sentences,
`xlm-roberta-base`, last layer, plus per-token vectors — and caches it to
`data/flores_embeddings.json`; later startups with the same config load
that cache instead (seconds instead of minutes). Serves `/points`,
`/project`, `/project/tokens/{id}`, and `/dataset` on port 8500.

The 40-language default takes a while on a cache miss (each language is a
separate model pass over 200 sentences). Customize via env vars, e.g. to
run with just a handful of languages instead:
```bash
PROJECTION_LANGUAGES=eng_Latn,deu_Latn,fra_Latn,spa_Latn,zho_Hans,jpn_Jpan,kor_Hang \
PROJECTION_N_SENTENCES=200 \
PROJECTION_MODEL=xlm-roberta-base \
    python3 -m uvicorn server:app --port 8500
```

## 4. Start the frontend (separate terminal)

```bash
python3 -m http.server 8000
```
Then open **http://localhost:8000** in a browser.

## Using it

- **Click a point** to isolate its parallel translations (same sentence,
  other languages) — everything else fades out. Click again, or click the
  background, to clear.
- **Scroll/drag on the plot** to zoom and pan; axes rescale live.
- **Filter languages** (sidebar) — search and select languages to give them
  real colors; everything unselected stays neutral gray. Clear the filter to
  return to the neutral, shape-only view.
- **Projection method** — switch between UMAP and t-SNE; re-runs server-side.
- **Show/hide tokens** — with a point selected, click "Show tokens" in the
  sidebar to see that sentence's individual token embeddings projected into
  the same UMAP space (auto-switches the projection method to UMAP if
  needed, since t-SNE can't share axes with a second projection). Click
  again to hide them.
- **Load embeddings JSON** — upload your own dataset instead of the default
  one; click "?" for the exact format.

## Troubleshooting

- **"Can't reach the backend"** in the status line — the backend
  (`uvicorn server:app --port 8500`) isn't running, or died. Check its
  terminal for errors.
- **`OSError: [Errno 48] Address already in use`** when starting either
  server — something is already listening on that port. Find and stop it:
  ```bash
  lsof -ti:8000 -sTCP:LISTEN   # or :8500
  kill <pid>
  ```
- **No default dataset loaded** — `data/flores_embeddings.json` doesn't
  exist yet; run step 2, or upload your own file via the UI.

## Project layout

```
index.html                          # page shell, loads client-js/app.js as a module
css/style.css                       # all styling
client-js/
  app.js                            # orchestrator: selection state, wiring
  projection-view.js                # composes the four view collaborators below
  point-layer.js                    # point data, color/opacity, canvas painting
  zoomable-axes.js                  # SVG axis overlay + d3.zoom
  tooltip.js                        # hover tooltip
  lang-filter.js                    # searchable multi-select language filter
  backend-client.js                 # fetch wrapper for server.py's endpoints
  detail-panel.js                   # sidebar detail-on-click panel
server.py                           # FastAPI backend: /points, /project, /project/tokens/{id}, /dataset
embedding_pipeline.py                # FLORES-200 + XLM-R embedding pipeline (sentence + token vectors)
tests/                               # pytest suite for embedding_pipeline.py and server.py
requirements.txt
requirements-dev.txt                # adds pytest, for running tests/
```