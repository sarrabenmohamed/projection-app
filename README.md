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

## 2. Get a dataset

**Option A — generate the default FLORES-200 dataset:**

One-time setup: log in to Hugging Face and accept the FLORES-200 dataset
terms (required once per account):

```bash
python3 -c "from huggingface_hub import login; login()"
```
then accept the terms at https://huggingface.co/datasets/facebook/flores

Then run the prep script (defaults to 5 languages × 200 sentences,
`xlm-roberta-base`, last layer — takes a few minutes):

```bash
python3 scripts/prepare_flores_embeddings.py
```

Customize with flags, e.g. more languages:
```bash
python3 scripts/prepare_flores_embeddings.py \
    --languages eng_Latn,deu_Latn,fra_Latn,spa_Latn,zho_Hans,jpn_Jpan,kor_Hang \
    --n-sentences 200 \
    --model xlm-roberta-base \
    --output data/flores_embeddings.json
```
More languages = proportionally longer (each language is a separate model
pass over `--n-sentences` sentences). This writes to `data/flores_embeddings.json`,
which `server.py` loads automatically on startup.

**Option B — skip this and upload your own embeddings** once the app is
running (see step 4). Click the "?" next to "Load embeddings JSON" in the
sidebar for the exact JSON format it expects.

## 3. Start the backend

```bash
python3 -m uvicorn server:app --port 8500
```
Leave this running. It loads `data/flores_embeddings.json` on startup (if
present) and serves `/points`, `/project`, and `/dataset` on port 8500.

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
server.py                           # FastAPI backend: /points, /project, /dataset
scripts/prepare_flores_embeddings.py  # FLORES-200 + XLM-R embedding prep pipeline
requirements.txt
```