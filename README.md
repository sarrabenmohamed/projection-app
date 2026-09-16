# Embedding Projection

A D3/canvas viewer for exploring high-dimensional embeddings (UMAP or
t-SNE) in 2D — built around FLORES-200 parallel sentence embeddings, but
accepts any embeddings JSON.

Two processes: a static frontend (plain JS, ES modules, TODO: use TS and add build step and package management) and
a FastAPI backend that computes projections server-side.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -c "from huggingface_hub import login; login()"
```
Then accept the terms at https://huggingface.co/datasets/facebook/flores
(skip HF setup if you only plan to upload your own embeddings JSON).

## Run

```bash
python3 -m uvicorn server:app --port 8500   # backend
python3 -m http.server 8000                 # frontend, separate terminal
```
Open **http://localhost:8000**. First backend startup computes and caches
the default 40-language dataset (a few minutes); later startups load the
cache instantly. Customize it via env vars: `PROJECTION_LANGUAGES`,
`PROJECTION_N_SENTENCES`, `PROJECTION_MODEL`, `PROJECTION_POOLING`
(`mean` | `max` | `cls` — how token vectors combine into a sentence vector).

## Demo

No HuggingFace access or GPU needed to try the app: open it, click "Load
embeddings JSON", and pick `examples/sample_embeddings.json` — 9 toy
sentences (3 languages × 3 parallel translations, random but
language-clustered 8-dim vectors) with token vectors included, so every
feature (translations, tokens, neighbors) has something to show.

## Using it

- **Pin a point** — check the checkbox in a data-table row (its only pin
  control; clicking elsewhere on the row does nothing), or click it on
  the plot. Pin more than one to compare them. Click empty canvas space
  to clear every pin at once.
- Each pinned point gets its own **Actions ▾** menu (in the data table)
  — independently toggle its tokens and nearest neighbors.
  **Translations** are the one exception: only available while exactly
  one point is pinned. Tokens overlay that sentence's token embeddings
  in the same UMAP space (auto-switches method to UMAP if needed);
  neighbors lists the *k* nearest points by full-vector distance (id,
  text, distance), with *k* set inline in the menu.
- **Show selected** (data table) filters the table down to just the
  pinned rows *and* isolates the plot to just those points — takes
  precedence over an active translations view rather than combining
  with it. Also disables the fisheye lens (a lens over an isolated,
  already-decluttered view isn't useful) until it's turned back off.
- **Method / metric** — switch UMAP ↔ t-SNE and cosine/euclidean/manhattan;
  **Refit** forces a manual re-run.
- **Label on hover** — choose what the tooltip shows (text/ID/language/
  language family).
- **Fisheye lens** — check the box, then hover the plot to magnify points
  near the cursor (a circular lens, not a graph-topology-aware distortion)
  — pulls apart overlapping points in a dense cluster so you can click an
  individual one; nearby points also push each other apart, not just
  away from the cursor, so magnifying a tight cluster actually separates
  them instead of leaving them stacked.
- **Color by language family** (default on) — every point colored by its
  language's linguistic family, with a legend underneath the checkbox.
  Uncheck to switch to **Filter languages** instead — pick specific
  languages to get real colors, with everything else neutral gray. The
  two modes are mutually exclusive; the language search is disabled
  while family coloring is active.
- **Resize the data table** — drag the thin handle between the plot and
  the table; the split is remembered across reloads.
- **Load embeddings JSON** — upload your own dataset; click "?" for the
  expected format.
- **Copy link** — copies a URL with the current method/metric/label and
  pinned point ids as query params; opening it re-pins the same points
  (translations/tokens/neighbors toggles are not restored — every
  re-pinned point starts with those off).

## Troubleshooting

- **"Can't reach the backend"** — start/check `uvicorn server:app --port 8500`.
- **Address already in use** — `lsof -ti:8000 -sTCP:LISTEN` (or `:8500`), `kill <pid>`.
- **No default dataset** — `data/flores_embeddings.json` missing; run the
  HF login step above, or upload your own file.
