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
feature (translations, tokens) has something to show.

## Using it

- **Pin a point** — check the checkbox in a data-table row (its only pin
  control; clicking elsewhere on the row does nothing), or click it on
  the plot. Pin more than one to compare them. Use the data table's
  **Clear selection** button to clear every pin at once — clicking empty
  canvas space no longer does this, so an incidental click doesn't wipe
  your selection.
- **Select all** (data table) pins every row matching the current
  search, across all pages. Narrow with the search box first, e.g.
  `lang=deu_Latn` then Select all pins just that language.
- **Statistics / Alignment panel** (bottom) — aggregate cosine statistics
  for the pinned points, computed in the full embedding space rather than
  the 2D plot: mean cosine within vs. across languages, and for every
  language pair the mean cosine of true translations vs. unrelated
  sentences, their gap, and retrieval accuracy (how often a translation
  is the mutual nearest neighbor of its source, among the selection).
  Translations are matched by `pairId`; without it only the overall
  numbers are shown. Hide it from the ⚙ layout menu.
- Each pinned point gets its own **Actions ▾** menu (in the data table)
  — independently toggle its tokens. **Translations** are the one
  exception: only available while exactly one point is pinned. Tokens
  overlay that sentence's token embeddings in the same UMAP space
  (auto-switches method to UMAP if needed).
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
  them instead of leaving them stacked. **Style** picks the magnification
  curve underneath: **D3** is a hand-ported version of the classic
  `d3.fisheye.circular()` plugin (exponential falloff); **Geometric** is
  this project's own simpler rational-falloff curve.
- **Color by language family** (default on) — every point colored by its
  language's linguistic family, with a legend underneath the checkbox.
  Uncheck to switch to **Filter languages** instead — pick specific
  languages to get real colors, with everything else neutral gray. The
  two modes are mutually exclusive; the language search is disabled
  while family coloring is active.
- **Search the data table** — a bare query matches language or text by
  substring; prefix it with a column name to target just that column
  instead: `id=`, `lang=`, `text=`, or `pairid=`. `id=`/`pairid=` require
  an exact match (so `pairid=1` doesn't also pull in "10", "11", ...);
  `lang=`/`text=` still match by substring. Ephemeral — clears whenever
  a new dataset loads, not part of the shareable link.
- **Resize the data table** — drag the thin handle between the plot and
  the table; the split is remembered across reloads.
- **Load embeddings JSON** — upload your own dataset; click "?" for the
  expected format. **Reset to default dataset** discards it and reloads
  FLORES-200 without restarting the server — reloading the full default
  dataset from its cache file can take a while (it's several GB), so this
  can take up to a minute or two.

## Language families

"Color by language family" looks each language up in
`resources/language_families.json`, generated from
[Glottolog](https://github.com/glottolog/glottolog-cldf) (CC BY 4.0) and
keyed by ISO 639-3 code, so any FLORES-style code (`deu_Latn` → `deu`) is
covered. ISO 639-1 codes (`de`) and English names (`German`) also resolve,
via aliases built from the SIL ISO 639-3 table. Indo-European is split into its main branches (Germanic,
Romance, Slavic, ...); other families use Glottolog's top level. To
refresh the file: `python3 scripts/build_language_families.py`.

For your own language codes, add a `lang_families` map (or a per-point
`family`) to the uploaded JSON — see the "?" next to *Load embeddings
JSON*. Anything unknown is grouped as "Other".

## Troubleshooting

- **"Can't reach the backend"** — start/check `uvicorn server:app --port 8500`.
- **Address already in use** — `lsof -ti:8000 -sTCP:LISTEN` (or `:8500`), `kill <pid>`.
- **No default dataset** — `data/flores_embeddings.json` missing; run the
  HF login step above, or upload your own file.
