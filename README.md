# Embedding Projection

A D3/canvas viewer for exploring high-dimensional embeddings (UMAP or
t-SNE) in 2D or 3D — built around FLORES-200 parallel sentence embeddings, but
accepts any embeddings JSON.

A single FastAPI process computes projections server-side and also serves
the static frontend (plain JS, ES modules; TODO: use TS and add a build
step and package management).

## Setup & run

```bash
make run     # creates .venv, installs requirements (only when they change), starts the app
make test    # runs the test suite
```
It opens your browser at **http://localhost:8500** once the server is up
(`make run PORT=9000` to change the port, `make run OPEN=0` to skip the browser).
Requires Python 3.9+.

For FLORES-200 loading, do the one-time Hugging Face setup (skip it if you
only upload your own embeddings JSON): `.venv/bin/python -c "from huggingface_hub import login; login()"`
(or set `HF_TOKEN`), then accept the terms at
https://huggingface.co/datasets/facebook/flores.

The app starts **empty** (nothing is embedded at startup, so it uses no
memory until you ask): either upload
an embeddings JSON, or expand **Or embed FLORES-200 sentences** in the
sidebar and choose the FLORES-200 language codes, sentences per language
(1–1012), model and pooling (`mean` | `max` | `cls` — how token vectors
combine into a sentence vector). Memory and time grow with languages ×
sentences, so start small (the default is 5 languages × 100 sentences).
Each language is cached under `data/`, as many sentences as have been
embedded so far, so a load only embeds what is new: a new language adds a
file and more sentences extend the existing ones. **Cancel** (or Esc) in the
loading overlay stops a running embedding job; the current dataset is kept.

### Editing the loaded dataset

Once a FLORES-200 dataset is loaded, the same form has two modes (the
buttons at its top):

- **Edit loaded** (the default) — add or remove languages, or change the
  number of sentences, and press **Update dataset**. Only what is new is
  embedded, and it is *placed into the existing UMAP layout* instead of
  recomputing it, so nothing already on the plot moves and your selection,
  pins, filters and colours stay. The model and pooling are locked here,
  since other embeddings would need a new projection. A line above the
  button shows what the update will change.
- **Load new** — embed a different selection, model or pooling from
  scratch and replace the dataset (and recompute the projection). An
  uploaded file can only be replaced this way.

New points are placed by their nearest already-fitted neighbours and then
refined with UMAP's layout optimiser with the fitted points held fixed, so a
new language lands among the languages it resembles in the original space
(the tokens of such a sentence are placed the same way, so they land next to it).
Treat their position as an approximation: they were not part of the fit
(they have a **dark outline** and an "overlaid" hover label), so distances
between them and fitted points say less than distances within the fitted
set. A note under the status line says how many points were overlaid; **Refit**
there (or in *Method parameters*) fits UMAP on all points together again,
which moves everything. Changing the method, metric, 2D/3D or the parameters
also refits. t-SNE has no out-of-sample step, so it always refits.

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
  numbers are shown. Collapse it with its ▾ button.
- Each pinned point gets its own **Actions ▾** menu (in the data table)
  — independently toggle its tokens, or **Select translations** to pin
  every point sharing its `pairId` and isolate the plot to them. Tokens
  overlay that sentence's token embeddings in the same UMAP space
  (auto-switches method to UMAP if needed).
- **Show selected** (data table) filters the table down to just the
  pinned rows *and* isolates the plot to just those points. Also disables the fisheye lens (a lens over an isolated,
  already-decluttered view isn't useful) until it's turned back off.
- **Method / metric** — switch UMAP ↔ t-SNE and cosine/euclidean/manhattan;
  only the selected method's parameters are shown (UMAP: neighbors,
  min_dist; t-SNE: perplexity, learning_rate). **Refit** re-runs with the
  edited values (and, after an edit, fits all points together).
- **Label on hover** — choose what the tooltip shows (text/ID/language/
  language family).
- **Fisheye lens** — check the box and a blue disk with a visible outline
  appears on the plot; drag it by its centre grip (✥) or its rim and drop
  it where you want to magnify points. The interior stays free, so you
  can still pan and hover under it. **Lens size** sets its radius. The lens stays fixed on screen while
  you pan/zoom. It magnifies points
  inside it (a circular lens, not a graph-topology-aware distortion)
  — pulls apart overlapping points in a dense cluster so you can click an
  individual one; nearby points also push each other apart, not just
  away from the lens centre, so magnifying a tight cluster actually separates
  them instead of leaving them stacked. **Style** picks the magnification
  curve underneath: **D3** is a hand-ported version of the classic
  `d3.fisheye.circular()` plugin (exponential falloff); **Geometric** is
  this project's own simpler rational-falloff curve.
- **Color points by** — **Language** (default; one hue per language, with a
  legend), **Language family**, or **Chosen languages only**, which shows a
  picker: only the languages you add get a color, everything else is
  neutral gray. The modes are mutually exclusive.
- **Select an area** — right-drag (or Ctrl+drag, for trackpads) on the plot
  to draw a box; every visible point inside it is added to the selection.
- **3D view** — re-projects into three dimensions. Drag to rotate, Shift+drag
  to pan, scroll to zoom; **Reset view** restores the default angle. Nearer
  points are drawn larger and more opaque. Selection, colours, the fisheye
  lens, box-select and tokens all work in 3D; switching 2D ↔ 3D keeps the
  selection (token overlays are dropped, as for any refit).
- **Search the data table** — a bare query matches language or text by
  substring. Pick a column in the dropdown next to the search box to
  search just that one, or prefix the query with it: `id=`, `lang=`,
  `text=`, or `pairid=`. `id=`/`pairid=` require
  an exact match (so `pairid=1` doesn't also pull in "10", "11", ...);
  `lang=`/`text=` still match by substring. Ephemeral — clears whenever
  a new dataset loads, not part of the shareable link.
- **Collapse any view** — the ▾/◂ button in each header (sidebar,
  projection, data table, statistics) folds it down to its title bar and
  the others take over the space; the state is remembered across
  reloads. ⤢ still makes one view fill the window.
- **Resize the panels** — drag the thin handle beside the sidebar, or
  above the data table and statistics; the sizes are remembered across
  reloads.
- **Load embeddings** — upload your own dataset as JSON, JSONL (one point
  per line) or CSV/TSV (`id`, `lang`, optional `text`/`pairId`/`family`, and
  either a `vector` column or one numeric column per dimension); click "?"
  for the details. Samples are in `examples/`. **Load FLORES-200** embeds the chosen subset on the
  backend (needs the Hugging Face setup above) and replaces whatever is
  loaded, without restarting the server; **Update dataset** edits it in
  place (see above).

## Language families

"Color by language family" looks each language up in
`resources/language_families.json`, generated from
[Glottolog](https://github.com/glottolog/glottolog-cldf) (CC BY 4.0) and
keyed by ISO 639-3 code, so any FLORES-style code (`deu_Latn` → `deu`) is
covered. ISO 639-1 codes (`de`) and English names (`German`) also resolve,
via aliases built from the SIL ISO 639-3 table. Indo-European is split
into its main branches (Germanic, Romance, Slavic, ...); other families
use Glottolog's top level. To refresh the file:
`python3 scripts/build_language_families.py`.

For your own language codes, add a `lang_families` map (or a per-point
`family`) to the uploaded JSON — see the "?" next to *Load embeddings
JSON*. Anything unknown is grouped as "Other".

## Development

`make test` runs the backend tests and the browser tests in `tests/e2e/`
(pytest; the latter drive the real page in headless Chromium, so run
`.venv/bin/playwright install chromium` once, or they are skipped; the
FLORES-200 ones use `tests/e2e/fake_server.py`, which swaps the model for
deterministic vectors). Frontend files are served
as-is, so a browser refresh picks up JS/CSS changes; restart `make run`
after editing `server.py` or `backend/`. To run the frontend against a backend on
another address, open the page with `?backend=http://host:port`.

## Troubleshooting

- **"Can't reach the backend"** — start/check `make run`.
- **Address already in use** — `lsof -ti:8500 -sTCP:LISTEN`, `kill <pid>`, or `make run PORT=9000`.
- **"Could not load FLORES-200"** — run the HF login step above and accept
  the dataset terms, or upload your own file.
