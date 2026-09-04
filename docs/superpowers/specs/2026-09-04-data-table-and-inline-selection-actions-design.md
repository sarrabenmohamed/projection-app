# Data table + inline selection actions — design

## Goal

Add a data table view of the embedding dataset (sentence and token rows,
side by side with their attributes) and replace the current "click a point
to auto-isolate its translations, plus a separate always-visible sidebar
tokens-toggle" interaction with a single, linked selection model driven
from the table: selecting a sentence reveals inline row actions to
show/hide its translations and show/hide its tokens.

Reference: `~/src/embeddings-vis-tool/prototype-lit.html`, a LIT
(Language Interpretability Tool)-style HTML prototype the user built
earlier for a related research project. Its "Data Table" module (checkbox
column, sortable Idx/Language/Sentence/Tok columns, search box, row
count, click-to-select) is the structural reference for this app's table.
That prototype has no floating/inline selection-action UI of its own —
the inline-row-actions interaction here is new, designed in this
conversation, not copied from the prototype.

## 1. Layout

Three panes, left to right, all in the existing `#app` flex container:

1. **Sidebar** (existing, ~260px, unchanged): "Load embeddings JSON" file
   picker, projection method dropdown, language filter
   (`#load-controls`, `#lang-filter-controls`). The old detail panel
   (`#detail-placeholder`, `#detail-lang`, `#detail-tokens-toggle`) is
   removed from here — fully superseded by the table's inline actions
   (section 3).
2. **Data table** (new): the embeddings table described below.
3. **Plot** (existing, unchanged): the canvas scatter plot, axes,
   tooltip, and token-overlay diamonds.

## 2. Data table

### Row types and columns

| Column | Sentence row | Token row |
|---|---|---|
| ID | point id (e.g. `eng_Latn-0`) | synthetic id (e.g. `eng_Latn-0::2`) |
| Language | `lang` | inherited from parent sentence |
| Text | sentence text | the token string |
| PairId | `pairId` (or blank if none) | blank |
| Type | `sentence` | `token` |
| Token strings | comma-joined list of that sentence's token strings (blank if `hasTokens` is false) | blank |

- **Sentence rows**: all of them, always — up to 8000 for the default
  dataset, from `GET /points`.
- **Token rows**: only for whichever sentence currently has its tokens
  shown (see section 3) — inserted directly under their parent row,
  visually indented/marked, and removed when tokens are hidden. Built
  entirely from data already fetched for the canvas overlay
  (`GET /project/tokens/{id}`) plus the parent's `lang` — no new backend
  call needed for token rows themselves. Token rows are pinned directly
  beneath their parent regardless of the active sort or search filter —
  they are never independently sorted into the row list, and a search
  that doesn't match the parent hides its token rows too.
- No raw vector column — decided out of scope (a 768-dim vector isn't
  meaningful as a table cell, and the frontend doesn't otherwise fetch
  raw vectors).

### Search, sort, pagination

- A single search box filters by language or text substring, applied
  before sorting/pagination.
- Column headers sort the *currently filtered* row set (sentence and any
  visible token rows together).
- Pagination is a fixed page size (50 rows) with Prev/Next controls and
  a page indicator — no infinite scroll, no virtual-scroll library.

### Backend change

`GET /points` gains one additive field per point:

```json
{
  "id": "eng_Latn-0", "lang": "eng_Latn", "pairId": 0, "text": "The cat sat.",
  "hasTokens": true,
  "tokenTexts": ["▁The", "▁cat", "▁sat"]
}
```

`tokenTexts` is `null` when the point has no `tokenVectors` (mirroring
`hasTokens`, kept alongside it rather than replacing it — both fields
stay simple booleans/arrays, no client-side derivation needed). This is
string-only — no vectors travel over this endpoint, consistent with its
existing "metadata only" contract.

## 3. Selection model

Both the table and the canvas plot read from and write to one selection
state owned by `App` — neither view has its own independent notion of
"what's selected."

- **Selecting a sentence row** (table row click, or canvas point click):
  - Highlights the point on canvas with a persistent selection ring
    (visually distinct from the existing hover-stroke treatment and from
    isolate-fade — selecting no longer fades anything by itself).
  - Highlights the row in the table. If the selection came from a canvas
    click, the table jumps to whichever page contains that row (computed
    from its index in the table's current filtered/sorted row list).
  - Reveals two inline icon-buttons in that row: a **translations
    toggle** and a **tokens toggle** — the tokens toggle is omitted
    entirely if `hasTokens` is false.
- **Selecting a token row**: highlights it in the table and (if visible)
  on the canvas overlay. No action icons — a token has no translations
  and can't be split into further tokens.
- **Translations toggle** (on a selected sentence row):
  - On: isolates/fades the canvas exactly as today's `highlightGroup`
    does, AND switches the table to show only that translation group's
    rows (a pairId-linked dataset can span far-apart pages once sorted
    across 40 languages — filtering to the group is more useful than a
    highlight the user can't see).
  - Off: clears the canvas fade and restores the full paginated table
    view.
- **Tokens toggle** (on a selected sentence row):
  - On: fetches `GET /project/tokens/{id}` (unchanged), overlays diamonds
    on canvas (unchanged), and inserts that sentence's token rows into
    the table directly beneath it.
  - Off: removes the overlay and the token rows.
- **Clearing selection** (canvas background click, clicking the
  already-selected point again, or selecting a different sentence):
  clears the selection ring, closes both toggles (un-isolating and
  un-overlaying as needed), and returns the table to its normal
  full/paginated view.
- Only one sentence's translations-group-filter and one sentence's token
  rows can be active at a time (selecting a new point clears the
  previous one's toggles first) — same "one at a time" constraint as the
  existing token-overlay feature.

## 4. File structure

**New: `client-js/data-table.js`**
Owns the table completely: the merged row list (sentence rows + any
inserted token rows), search/sort/pagination state and rendering, row
selection highlighting, and the inline action icons. Public interface:
- `setPoints(points)` — sets the full sentence dataset (called whenever
  `App._loadDataset` runs).
- `selectRow(id)` — highlights a row and pages to it (called when
  selection originates from a canvas click).
- `showTranslationsFor(groupKey, rows)` / `hideTranslations()` — switch
  into/out of the group-filtered view.
- `showTokensFor(pointId, tokenRows)` / `hideTokensFor(pointId)` —
  insert/remove child token rows for one point.
- `clearSelection()` — drop row highlight and any active toggle view.
- Callbacks (passed in the constructor, like the existing
  `LangFilter`/`ProjectionView` pattern): `onRowClick(row)`,
  `onToggleTranslations(row)`, `onToggleTokens(row)`.

**Removed: `client-js/detail-panel.js`**
Fully superseded by the table's inline actions. Its DOM elements
(`#detail-placeholder`, `#detail-lang`, `#detail-tokens-toggle`) are
removed from `index.html`; its CSS rules removed from `css/style.css`.

**Modified:**
- `server.py`: `GET /points` gains `tokenTexts` (section 2).
- `client-js/point-layer.js`: add a `_selectedId` concept — a persistent
  ring around the selected point, independent of the existing hover
  treatment and of the isolate-fade group.
- `client-js/projection-view.js`: add `setSelectedId(id)` /
  `clearSelectedId()` passthrough to `PointLayer`. Canvas click now
  reports the clicked point to `App` without any longer calling
  `highlightGroup` itself — isolation is driven by the toggle, not the
  click.
- `client-js/app.js`: owns the unified selection state
  (`_selectedPoint`, `_translationsShown`, `_tokenOverlayPointId`) and
  wires table row clicks and canvas point clicks to the same handler;
  wires the table's `onToggleTranslations`/`onToggleTokens` callbacks to
  the existing `view.highlightGroup`/`clear` and
  `view.setTokenOverlay`/`clearTokenOverlay` methods (reused as-is —
  their behavior doesn't change, only what triggers them does).
- `index.html`: new `<div id="data-table">` pane between the sidebar and
  `#main`; old detail-panel divs removed.
- `css/style.css`: new rules for the data-table pane (layout, row
  styling, type badges, inline action icons, pagination controls); old
  `#detail-lang`/`#detail-placeholder`/`#detail-tokens-toggle` rules
  removed.

No changes to `client-js/backend-client.js` beyond none needed — the
`tokenTexts` field rides along on the existing `fetchPoints()` call.

## 5. Testing

- Backend: a test for `tokenTexts` in `GET /points`, mirroring the
  existing `hasTokens` test (present when `tokenVectors` exists, `null`
  otherwise).
- Frontend: no test framework exists in this repo (pre-existing
  constraint, unchanged by this feature) — manual verification only,
  consistent with how the prior token-overlay feature was verified
  (structural checks + hand-traced logic during implementation; a real
  browser check by the user or controller afterward).

## Out of scope

- A raw vector column (decided out during brainstorming).
- Multi-sentence simultaneous translations-filter or token-overlay (the
  existing "one at a time" constraint carries over unchanged).
- Column customization/CSV export/the "Alignment" and advanced
  projection-hyperparameter modules from the LIT prototype — this spec
  is scoped to the data table and its selection actions only.
