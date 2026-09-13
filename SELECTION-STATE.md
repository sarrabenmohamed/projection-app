# Selection state machine

Reference for how point selection, translations-isolate, and the token
overlay interact — all owned by `App` (`client-js/app.js`) and mirrored
into `DataTable`/`ProjectionView`. Written down once so future features
(multi-select, pinning, shareable-URL state) have a known-correct baseline
to extend rather than reverse-engineer.

## State

Four pieces of state, all on `App`:

- `_selectedPoint` — the one selected sentence (or `null`). Mirrored as
  `ProjectionView`'s `_selectedId` (canvas ring) and `DataTable`'s
  `_selectedId`/`_selectedSentenceId` (row highlight + which row shows
  inline actions).
- `_translationsShown` — whether the selected point's translation group is
  currently isolated (canvas fade) and table-filtered.
- `_tokenOverlayPointId` — the point (if any) whose tokens are currently
  overlaid on canvas and inserted as rows in the table. Always either
  `null` or equal to `_selectedPoint.id` — there is no way to have tokens
  showing for a point that isn't also the current selection.
- `_neighborsPointId` — the point (if any) whose nearest neighbors (by
  full-vector distance, see `server.py`'s `/neighbors`) are currently
  listed as extra rows in the table. Same one-point constraint as tokens:
  always either `null` or equal to `_selectedPoint.id`. Unlike tokens,
  neighbors have no canvas overlay — table rows only — and don't force a
  UMAP switch, since they're computed on the raw embedding vectors rather
  than a 2D projection.

At most one point is selected at a time, and at most one point's
translations/tokens/neighbors are shown at a time (always the *same*
point — you cannot have point A's translations isolated while point B's
tokens or neighbors are shown).

All of the above (plus the method/metric/label-by dropdowns) round-trips
through the URL query string via `client-js/url-state.js`'s
`readUrlState`/`writeUrlState` — `App._syncUrl()` is called at the end of
every state-changing action so a "Copy link" click always hands off
exactly what's currently showing. This only encodes a *single* selected
point; see "Future work" below for why it doesn't extend to multiple.

## Transitions

**Selecting a sentence** (`onCanvasClickPoint` from a canvas click, or
`onSelectRow` from a table row click, both routed through
`_selectSentence`):
- Clicking a *different* point than the current selection: clears the
  previous point's toggles first (`_clearSelectionSideEffects` —
  un-isolates, un-overlays), then selects the new point with nothing
  toggled on. Toggles never carry over between points.
- Clicking the *already-selected* point again: equivalent to a background
  click (see below) — fully clears the selection.
- Clicking a **token row** (`onSelectRow` with `row.type === 'token'`):
  routes to `table.highlightRow(id)` instead — highlights that row only,
  does **not** touch `_selectedPoint`/toggles, does not move the canvas
  ring. This is the one interaction that never clears anything.

**Clearing selection** (`onBackgroundClick` — canvas background click, or
re-clicking the selected point): clears both toggles, clears
`_selectedPoint`, clears the canvas ring, clears the table's row
highlight/actions.

**Show/Hide translations** (`onToggleTranslations`, only reachable via the
selected row's inline button): toggles `_translationsShown`; isolates/fades
the canvas and switches the table to a group-filtered view, or reverses
both.

**Show/Hide tokens** (`onToggleTokens`, inline button, only present if the
point `hasTokens`): toggles `_tokenOverlayPointId`. If the current
projection method isn't UMAP, this first flips the method dropdown to
`umap` and calls `_refresh()` — which, like any `_refresh()`, wipes
*all* selection state — and then explicitly re-selects the same point
(`_selectSentence(reselected)`) before fetching its tokens. This is the
**only** path where a `_refresh()` is followed by automatic
re-selection; every other refresh trigger below just resets to nothing
selected.

**Show/Hide neighbors** (`onToggleNeighbors`, inline button, always
present on the selected row): toggles `_neighborsPointId`. No method
switch is needed — `/neighbors` operates on the full embedding vectors
regardless of what's plotted — so this never triggers a `_refresh()`.

**Anything that calls `_refresh()`** — changing the method or metric
dropdown, clicking **Refit**, or uploading a new dataset: full reset.
Selection, both toggles, canvas ring, and table highlight all clear; nothing
is restored afterward (unlike the token-overlay auto-switch case above).

**Independent of selection entirely** — these never read or modify
`_selectedPoint`/toggles:
- The **label-on-hover** dropdown (`view.setLabelBy`) — only changes what
  the tooltip shows.
- The **language filter** (`view.setLangFilter`) — only changes point
  colors/opacity.
- Table **search, sort, and pagination** — if the selected/actioned row
  scrolls out of the current filtered/sorted page, its inline actions
  simply aren't rendered until you navigate back to it; the underlying
  state (`_selectedPoint`, `_translationsShown`, etc.) is untouched and
  the row picks its rendering back up correctly once visible again.

## Known asymmetry (intentional)

A canvas click can only ever select a **sentence** — token-overlay points
are hoverable but not clickable on canvas (`ProjectionView._onCanvasClick`
checks `_overlayPointAt` first and returns early). Selecting a *token* is
only possible from the table. This is why `onCanvasClickPoint` never
checks `row.type` the way `onSelectRow` does.

## Future work: pin / multi-select (parked)

The TODO list asks for pinning several points at once, each with its own
independent translations/tokens/neighbors — a real extension of the
single-`_selectedPoint` model above, not a bolt-on. Brainstormed but
deliberately not implemented yet, because it exposed a concrete blocker
worth recording rather than working around silently:

`PointLayer.highlightGroup()` (what "Show translations" calls) doesn't
highlight the translation group — it **isolates** it, fading every other
point's opacity to `ISOLATED_OUT_OPACITY` (0, fully invisible). That's a
single exclusive piece of state (`_activeGroupKey`). If two different
pinned points each called "Show translations" independently, they'd both
try to be the one thing left visible — there's no way to render two
simultaneous "hide everything except my group" states at once.

Making per-pin translations composable means changing that mechanic from
isolate-to-invisible to dim-and-highlight (points in *any* active
translation group render at full opacity with a highlight; everything
else dims like the language filter's `FILTERED_OUT_OPACITY`, rather than
vanishing) — a visible behavior change to the existing single-selection
case too, not just new code for the multi-select case. That trade-off
needs a real decision before implementing, so pin/multi-select is parked
until then rather than forced through.
