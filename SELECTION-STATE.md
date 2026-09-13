# Selection state machine

Reference for how point selection, translations-isolate, and the token
overlay interact — all owned by `App` (`client-js/app.js`) and mirrored
into `DataTable`/`ProjectionView`. Written down once so future features
(multi-select, pinning, shareable-URL state) have a known-correct baseline
to extend rather than reverse-engineer.

## State

Three pieces of state, all on `App`:

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

At most one point is selected at a time, and at most one point's
translations/tokens are shown at a time (always the *same* point — you
cannot have point A's translations isolated while point B's tokens are
overlaid).

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
