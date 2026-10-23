# Selection state machine

Reference for how the pinned-point set, the "show selected" isolate, the
per-pin token overlays, and the two canvas-coloring modes
interact — all owned by `App` (`client-js/app.js`) and mirrored into
`DataTable`/`ProjectionView`. Written down once so future features have
a known-correct baseline to extend rather than reverse-engineer.

This describes the **multi-pin selection set** model: a checkbox in each
sentence row of the data table toggles that point's membership in the
set (check = pin, uncheck = unpin) — the checkbox is each row's *only*
pin control; clicking elsewhere on the row does nothing. One point can
be pinned, or several, with symmetric treatment — no special-casing
between "the first one" and "the rest".

## State

Four pieces of state, all on `App`:

- `_selection` — an ordered array of pinned point ids (order = pin
  order; array, not `Set`, so "last pinned" is cheap to find and
  iteration order is stable for rendering/coloring).
- `_focusedId` — the id most recently pinned; always a member of
  `_selection`, or `null` when the set is empty. Purely a rendering hint
  (heavier canvas ring, highlighted table row) — it does not gate which
  rows get an `Actions ▾` dropdown; every pinned row gets one regardless
  of focus.
- `_detailState` — `Map<id, { tokens, tokenPoints }>`. An
  entry is created (tokens off) when a point is first pinned via
  `_togglePin`, and deleted when it's unpinned — no stale entries linger
  for a point that isn't currently pinned.
- `_pinColors` — `Map<id, color>` assigning each currently-pinned point
  a color from a fixed 10-color palette (`d3.schemeTableau10`,
  module-level `PIN_PALETTE` in `app.js` — a separate color system from
  both canvas coloring modes below, so a pin's ring color never
  conflicts with a point's fill color), assigned by pin order and
  cycling if more than 10 points are pinned at once.

`ProjectionView`/`PointLayer` mirror the pinned set via
`setPinnedPoints(Array<{id, color, focused}>)` (one ring per pinned
point, in that pin's color; the focused pin's ring is drawn with a
heavier stroke) and `setTokenOverlays(Map<pinId, points[]>)` (each pin's
token diamonds render in that pin's color instead of a single fixed
color). `DataTable` mirrors it via `setPins(Array<{point, color,
focused, detail}>)`. `App._syncPins()` rebuilds both from
`_selection`/`_focusedId`/`_detailState`/`_pinColors` from scratch on
every change, rather than patching incrementally, so neither can drift
out of sync with the source of truth.

## Transitions

**Toggling a pin** (`DataTable`'s pin checkbox, or a canvas point click
via `onCanvasClickPoint`; both route through `_togglePin`):

- Not currently pinned: assign it the next color in `PIN_PALETTE`, push
  its id onto `_selection`, create its `_detailState` entry with every
  flag off, and set `_focusedId` to it.
- Currently pinned: remove its id from `_selection`, delete its
  `_detailState` and `_pinColors` entries. If it was `_focusedId`, focus
  moves to the new last item in `_selection`, or `null` if the set is
  now empty.
- Either way, `_syncPins()` runs at the end, so canvas rings/overlays and
  the table stay current.

**Clicking anywhere else on a sentence row** (not the pin checkbox):
does nothing — no pin toggle, no highlight. This is a deliberate
simplification once the Selection panel (which used to give plain row
clicks a "focus without pinning" meaning) was removed — see "Removed:
Selection panel" below.

**Clicking a token row** (`onSelectRow` with `row.type === 'token'`):
routes to `table.highlightRow(id)`, which highlights that row only and
never touches `_selection`/`_detailState`. This is the one row-click
interaction that still does something, since token rows have no pin
checkbox of their own.

**Clicking empty canvas space** does nothing — it used to clear the
entire selection, but now that the data table has an explicit **Clear
selection** button (`App.clearSelection()`), an incidental background
click no longer wipes pins. (Clicking an already-pinned point again on
the canvas still just unpins that one point via the normal toggle-off
path above.)

**Clear selection button** (`DataTable`'s `.dt-clear-selection`,
`App.clearSelection()`): clears the entire selection — every pinned id
is removed from `_selection`, every `_detailState`/`_pinColors` entry is
deleted, `_focusedId` resets to `null`. Disabled whenever `_selection` is
empty.

**Select all button** (`DataTable`'s `.dt-select-all`,
`App.selectPoints(points)`): pins every row matching the table's current
filter (search, show-selected), across all pages —
not just the visible page. Points that are already pinned are skipped;
the rest get the same treatment as the pin branch of `_togglePin`
(next palette color, empty `_detailState` entry), in one batch so
`_syncPins()` and the stats refresh run once. `_focusedId` becomes the *first* newly
pinned row (not the last), so the table pages to the start of the
batch instead of jumping to the end of the list. Disabled when the
table has no rows. The palette only has 10 colors, so a large batch
reuses them cyclically.

**Statistics panel** (`StatsView`, `App._refreshStats()`): every change
to `_selection` (`_togglePin`, `selectPoints`, `clearSelection`, a new
dataset) re-requests `POST /stats` with the pinned ids; token
toggles don't, since they don't change the pinned set. The
request is debounced (150ms) and guarded by a counter so a slow response
for an older selection is dropped. An empty selection shows a hint
instead of calling the backend. The statistics are cosine in the full
embedding space, independent of the plot's method/metric: overall
within- vs. cross-language mean cosine, and per language pair the mean
cosine of true translations (shared `pairId`) vs. all other
cross-language pairs, plus MEXA-style retrieval accuracy over the
translations (needs at least 2). Datasets without `pairId` get the
overall numbers only.

**Select translations** (`onSelectTranslations(point)`): pins every
point that shares `point`'s `groupKey` (its `pairId`) through
`selectPoints`, then turns "Show selected" on, so the whole translation
group is both selected and the only thing left at full opacity. It is a
one-shot action, not a toggle: there is no per-pin translations flag,
and it works with any number of pins. Untick "Show selected" to see the
rest of the plot again.

**Show/Hide tokens** (`onToggleTokens(point)`): toggles
`_detailState.get(point.id).tokens`/`tokenPoints`, independently per
pinned point — any number of pins can have tokens shown at once, each
rendered in its own pin color via `setTokenOverlays`. Token projection
only exists in the shared UMAP space (`server.py`'s
`_get_umap_reducer`), so if the method dropdown isn't already `umap`,
this flips it and calls `_refresh({ keepState: true })` first, which keeps
the selection, before fetching its tokens.

**`_refresh()`** (`_loadDataset`) has two modes. Uploading or embedding a
new dataset is a **full reset**: `_selection`, `_focusedId`,
`_detailState`, `_pinColors`, the language filter and the table state are
all cleared. A method/metric dropdown change or **Refit** calls it with
`keepState: true`; if the point ids are unchanged, the selection, pins,
language filter, "Show selected" and the table's search/sort/page all
survive, and only coordinates are refreshed. Token overlays are dropped
(they were projected into the previous fit), and `PointLayer.setPoints()`
state is re-applied (`setLangFilter`, `setShowPinnedOnly`, `_syncPins`).

**Independent of the selection entirely** — these never read or modify
`_selection`/`_detailState`:
- The **label-on-hover** dropdown (`view.setLabelBy`) — only changes
  what the tooltip shows.
- The **fisheye lens** toggle and draggable disk (`view.setFisheyeEnabled`)
  — a position distortion on the canvas; while it's on, hover/click
  hit-testing uses the displaced positions that were last drawn
  (`PointLayer.pointAtDisplayed`).

## Two canvas-coloring modes (mutually exclusive)

`PointLayer._familyModeEnabled` (default `true`) picks between:

- **Family mode** (default): every point is colored by its language's
  linguistic family — the backend attaches a `family` field to every
  point (`server.py`'s `point_family()`: a point's own `family`, else the
  dataset's `lang_families` map, else a Glottolog lookup of the code's
  ISO 639-3 / 639-1 code or English name via `resources/language_families.json`, generated by
  `scripts/build_language_families.py`; anything still unknown falls
  into a shared `"Other"` bucket), and `App._loadDataset`
  copies it onto each point object alongside `lang`/`text`/etc.
  `PointLayer.setPoints()` builds a `d3.scaleOrdinal` color scale from
  whatever distinct families are actually present in the loaded data —
  `PointLayer.getFamilyLegend()` exposes that mapping for the sidebar's
  legend. Every point is fully opaque; there's no "background gray" tier.
- **Language-filter mode**: the pre-existing behavior. Every point is
  neutral gray (`NEUTRAL_COLOR`) by default; only languages explicitly
  selected via the sidebar's language filter get a real color
  (`_langColorScale`, built fresh from just the selected subset), and
  everything else dims to `FILTERED_OUT_OPACITY`.

The sidebar's `#family-color-toggle` checkbox (default checked) drives
`view.setFamilyColorMode(enabled)`; `App` also disables/enables
`#lang-search` alongside it, since the language filter has no visible
effect while family mode is active. Both modes read/write the same
`_colorFor(d)`/`_baseOpacity(d)` methods — switching modes just changes
which branch they take, and re-triggers the opacity animation so the
switch fades rather than snaps.

## Canvas z-order

`PointLayer.draw()` stratifies every point into three tiers before
drawing, regardless of their order in the dataset array (canvas paints
in draw order, so a plain point drawn after a colored/pinned one at the
same screen location would otherwise bury it):

1. **Background** — plain, uncolored points (only reachable in
   language-filter mode; family mode has no background tier).
2. **Colored** — actively colored points (every point in family mode,
   or the language filter's selected subset otherwise).
3. **Pinned** — always drawn last, on top of everything else, even a
   colored point at the same location.

Token overlay diamonds are drawn in their own pass after all three
tiers, unaffected by this stratification (they already always render on
top by virtue of being drawn last).

## The isolate

`PointLayer._showPinnedOnly` (the data table's "Show selected" checkbox,
wired through `App`/`ProjectionView.setShowPinnedOnly`) fades everything
that isn't pinned. With it off, opacity falls back to whichever coloring
mode's base opacity applies.

`ProjectionView._pointAt`'s hit-testing guard uses
`PointLayer.isIsolatedOut(d)` — true whenever `d`'s target opacity would
currently be `ISOLATED_OUT_OPACITY` — so a click on a
point that's invisible because of the isolate correctly falls
through to the background-click handler instead of silently pinning
something the user can't see.

## `Actions ▾` dropdown

One shared component (`client-js/actions-menu.js`'s `renderActionsMenu`,
a render-a-dropdown-for-this-point-id function, not a class with its own
selection state — `App`'s `_detailState` is the single source of truth
it reads from and calls back into). One `Actions ▾` button per pinned
row opens a popover: Select translations (an action), Tokens (a checkbox;
omitted if `!point.hasTokens`). The dropdown stays open
across multiple clicks inside it (`openMenuId` module-level state in
`actions-menu.js`) — closing only on an outside click or Escape
(`installActionsMenuOutsideClickHandler`, installed once in
`App.init()`) — so toggling more than one action doesn't require
reopening it each time. "Select translations" is greyed out
(`actions-menu-disabled` styling plus a tooltip) when no other point
shares the pairId.

## Data table changes

- A pin checkbox (`.dt-pin-checkbox`) is the leftmost column of every
  sentence row — the row's only pin control (see "Toggling a pin"
  above).
- Every row currently in `_selection` gets an `Actions ▾` dropdown
  (previously only the single selected row did) — potentially several
  visible per page.
- Row highlight uses that point's pin color instead of a single fixed
  selected-row color.
- The `dt-show-selected` checkbox ("Show selected") in the `dt-top` bar
  sets `_showSelectedOnly`, which makes `_visibleRows()` additionally
  filter down to whatever's in `DataTable`'s own mirrored `_pins` array
  (`const pinnedIds = new Set(this._pins.map(p => p.point.id)); rows =
  rows.filter(p => pinnedIds.has(p.id));` — `DataTable` never reads
  `App._selection` directly, only the `_pins` snapshot `setPins()` last
  handed it), combined with the existing search filter by AND. The same
  checkbox also drives the canvas isolate described above via
  `onToggleShowSelectedOnly`.

## Removed: Selection panel

An earlier iteration had a fourth toggleable view (`SelectionPanel`)
listing every pinned point, with its own "click focuses without
unpinning, only × unpins" exception. It's been removed entirely —
pinning now lives exclusively in the data table's checkbox column, with
no separate focus-vs-unpin distinction to maintain. `App._focusPin` and
the panel's layout-config/fullscreen wiring were deleted along with it.

## Known asymmetry (intentional)

A canvas click can only ever pin/unpin a **sentence** point — token
overlay points are hoverable but not clickable on canvas
(`ProjectionView._onCanvasClick` checks `_overlayPointAt` first and
returns early). Pinning a *token* point is not possible from either
canvas or table; clicking a token row in the table only highlights it
(see "Clicking a token row" above).

## URL state (removed)

An earlier iteration mirrored method/metric/hyperparams/label-by/the
pinned id set into the URL query string (`client-js/url-state.js`,
`App._syncUrl()`/`_applyPendingSelection()`) behind a "Copy link"
button, so a shared link would restore the same view. Pulled out for
now — nothing currently reads or writes the URL's query string.
