# Selection state machine

Reference for how the pinned-point set, translations-isolate, and the
per-pin token/neighbor overlays interact — all owned by `App`
(`client-js/app.js`) and mirrored into `DataTable`/`SelectionPanel`/
`ProjectionView`. Written down once so future features have a
known-correct baseline to extend rather than reverse-engineer.

This describes the **multi-pin selection set** model: clicking a point
toggles its membership in the set (click = pin, click again = unpin)
rather than "select" and "pin" being two different concepts. One point
can be pinned, or several, with symmetric treatment — no special-casing
between "the first one" and "the rest".

## State

Four pieces of state, all on `App`:

- `_selection` — an ordered array of pinned point ids (order = pin
  order; array, not `Set`, so "last pinned" is cheap to find and
  iteration order is stable for rendering/coloring).
- `_focusedId` — the id last clicked/pinned, or whichever pin the
  Selection panel was last used to focus; always a member of
  `_selection`, or `null` when the set is empty. Purely a rendering hint
  (which ring style to use, which table row to page/scroll to) — it does
  not gate which rows get an `Actions ▾` dropdown; every pinned row gets
  one regardless of focus.
- `_detailState` — `Map<id, { translations, tokens, tokenPoints,
  neighbors, neighborsK, neighborRows }>`. An entry is created (all
  flags off, `neighborsK: 5`) when a point is first pinned via
  `_togglePin`, and deleted when it's unpinned — no stale entries linger
  for a point that isn't currently pinned.
- `_pinColors` — `Map<id, color>` assigning each currently-pinned point
  a color from a fixed 10-color palette (`d3.schemeTableau10`,
  module-level `PIN_PALETTE` in `app.js` — distinct from the language
  filter's own color scale; a pin and a language-filter selection are two
  independent color systems that can coexist), assigned by pin order and
  cycling if more than 10 points are pinned at once.

`ProjectionView`/`PointLayer` mirror the pinned set via
`setPinnedPoints(Array<{id, color, focused}>)` (one ring per pinned
point, in that pin's color; the focused pin's ring is drawn with a
heavier stroke) and `setTokenOverlays(Map<pinId, points[]>)` (each pin's
token diamonds render in that pin's color instead of a single fixed
color). `DataTable`/`SelectionPanel` mirror it via `setPins(Array<{point,
color, focused, detail}>)`. `App._syncPins()` rebuilds all three from
`_selection`/`_focusedId`/`_detailState`/`_pinColors` from scratch on
every change, rather than patching incrementally, so none of them can
drift out of sync with the source of truth.

## Transitions

**Toggling a pin** (`onCanvasClickPoint` from a canvas click, or
`onSelectRow` from a data-table row click on a sentence row, both routed
through `_togglePin`; a neighbor-row click in the table resolves back to
the full point object first, then also calls `_togglePin`):

- Not currently pinned: assign it the next color in `PIN_PALETTE`, push
  its id onto `_selection`, create its `_detailState` entry with every
  flag off, and set `_focusedId` to it.
  - Exception: if this is the *second* pin and the sole existing pin
    currently has `translations` on, that flag (and the canvas isolate
    it was driving) is cleared first, in the same `_togglePin` call —
    translations can only ever be active while exactly one point is
    pinned (see below), so going from 1 → 2 pins must never leave it
    active on the point that's about to stop being the only one.
- Currently pinned: remove its id from `_selection`, delete its
  `_detailState` and `_pinColors` entries. If it was `_focusedId`, focus
  moves to the new last item in `_selection`, or `null` if the set is
  now empty.
- Either way, `_syncPins()` and `_syncUrl()` run at the end, so canvas
  rings/overlays, the table, the Selection panel, and the URL's `ids=`
  param all stay current.

**Clicking a token row** (`onSelectRow` with `row.type === 'token'`):
unchanged from the single-selection model — routes to
`table.highlightRow(id)`, which highlights that row only and never
touches `_selection`/`_detailState`. This is the one interaction that
never changes pin membership.

**Background click** (`onBackgroundClick` — canvas empty space):
clears the entire selection — every pinned id is removed from
`_selection`, every `_detailState`/`_pinColors` entry is deleted,
`_focusedId` resets to `null`, and any active translations isolate is
cleared on the canvas. (Clicking an already-pinned point again still
just unpins that one point via the normal toggle-off path above — it
doesn't go through this method.)

**Focusing without unpinning** (`_focusPin(id)` — the Selection panel's
row click, wired as `onFocusPin`): moves `_focusedId` to `id` if it's
currently pinned, then calls `_syncPins()`; does not touch `_selection`
or `_detailState` at all. This is the Selection panel's deliberate
exception to "click toggles the pin" — the panel is the one place in the
app where an accidental click on a row shouldn't cost you your working
set, so only its explicit `×` button (wired as `onUnpin`, which just
calls `_togglePin`) unpins from there. Clicking the same point again on
the canvas, or its row in the full data table, still toggles as usual.

**Show/Hide translations** (`onToggleTranslations(point)`): toggles
`_detailState.get(point.id).translations`. Guarded by `_selection.length
!== 1 || _selection[0] !== point.id` — a no-op if reached any other way,
though callers (the `Actions ▾` dropdown's "Translations" line) are
already expected to have it disabled whenever more than one point is
pinned, so this is defense-in-depth rather than the primary gate. When
turned on, isolates the point's translation group on the canvas
(`view.highlightParallel(point.groupKey)`) and collapses the table to
that group (`DataTable._visibleRows()`'s `_translationsActivePin()`
check); turning it off calls `view.clearHighlight()`. Translations are
the one flag in `_detailState` that is **not** independent per pin —
by construction there is never more than one pinned point with
`translations: true` at a time (enforced by the guard above plus the
auto-clear on 1 → 2 pin described earlier), so `PointLayer`'s
`_activeGroupKey`/`highlightGroup()`/`getActiveGroupKey()` stay a single
exclusive piece of state, unchanged from the old single-selection model
— there is never a second group competing to be isolated.

**Show/Hide tokens** (`onToggleTokens(point)`): toggles
`_detailState.get(point.id).tokens`/`tokenPoints`, independently per
pinned point — any number of pins can have tokens shown at once, each
rendered in its own pin color via `setTokenOverlays`. Token projection
only exists in the shared UMAP space (`server.py`'s
`_get_umap_reducer`), so if the method dropdown isn't already `umap`,
this flips it and calls `_refresh()` first — which, like any
`_refresh()` (see below), wipes the *entire* selection — and then
explicitly re-pins just this one point (`_togglePin(reselected)`) before
fetching its tokens. This is the **only** path where a `_refresh()` is
followed by automatic re-pinning; every other refresh trigger just
resets to an empty selection.

**Show/Hide neighbors** (`onToggleNeighbors(point)`, and
`onChangeNeighborsK(point, k)` for the per-pin `k` input): toggles
`_detailState.get(point.id).neighbors`/`neighborRows`, independently per
pinned point, same as tokens. No method switch is ever needed —
`/neighbors` operates on the full embedding vectors regardless of what's
plotted — so this never triggers a `_refresh()`. Changing `k` while
neighbors are already shown re-fetches immediately at the new count
rather than just remembering it for next time.

**Anything that calls `_refresh()`** (`_loadDataset`, triggered by a
method/metric dropdown change, **Refit**, or uploading a new dataset):
full reset. `_selection`, `_focusedId`, `_detailState`, and `_pinColors`
are all cleared, the canvas ring/isolate and token overlays are cleared,
and nothing is restored afterward — unless it's the token-overlay
auto-switch case above, the only deliberate exception.

**Independent of the selection entirely** — these never read or modify
`_selection`/`_detailState`:
- The **label-on-hover** dropdown (`view.setLabelBy`) — only changes
  what the tooltip shows.
- The **language filter** (`view.setLangFilter`) — only changes point
  colors/opacity; a completely separate color system from pin colors.
- Table search, sort, and pagination — if a pinned/focused row scrolls
  out of the current filtered/sorted page, its `Actions ▾` dropdown
  simply isn't rendered until you navigate back to it; the underlying
  `_detailState` entry is untouched and the row picks its rendering back
  up correctly once visible again.

## `Actions ▾` dropdown

One shared component (`client-js/actions-menu.js`'s `renderActionsMenu`,
a render-a-dropdown-for-this-point-id function, not a class with its own
selection state — `App`'s `_detailState` is the single source of truth
it reads from and calls back into) used identically by `DataTable` and
`SelectionPanel`. One `Actions ▾` button per pinned row opens a
checklist popover: Translations, Tokens (omitted if `!point.hasTokens`),
Neighbors (with an inline `k` input), a divider, and a disabled
"Configure…" placeholder reserving the menu shape for a future holistic
per-point settings panel. The dropdown stays open across multiple clicks
inside it (`openMenuId` module-level state in `actions-menu.js`) —
closing only on an outside click or Escape
(`installActionsMenuOutsideClickHandler`, installed once in
`App.init()`) — so toggling more than one action doesn't require
reopening it each time. "Translations" is greyed out
(`aria-disabled`-equivalent styling plus a tooltip) whenever
`_selection.length !== 1`, matching the guard in `onToggleTranslations`
above.

## Data table changes

- Every row currently in `_selection` gets an `Actions ▾` dropdown
  (previously only the single selected row did) — potentially several
  visible per page.
- Row highlight uses that point's pin color instead of a single fixed
  selected-row color.
- The `dt-show-selected` checkbox ("Show selected") in the `dt-top` bar
  sets `_showSelectedOnly`*, which makes `_visibleRows()` additionally
  filter down to whatever's in `DataTable`'s own mirrored `_pins` array
  (`const pinnedIds = new Set(this._pins.map(p => p.point.id)); rows =
  rows.filter(p => pinnedIds.has(p.id));` — `DataTable` never reads
  `App._selection` directly, only the `_pins` snapshot `setPins()` last
  handed it), combined with the existing search filter by AND.
- Translations keep the table's whole-table group filter, unchanged:
  `_translationsActivePin()` still makes `_visibleRows()` collapse to
  just that one group while active. This stays correct precisely
  because translations are restricted to exactly one pinned point —
  there's never a second group that could also want the table filtered
  to it.

\* the checkbox is internally still referred to by its original
`show_selected` name in the design doc and README.

## Selection panel

A fourth toggleable view (`client-js/selection-panel.js`'s
`SelectionPanel`, `#selection-panel` in the layout, participating in the
layout-config panel via its own `toggle-selection` checkbox, same
localStorage persistence shape as the other panels). Lists every entry
in `_selection`, in pin order, as a simple non-paginated,
non-searchable list — bounded by how many points a person will
realistically pin, unlike the full dataset. Each row shows a color
swatch matching the point's canvas ring, its id/lang/text, the same
shared `Actions ▾` dropdown, and an explicit `×` button. See "Focusing
without unpinning" above for the panel's one behavioral exception:
clicking a row focuses it (`_focusPin`) without unpinning; only `×`
(`onUnpin`, i.e. `_togglePin`) removes it from the set.

## Known asymmetry (intentional)

A canvas click can only ever pin/unpin a **sentence** point — token
overlay points are hoverable but not clickable on canvas
(`ProjectionView._onCanvasClick` checks `_overlayPointAt` first and
returns early, unchanged from the single-selection model). Pinning a
*token* point is not possible from either canvas or table; clicking a
token row in the table only highlights it (see "Clicking a token row"
above).

## URL state

`client-js/url-state.js`'s `ids` key holds a comma-separated list of
pinned ids, in pin order. `App._syncUrl()` writes it from `_selection`
at the end of every state-changing action, and `App._applyPendingSelection()`
re-pins each id from a restored URL in that same order once the initial
dataset has loaded — since `_togglePin` always focuses whatever it just
pinned, the last id in the list ends up focused after restore, matching
"last = focused" without needing any separate bookkeeping. Per-pin
`translations`/`tokens`/`neighbors` flags are **not** encoded in the
URL — restoring a link re-pins the same points with all three toggles
off. This is a deliberate scope cut (avoiding a combinatorial
per-id-per-flag query string), not a silent gap — see the README's
"Copy link" bullet.
