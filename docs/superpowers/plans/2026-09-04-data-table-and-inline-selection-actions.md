# Data Table + Inline Selection Actions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a data table view of the embedding dataset (sentence and token rows with their attributes) and replace the current auto-isolate-on-click interaction with a linked selection model: selecting a sentence — in the table or on the canvas — reveals inline row actions to show/hide its translations and show/hide its tokens.

**Architecture:** A new `DataTable` component owns a table of sentence rows (always, from `GET /points`) plus token rows (inserted only for whichever sentence currently has its tokens shown). `App` owns one unified selection state and drives both the table and the canvas (`ProjectionView`/`PointLayer`) from it — neither view has its own independent selection. The old sidebar `DetailPanel` is retired; its job (show tokens toggle, implicitly language display) is fully absorbed by the table's inline row actions.

**Tech Stack:** Vanilla JS (D3 v7, Canvas, ES modules, no build step, no frontend test framework — pre-existing constraint), Python/FastAPI backend (pytest).

**Spec:** `docs/superpowers/specs/2026-09-04-data-table-and-inline-selection-actions-design.md`

## Global Constraints

- Layout: sidebar (unchanged: file picker, method select, language filter) | data table (new) | scatter plot (unchanged), left to right.
- Table columns: ID | Language | Text | PairId | Type (sentence/token) | Token strings (sentence rows only, blank for token rows) | Actions (inline buttons, sentence rows only, only on the currently-selected sentence).
- No raw vector column (out of scope per spec).
- Pagination: fixed page size of 50 rows, Prev/Next controls — no infinite scroll, no virtual-scroll library.
- Token rows are pinned directly beneath their parent regardless of active sort/search — never independently sorted, and hidden if their parent is filtered out.
- Selecting (table row or canvas point) only highlights — it does not isolate/fade or show tokens by itself. Isolating and token-overlay are driven exclusively by the two inline action buttons.
- Only one sentence's translations-group-filter and one sentence's token-overlay can be active at a time; selecting a different sentence clears the previous one's toggles first. Clicking the already-selected point again, or clicking canvas background, clears the selection and both toggles.
- Selecting a token row highlights it but shows no action buttons (a token has no translations and can't be split further) and does not disturb the current sentence's active toggles.
- `GET /points` gains one additive field, `tokenTexts` (token strings only, no vectors) — `GET /project`, `GET /project/tokens/{id}`, `POST /dataset` stay unchanged.
- User-controlled text (sentence text, language, token strings) rendered into table cells must be HTML-escaped — this app accepts arbitrary uploaded JSON via `POST /dataset`, so text fields are not trusted input.

---

### Task 1: `server.py` — add `tokenTexts` to `GET /points`

**Files:**
- Modify: `server.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `GET /points` response points gain `"tokenTexts": list[str] | None` (the token strings from `tokenVectors`, or `None` if the point has none). Task 4 (`DataTable`) reads this field directly off `App.points` entries.

- [ ] **Step 1: Write the failing test**

In `tests/test_server.py`, find:

```python
def test_points_endpoint_reports_has_tokens(client):
    res = client.get("/points")
    body = res.json()
    by_id = {p["id"]: p for p in body["points"]}
    assert by_id["en-0"]["hasTokens"] is False
    assert by_id["en-1"]["hasTokens"] is True
```

Replace with:

```python
def test_points_endpoint_reports_has_tokens(client):
    res = client.get("/points")
    body = res.json()
    by_id = {p["id"]: p for p in body["points"]}
    assert by_id["en-0"]["hasTokens"] is False
    assert by_id["en-1"]["hasTokens"] is True


def test_points_endpoint_reports_token_texts(client):
    res = client.get("/points")
    body = res.json()
    by_id = {p["id"]: p for p in body["points"]}
    assert by_id["en-0"]["tokenTexts"] is None
    assert by_id["en-1"]["tokenTexts"] == ["bye", "!"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_server.py::test_points_endpoint_reports_token_texts -v`
Expected: FAIL — `KeyError: 'tokenTexts'`.

- [ ] **Step 3: Add `tokenTexts` to the `/points` response**

In `server.py`, find:

```python
        "points": [
            {
                "id": p["id"], "lang": p["lang"], "pairId": p.get("pairId"), "text": p.get("text"),
                "hasTokens": p.get("tokenVectors") is not None,
            }
            for p in d["points"]
        ],
```

Replace with:

```python
        "points": [
            {
                "id": p["id"], "lang": p["lang"], "pairId": p.get("pairId"), "text": p.get("text"),
                "hasTokens": p.get("tokenVectors") is not None,
                "tokenTexts": [t["token"] for t in p["tokenVectors"]] if p.get("tokenVectors") is not None else None,
            }
            for p in d["points"]
        ],
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_server.py -v`
Expected: 21 passed.

- [ ] **Step 5: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "Add tokenTexts to GET /points for the data table's token-strings column"
```

---

### Task 2: `point-layer.js` / `projection-view.js` — persistent selection ring

**Files:**
- Modify: `client-js/point-layer.js`
- Modify: `client-js/projection-view.js`

**Interfaces:**
- Consumes: nothing new.
- Produces: `PointLayer.setSelectedId(id)` / `PointLayer.clearSelectedId()`; `ProjectionView.setSelectedId(id)` / `ProjectionView.clearSelectedId()`. Task 5 (`app.js`) calls the `ProjectionView` methods whenever the unified selection state changes.

No automated tests exist for the frontend (pre-existing constraint) — verify with `node --check` and by hand-tracing the draw logic.

- [ ] **Step 1: Add a `_selectedId` field to `PointLayer`**

In `client-js/point-layer.js`, find:

```js
  constructor(onChange) {
    this._onChange = onChange; // called after every opacity update, so the owner redraws
    this.points = [];
    this.baseXScale = null;
    this.baseYScale = null;
    this._timer = null;
    this._activeGroupKey = null;
    this._selectedLangs = new Set();
    this._langColorScale = null;
    this._tokenOverlay = [];
  }

  setPoints(points, baseXScale, baseYScale) {
    this._activeGroupKey = null;
    this._selectedLangs = new Set();
    this._langColorScale = null;
    this._tokenOverlay = [];
    this.points = points.map(p => ({ ...p, opacity: 1 }));
    this.baseXScale = baseXScale;
    this.baseYScale = baseYScale;
  }
```

Replace with:

```js
  constructor(onChange) {
    this._onChange = onChange; // called after every opacity update, so the owner redraws
    this.points = [];
    this.baseXScale = null;
    this.baseYScale = null;
    this._timer = null;
    this._activeGroupKey = null;
    this._selectedLangs = new Set();
    this._langColorScale = null;
    this._tokenOverlay = [];
    this._selectedId = null;
  }

  setPoints(points, baseXScale, baseYScale) {
    this._activeGroupKey = null;
    this._selectedLangs = new Set();
    this._langColorScale = null;
    this._tokenOverlay = [];
    this._selectedId = null;
    this.points = points.map(p => ({ ...p, opacity: 1 }));
    this.baseXScale = baseXScale;
    this.baseYScale = baseYScale;
  }

  // Selection is independent of the isolate-fade group and of hover: a
  // point can be selected without anything being faded (selecting no
  // longer isolates by itself — that's driven by the data table's
  // "Show translations" action instead).
  setSelectedId(id) {
    this._selectedId = id;
    this._onChange();
  }

  clearSelectedId() {
    this._selectedId = null;
    this._onChange();
  }
```

- [ ] **Step 2: Draw a selection ring around the selected point**

In `client-js/point-layer.js`, find:

```js
      ctx.globalAlpha = d.opacity;
      ctx.lineWidth = (hovered ? 1.5 : 0.5) / transform.k;
      ctx.strokeStyle = hovered ? '#333' : '#fff';
      ctx.stroke();
    }
    for (const d of this._tokenOverlay) {
```

Replace with:

```js
      ctx.globalAlpha = d.opacity;
      ctx.lineWidth = (hovered ? 1.5 : 0.5) / transform.k;
      ctx.strokeStyle = hovered ? '#333' : '#fff';
      ctx.stroke();
      if (d.id === this._selectedId) {
        ctx.beginPath();
        ctx.arc(cx, cy, 8.5 / transform.k, 0, 2 * Math.PI);
        ctx.globalAlpha = d.opacity;
        ctx.lineWidth = 1.8 / transform.k;
        ctx.strokeStyle = '#1a73e8';
        ctx.stroke();
      }
    }
    for (const d of this._tokenOverlay) {
```

- [ ] **Step 3: Expose the selection methods on `ProjectionView`**

In `client-js/projection-view.js`, find:

```js
  highlightParallel(groupKey) { this.pointLayer.highlightGroup(groupKey); }
  clearHighlight() { this.pointLayer.clear(); }
  setLangFilter(langs) { this.pointLayer.setLangFilter(langs); }
  setTokenOverlay(tokenPoints) { this.pointLayer.setTokenOverlay(tokenPoints); }
  clearTokenOverlay() { this.pointLayer.clearTokenOverlay(); }
```

Replace with:

```js
  highlightParallel(groupKey) { this.pointLayer.highlightGroup(groupKey); }
  clearHighlight() { this.pointLayer.clear(); }
  setLangFilter(langs) { this.pointLayer.setLangFilter(langs); }
  setTokenOverlay(tokenPoints) { this.pointLayer.setTokenOverlay(tokenPoints); }
  clearTokenOverlay() { this.pointLayer.clearTokenOverlay(); }
  setSelectedId(id) { this.pointLayer.setSelectedId(id); }
  clearSelectedId() { this.pointLayer.clearSelectedId(); }
```

- [ ] **Step 4: Verify**

```bash
node --check client-js/point-layer.js
node --check client-js/projection-view.js
```

Expected: no output (syntax OK). Hand-trace: `setSelectedId('en-1')` sets `_selectedId`, next `draw()` call draws an extra blue ring (radius `8.5/k`) around whichever point in `this.points` has `id === 'en-1'`, regardless of that point's current `opacity` — a selected point is never invisible since selecting never fades anything by itself (Global Constraints).

- [ ] **Step 5: Commit**

```bash
git add client-js/point-layer.js client-js/projection-view.js
git commit -m "Add a persistent selection ring, independent of isolate-fade and hover"
```

---

### Task 3: `index.html` / `css/style.css` — data table pane, retire detail panel

**Files:**
- Modify: `index.html`
- Modify: `css/style.css`

**Interfaces:**
- Consumes: nothing new.
- Produces: a `<div id="data-table"></div>` element between `#sidebar` and `#main`, ready for Task 4's `DataTable` to render into. The old `#detail-placeholder`/`#detail-lang`/`#detail-tokens-toggle` elements no longer exist.

- [ ] **Step 1: Replace the sidebar's detail-panel markup and add the data-table pane**

In `index.html`, find:

```html
    <div id="lang-filter-controls">
      <label for="lang-search">Filter languages:</label>
      <input type="text" id="lang-search" placeholder="Type to search...">
      <div id="lang-suggestions"></div>
      <div id="lang-chips"></div>
    </div>

    <div id="detail-placeholder">Click a point to isolate its translations</div>
    <div id="detail-lang"></div>
    <button type="button" id="detail-tokens-toggle">Show tokens</button>
  </div>

  <div id="main">
```

Replace with:

```html
    <div id="lang-filter-controls">
      <label for="lang-search">Filter languages:</label>
      <input type="text" id="lang-search" placeholder="Type to search...">
      <div id="lang-suggestions"></div>
      <div id="lang-chips"></div>
    </div>
  </div>

  <div id="data-table"></div>

  <div id="main">
```

- [ ] **Step 2: Update the format-help wording for `tokenVectors`**

In `index.html`, find:

```html
        <p><code>tokenVectors</code> is optional — if present, a "Show tokens" button appears when this point is selected, projecting its tokens into the same UMAP space as the main plot.</p>
```

Replace with:

```html
        <p><code>tokenVectors</code> is optional — if present, selecting this point in the data table reveals a "Show tokens" action, projecting its tokens into the same UMAP space as the main plot.</p>
```

- [ ] **Step 3: Replace the detail-panel CSS rules with data-table rules**

In `css/style.css`, find:

```css
#detail-placeholder { color: #bbb; margin-top: 20px; text-align: center; font-size: 0.9em; }
#detail-lang { font-size: 0.85em; font-weight: bold; display: none; margin-top: 20px; }

#detail-tokens-toggle { display: none; margin-top: 12px; font-size: 0.78em; padding: 4px 10px; cursor: pointer; }

#main { flex: 1; position: relative; }
```

Replace with:

```css
#data-table {
  width: 420px;
  min-width: 300px;
  display: flex;
  flex-direction: column;
  border-right: 1px solid #ddd;
  background: #fff;
  overflow: hidden;
}
#data-table .dt-top { display: flex; align-items: center; gap: 8px; padding: 8px 10px; border-bottom: 1px solid #ddd; flex-shrink: 0; }
#data-table .dt-search { flex: 1; font-size: 0.8em; padding: 4px 6px; }
#data-table .dt-count { font-size: 0.75em; color: #888; white-space: nowrap; }
#data-table .dt-scroll { flex: 1; overflow: auto; }
#data-table table { width: 100%; border-collapse: collapse; font-size: 0.78em; }
#data-table thead { position: sticky; top: 0; background: #f7f7f7; }
#data-table th { text-align: left; padding: 5px 8px; border-bottom: 2px solid #ddd; cursor: pointer; white-space: nowrap; }
#data-table th:hover { background: #eee; }
#data-table td { padding: 4px 8px; border-bottom: 1px solid #f0f0f0; vertical-align: top; }
#data-table td.dt-text { max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#data-table tr.dt-row { cursor: pointer; }
#data-table tr.dt-row:hover td { background: #f8f8f8; }
#data-table tr.dt-row-selected td { background: #e8f0fe; }
#data-table tr.dt-row-token td { color: #666; }
#data-table tr.dt-row-token td.dt-text { padding-left: 20px; }
#data-table .dt-type { font-size: 0.9em; padding: 1px 6px; border-radius: 8px; }
#data-table .dt-type-sentence { background: #e6f4ea; color: #137333; }
#data-table .dt-type-token { background: #fce8e6; color: #c5221f; }
#data-table .dt-actions-cell { white-space: nowrap; }
#data-table .dt-action-btn { font-size: 0.85em; padding: 2px 6px; margin-right: 4px; cursor: pointer; }
#data-table .dt-foot { display: flex; align-items: center; justify-content: center; gap: 10px; padding: 6px; border-top: 1px solid #ddd; font-size: 0.78em; color: #666; flex-shrink: 0; }
#data-table .dt-foot button { cursor: pointer; }

#main { flex: 1; position: relative; }
```

- [ ] **Step 4: Commit**

```bash
git add index.html css/style.css
git commit -m "Add data-table pane, retire sidebar detail-panel markup"
```

---

### Task 4: `client-js/data-table.js` — the data table component

**Files:**
- Create: `client-js/data-table.js`

**Interfaces:**
- Consumes: `#data-table` element (Task 3).
- Produces: `export class DataTable`, constructed as `new DataTable({ containerId }, { onRowClick, onToggleTranslations, onToggleTokens })`. Public methods: `setPoints(points)`, `selectRow(id)`, `highlightRow(id)`, `clearSelection()`, `showTranslationsFor(groupKey)`, `hideTranslations()`, `showTokensFor(pointId, tokenRows)`, `hideTokensFor(pointId)`. Callback shapes: `onRowClick({...row, type: 'sentence'|'token'})`, `onToggleTranslations(point)`, `onToggleTokens(point)`. Task 5 (`app.js`) is the only consumer.

Row objects passed via `setPoints` must have `{id, lang, text, pairId, hasTokens, tokenTexts, groupKey}`. Token rows passed via `showTokensFor` must have `{id, lang, text}`.

- [ ] **Step 1: Create `client-js/data-table.js`**

```js
// Data table: sentence rows always, token rows only for whichever point
// currently has its tokens shown. Selecting a sentence row reveals inline
// action buttons (toggle translations / toggle tokens) directly in that
// row — App drives all of this through the same selection state it drives
// the canvas with, so table and canvas selection always agree.

const PAGE_SIZE = 50;

// Sentence text, language, and token strings all come from whatever
// dataset JSON was uploaded via POST /dataset — untrusted input, not
// baked into the app — so anything rendered into a cell goes through this
// before it touches innerHTML.
function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

export class DataTable {
  constructor({ containerId }, { onRowClick, onToggleTranslations, onToggleTokens }) {
    this.container = document.getElementById(containerId);
    this._onRowClick = onRowClick;
    this._onToggleTranslations = onToggleTranslations;
    this._onToggleTokens = onToggleTokens;

    this.points = [];
    this._tokenRows = [];
    this._tokensParentId = null;
    this._translationsGroupKey = null;
    this._selectedId = null;
    this._selectedSentenceId = null;
    this._search = '';
    this._sortKey = null;
    this._sortAsc = true;
    this._page = 0;

    this.container.innerHTML = `
      <div class="dt-top">
        <input type="text" class="dt-search" placeholder="Search language or text...">
        <span class="dt-count"></span>
      </div>
      <div class="dt-scroll">
        <table>
          <thead>
            <tr>
              <th data-key="id">ID</th>
              <th data-key="lang">Language</th>
              <th data-key="text">Text</th>
              <th data-key="pairId">PairId</th>
              <th>Type</th>
              <th>Token strings</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
      <div class="dt-foot">
        <button type="button" class="dt-prev">&larr;</button>
        <span class="dt-page"></span>
        <button type="button" class="dt-next">&rarr;</button>
      </div>
    `;

    this._searchInput = this.container.querySelector('.dt-search');
    this.tbody = this.container.querySelector('tbody');
    this._countEl = this.container.querySelector('.dt-count');
    this._pageEl = this.container.querySelector('.dt-page');

    this._searchInput.addEventListener('input', () => {
      this._search = this._searchInput.value;
      this._page = 0;
      this._render();
    });
    this.container.querySelectorAll('th[data-key]').forEach(th => {
      th.addEventListener('click', () => this._sortBy(th.dataset.key));
    });
    this.container.querySelector('.dt-prev').addEventListener('click', () => {
      if (this._page > 0) { this._page--; this._render(); }
    });
    this.container.querySelector('.dt-next').addEventListener('click', () => {
      this._page++; this._render();
    });
  }

  // Called whenever a new dataset loads — resets every piece of state that
  // only made sense for the previous dataset's points.
  setPoints(points) {
    this.points = points;
    this._tokenRows = [];
    this._tokensParentId = null;
    this._translationsGroupKey = null;
    this._selectedId = null;
    this._selectedSentenceId = null;
    this._search = '';
    this._searchInput.value = '';
    this._sortKey = null;
    this._page = 0;
    this._render();
  }

  // Selects a sentence row: highlights it, makes it the row that shows
  // inline actions, and pages to wherever it currently sits in the
  // filtered/sorted list. Used both for table row clicks and for canvas
  // point clicks (which always select a sentence — token overlay points
  // aren't clickable on canvas).
  selectRow(id) {
    this._selectedId = id;
    this._selectedSentenceId = id;
    this._pageToRow(id);
    this._render();
  }

  // Highlights a row without touching which sentence's actions are shown
  // — used for token row clicks, since a token has no actions of its own
  // and shouldn't hide its parent sentence's.
  highlightRow(id) {
    this._selectedId = id;
    this._render();
  }

  clearSelection() {
    this._selectedId = null;
    this._selectedSentenceId = null;
    this._render();
  }

  showTranslationsFor(groupKey) {
    this._translationsGroupKey = groupKey;
    this._pageToRow(this._selectedSentenceId);
    this._render();
  }

  hideTranslations() {
    this._translationsGroupKey = null;
    this._pageToRow(this._selectedSentenceId);
    this._render();
  }

  showTokensFor(pointId, tokenRows) {
    this._tokensParentId = pointId;
    this._tokenRows = tokenRows;
    this._render();
  }

  hideTokensFor(pointId) {
    if (this._tokensParentId !== pointId) return;
    this._tokensParentId = null;
    this._tokenRows = [];
    this._render();
  }

  _pageToRow(id) {
    if (id === null) return;
    const rows = this._visibleRows();
    const index = rows.findIndex(r => r.id === id);
    if (index >= 0) this._page = Math.floor(index / PAGE_SIZE);
  }

  _visibleRows() {
    let rows;
    if (this._translationsGroupKey !== null) {
      rows = this.points.filter(p => p.groupKey === this._translationsGroupKey);
    } else {
      const q = this._search.trim().toLowerCase();
      rows = q
        ? this.points.filter(p => p.lang.toLowerCase().includes(q) || p.text.toLowerCase().includes(q))
        : this.points;
    }
    if (this._sortKey) {
      const key = this._sortKey;
      rows = [...rows].sort((a, b) => {
        const v = a[key] < b[key] ? -1 : a[key] > b[key] ? 1 : 0;
        return this._sortAsc ? v : -v;
      });
    }
    return rows;
  }

  _sortBy(key) {
    if (this._sortKey === key) this._sortAsc = !this._sortAsc;
    else { this._sortKey = key; this._sortAsc = true; }
    this._page = 0;
    this._render();
  }

  _render() {
    const rows = this._visibleRows();
    const total = rows.length;
    const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
    if (this._page >= totalPages) this._page = totalPages - 1;
    const pageRows = rows.slice(this._page * PAGE_SIZE, (this._page + 1) * PAGE_SIZE);

    this._countEl.textContent = `${total} rows`;
    this._pageEl.textContent = `Page ${this._page + 1} / ${totalPages}`;

    this.tbody.innerHTML = '';
    pageRows.forEach(p => {
      this.tbody.appendChild(this._renderSentenceRow(p));
      if (this._tokensParentId === p.id) {
        this._tokenRows.forEach(t => this.tbody.appendChild(this._renderTokenRow(t)));
      }
    });
  }

  _renderSentenceRow(p) {
    const tr = document.createElement('tr');
    const highlighted = p.id === this._selectedId;
    const hasActions = p.id === this._selectedSentenceId;
    tr.className = 'dt-row' + (highlighted ? ' dt-row-selected' : '');
    const translationsOn = hasActions && this._translationsGroupKey === p.groupKey;
    const tokensOn = hasActions && this._tokensParentId === p.id;

    tr.innerHTML = `
      <td>${escapeHtml(String(p.id))}</td>
      <td>${escapeHtml(p.lang)}</td>
      <td class="dt-text" title="${escapeHtml(p.text)}">${escapeHtml(p.text)}</td>
      <td>${p.pairId ?? ''}</td>
      <td><span class="dt-type dt-type-sentence">sentence</span></td>
      <td class="dt-text">${p.hasTokens ? escapeHtml(p.tokenTexts.join(', ')) : ''}</td>
      <td class="dt-actions-cell"></td>
    `;

    if (hasActions) {
      const actionsCell = tr.querySelector('.dt-actions-cell');
      const transBtn = document.createElement('button');
      transBtn.type = 'button';
      transBtn.className = 'dt-action-btn';
      transBtn.textContent = translationsOn ? 'Hide translations' : 'Show translations';
      transBtn.addEventListener('click', e => { e.stopPropagation(); this._onToggleTranslations(p); });
      actionsCell.appendChild(transBtn);

      if (p.hasTokens) {
        const tokBtn = document.createElement('button');
        tokBtn.type = 'button';
        tokBtn.className = 'dt-action-btn';
        tokBtn.textContent = tokensOn ? 'Hide tokens' : 'Show tokens';
        tokBtn.addEventListener('click', e => { e.stopPropagation(); this._onToggleTokens(p); });
        actionsCell.appendChild(tokBtn);
      }
    }

    tr.addEventListener('click', () => this._onRowClick({ ...p, type: 'sentence' }));
    return tr;
  }

  _renderTokenRow(t) {
    const tr = document.createElement('tr');
    const highlighted = t.id === this._selectedId;
    tr.className = 'dt-row dt-row-token' + (highlighted ? ' dt-row-selected' : '');
    tr.innerHTML = `
      <td>${escapeHtml(t.id)}</td>
      <td>${escapeHtml(t.lang)}</td>
      <td class="dt-text" title="${escapeHtml(t.text)}">${escapeHtml(t.text)}</td>
      <td></td>
      <td><span class="dt-type dt-type-token">token</span></td>
      <td></td>
      <td></td>
    `;
    tr.addEventListener('click', () => this._onRowClick({ ...t, type: 'token' }));
    return tr;
  }
}
```

- [ ] **Step 2: Verify**

```bash
node --check client-js/data-table.js
```

Expected: no output (syntax OK). Hand-trace: `setPoints([...8000 rows...])` then `selectRow('eng_Latn-5')` should set `_selectedId`/`_selectedSentenceId` to `'eng_Latn-5'` and page to `Math.floor(index/50)` where `index` is that row's position in `this.points` (no search/sort/filter active, so `_visibleRows()` returns `this.points` unchanged). Then `showTranslationsFor('pair:5')` should narrow `_visibleRows()` to just that pairId's rows and re-page — since a translation group is small, `_pageToRow` will find the selected row near the top of a very short list, landing on page 0.

- [ ] **Step 3: Commit**

```bash
git add client-js/data-table.js
git commit -m "Add DataTable component: sentence/token rows, search, sort, pagination, inline actions"
```

---

### Task 5: `app.js` — unify selection, wire the table, retire `DetailPanel`

**Files:**
- Modify: `client-js/app.js`
- Delete: `client-js/detail-panel.js`

**Interfaces:**
- Consumes: `DataTable` (Task 4), `ProjectionView.setSelectedId`/`clearSelectedId` (Task 2), `tokenTexts`/`pairId` on `/points` response points (Task 1, already present for `pairId`).
- Produces: no further consumers — this is the last task.

- [ ] **Step 1: Replace `client-js/app.js` in full**

```js
// Embedding Projection — app orchestrator. Owns selection state and wires
// together the projection view, backend client, data table, and language
// filter; each of those owns its own domain and App never reaches past
// their interfaces.

import { ProjectionView } from './projection-view.js';
import { BackendClient } from './backend-client.js';
import { DataTable } from './data-table.js';
import { LangFilter } from './lang-filter.js';

const BACKEND_URL = 'http://localhost:8500';

class App {
  constructor() {
    this.view = null;
    this.backend = new BackendClient(BACKEND_URL);
    this.table = new DataTable(
      { containerId: 'data-table' },
      {
        onRowClick: row => this.onSelectRow(row),
        onToggleTranslations: point => this.onToggleTranslations(point),
        onToggleTokens: point => this.onToggleTokens(point),
      }
    );
    this.langFilter = new LangFilter(
      { inputId: 'lang-search', suggestionsId: 'lang-suggestions', chipsId: 'lang-chips' },
      selected => this.view.setLangFilter(selected)
    );
    this.points = [];
    this._selectedPoint = null;
    this._translationsShown = false;
    this._tokenOverlayPointId = null;
  }

  init() {
    this.view = new ProjectionView('projection');
    this.view._onBackgroundClick = () => this.onBackgroundClick();

    document.getElementById('file-input')
      .addEventListener('change', e => this._onFileSelected(e.target.files[0]));
    document.getElementById('method-select')
      .addEventListener('change', () => this._refresh());
    document.getElementById('format-help-toggle')
      .addEventListener('click', () => this._toggleFormatHelp());

    this._refresh();
  }

  _toggleFormatHelp() {
    const el = document.getElementById('format-help');
    el.style.display = el.style.display === 'block' ? 'none' : 'block';
  }

  _onFileSelected(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async () => {
      let raw;
      try {
        raw = JSON.parse(reader.result);
      } catch (err) {
        document.getElementById('status').textContent = `Invalid file: ${err.message}`;
        return;
      }

      try {
        await this.backend.postDataset(raw);
      } catch (err) {
        document.getElementById('status').textContent = `Invalid file: ${err.message}`;
        return;
      }

      this._refresh();
    };
    reader.readAsText(file);
  }

  async _refresh() {
    const method = document.getElementById('method-select').value;
    document.getElementById('status').textContent = `Running ${method}...`;

    let meta, proj;
    try {
      [meta, proj] = await Promise.all([
        this.backend.fetchPoints(),
        this.backend.fetchProjection(method),
      ]);
    } catch (err) {
      document.getElementById('status').textContent = err.message;
      return;
    }

    this._loadDataset(meta, proj);
  }

  _loadDataset(meta, proj) {
    const hint = document.getElementById('projection-hint');
    if (hint) hint.remove();

    this._selectedPoint = null;
    this._translationsShown = false;
    this._tokenOverlayPointId = null;
    this.view.clearHighlight();
    this.view.clearSelectedId();
    this.view.clearTokenOverlay();

    const coordsById = new Map(proj.points.map(p => [p.id, p]));
    this.points = meta.points.map(p => {
      const c = coordsById.get(p.id);
      return {
        id: p.id, lang: p.lang, text: p.text, pairId: p.pairId,
        hasTokens: p.hasTokens, tokenTexts: p.tokenTexts,
        // no pairId at all -> each point is its own group, so it isolates alone
        groupKey: p.pairId != null ? `pair:${p.pairId}` : `id:${p.id}`,
        x: c.x, y: c.y,
      };
    });

    document.getElementById('status').textContent =
      `${meta.points.length} points · ${meta.lang_codes.length} languages · ${meta.model} (${meta.layer} layer) · ${proj.method}`;

    this.langFilter.setLangs(meta.lang_codes);
    this.table.setPoints(this.points);
    this.view.update(this.points, d => this.onCanvasClickPoint(d));
  }

  // Canvas points are always sentences — token overlay points aren't
  // clickable on canvas (see projection-view.js's _onCanvasClick).
  onCanvasClickPoint(d) {
    this._selectSentence(d);
  }

  onSelectRow(row) {
    if (row.type === 'token') {
      this.table.highlightRow(row.id);
      return;
    }
    this._selectSentence(row);
  }

  _selectSentence(point) {
    if (this._selectedPoint && this._selectedPoint.id === point.id) {
      this.onBackgroundClick();
      return;
    }
    this._clearSelectionSideEffects();
    this._selectedPoint = point;
    this.view.setSelectedId(point.id);
    this.table.selectRow(point.id);
  }

  onBackgroundClick() {
    if (this._selectedPoint === null) return;
    this._clearSelectionSideEffects();
    this._selectedPoint = null;
    this.view.clearSelectedId();
    this.table.clearSelection();
  }

  _clearSelectionSideEffects() {
    if (this._translationsShown) {
      this.view.clearHighlight();
      this.table.hideTranslations();
      this._translationsShown = false;
    }
    this._clearTokenOverlay();
  }

  onToggleTranslations(point) {
    if (this._translationsShown) {
      this.view.clearHighlight();
      this.table.hideTranslations();
      this._translationsShown = false;
    } else {
      this.view.highlightParallel(point.groupKey);
      this.table.showTranslationsFor(point.groupKey);
      this._translationsShown = true;
    }
  }

  // Toggles the token overlay for the currently selected point. Token
  // projection only exists in the shared UMAP space (see server.py's
  // _get_umap_reducer), so if the main view is on t-SNE this switches it to
  // UMAP first — re-running _refresh() resets selection state, so the
  // point is re-selected afterward before fetching its tokens. `point` is
  // captured as a whole object (not just its id) before the await, since
  // `_refresh()` replaces `this.points` with new objects but this
  // reference and its fields stay valid regardless.
  async onToggleTokens(point) {
    if (this._tokenOverlayPointId === point.id) {
      this._clearTokenOverlay();
      return;
    }

    const methodSelect = document.getElementById('method-select');
    if (methodSelect.value !== 'umap') {
      methodSelect.value = 'umap';
      await this._refresh();
      const reselected = this.points.find(p => p.id === point.id);
      if (!reselected) return;
      this._selectSentence(reselected);
      document.getElementById('status').textContent += ' · switched to UMAP to show tokens';
    }

    let proj;
    try {
      proj = await this.backend.fetchTokenProjection(point.id);
    } catch (err) {
      document.getElementById('status').textContent = err.message;
      return;
    }

    this._tokenOverlayPointId = point.id;
    const tokenPoints = proj.tokens.map((t, i) => ({
      id: `${point.id}::${i}`,
      lang: point.lang,
      text: t.token,
      x: t.x,
      y: t.y,
    }));
    this.view.setTokenOverlay(tokenPoints);
    this.table.showTokensFor(point.id, tokenPoints);
  }

  _clearTokenOverlay() {
    if (this._tokenOverlayPointId === null) return;
    this.table.hideTokensFor(this._tokenOverlayPointId);
    this._tokenOverlayPointId = null;
    this.view.clearTokenOverlay();
  }
}


// ── Entry point ──────────────────────────────────────────────────────────────
// type="module" scripts are deferred automatically, so the DOM is already
// parsed by the time this runs — no onload handler needed.

new App().init();
```

- [ ] **Step 2: Delete the retired detail panel**

```bash
git rm client-js/detail-panel.js
```

- [ ] **Step 3: Verify**

```bash
node --check client-js/app.js
grep -rn "detail-panel\|DetailPanel" client-js/ index.html
```

Expected: `node --check` prints nothing (syntax OK); the `grep` finds no remaining references anywhere.

Hand-trace the two scenarios most likely to break:
1. **Toggling tokens while selected point has `hasTokens: false`**: `DataTable._renderSentenceRow` only appends the tokens button `if (p.hasTokens)`, so `onToggleTokens` is never reachable for such a point — no crash path.
2. **Selecting a different sentence while translations are shown for the current one**: `_selectSentence` calls `_clearSelectionSideEffects()` (which calls `table.hideTranslations()` and `_clearTokenOverlay()`) *before* setting `this._selectedPoint` to the new point and calling `table.selectRow(newId)` — so the old point's group-filtered table view and canvas fade are both cleared before the new selection's `selectRow` re-pages against the now-restored full list.

- [ ] **Step 4: Commit**

```bash
git add client-js/app.js
git commit -m "Unify selection through the data table, retire DetailPanel"
```

---

### Task 6: Manual end-to-end verification

**Files:** none (verification only).

- [ ] **Step 1: Start the app**

From the repo root:
```bash
source .venv/bin/activate
python3 -m uvicorn server:app --port 8500 &
python3 -m http.server 8000 &
```
Open `http://localhost:8000`.

- [ ] **Step 2: Verify the table renders**

Confirm the data table pane appears between the sidebar and the plot, showing sentence rows with ID/Language/Text/PairId/Type/Token strings columns, a working search box, sortable column headers, and Prev/Next pagination through more than one page.

- [ ] **Step 3: Verify selection and inline actions**

Click a point on the canvas plot: confirm a blue ring appears around it, the table pages to and highlights the matching row, and that row shows "Show translations" (and "Show tokens" if it has tokens). Click a different row directly in the table: confirm the canvas ring moves to that point and the previous row's action buttons disappear.

- [ ] **Step 4: Verify translations toggle**

Click "Show translations" on a selected row: confirm the canvas fades everything except that translation group, and the table switches to show only that group's rows. Click "Hide translations": confirm the canvas un-fades and the table returns to the full paginated view.

- [ ] **Step 5: Verify tokens toggle and token rows**

Click "Show tokens" on a row that has them: confirm small diamonds appear on the canvas near that point, and the table inserts token rows (Type = "token", indented) directly beneath the parent row. Click a token row: confirm it highlights without showing any action buttons and without hiding the parent row's own action buttons. Click "Hide tokens": confirm the diamonds and token rows disappear.

- [ ] **Step 6: Verify the t-SNE auto-switch still works**

Switch the projection method dropdown to t-SNE, select a point, click "Show tokens": confirm the method dropdown auto-switches back to UMAP, the selection survives the reload, and tokens then appear.

- [ ] **Step 7: Verify background click clears everything**

With a point selected and either toggle active, click empty canvas space: confirm the ring, fade, overlay, and table selection all clear together.

- [ ] **Step 8: Report findings**

Note any deviations from the above in the task's report (this task has no code changes to commit, only observations) — surface them to the controller for triage rather than fixing silently.
