// Data table: sentence rows always, token rows only for whichever point
// currently has its tokens shown. A pin checkbox is each sentence row's
// only pin control (clicking elsewhere on the row does nothing); a pinned
// row reveals its Actions dropdown. App drives all of this through the
// same selection state it drives the canvas with, so table and canvas
// pinning always agree.

import { renderActionsMenu } from './actions-menu.js';

const PAGE_SIZE = 50;
// id/pairId are identifiers, not free text — a substring match on them would
// be misleading (filtering pairId "1" would also match "10", "11", ...), so
// these two require exact equality; lang/text stay substring.
const EXACT_MATCH_FILTER_KEYS = new Set(['id', 'pairId']);
// Lets the single search box target one column ("id=eng_Latn-3",
// "pairid=5") instead of guessing — aliases are lowercase since the key is
// matched case-insensitively.
const COLUMN_ALIASES = { id: 'id', lang: 'lang', language: 'lang', text: 'text', pairid: 'pairId', pair_id: 'pairId' };

// A bare query (no "key=") searches lang+text, same as the original search
// box; "key=value" targets just that column instead.
function parseSearchQuery(raw) {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const eq = trimmed.indexOf('=');
  if (eq > 0) {
    const key = COLUMN_ALIASES[trimmed.slice(0, eq).trim().toLowerCase()];
    const value = trimmed.slice(eq + 1).trim();
    if (key && value) return { key, value };
  }
  return { key: null, value: trimmed };
}

// Sentence text, language, and token strings all come from whatever
// dataset JSON was uploaded via POST /dataset — untrusted input, not
// baked into the app — so anything rendered into a cell goes through this
// before it touches innerHTML.
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export class DataTable {
  constructor({ containerId }, { onRowClick, onSelectTranslations, onToggleTokens, onToggleShowSelectedOnly, onClearSelection, onSelectAll }) {
    this.container = document.getElementById(containerId);
    this._onRowClick = onRowClick;
    this._onSelectTranslations = onSelectTranslations;
    this._onToggleTokens = onToggleTokens;
    this._onToggleShowSelectedOnly = onToggleShowSelectedOnly;
    this._onClearSelection = onClearSelection;
    this._onSelectAll = onSelectAll;

    this.points = [];
    this._pins = []; // Array<{point, color, focused, detail}>
    this._groupSizes = new Map();
    this._highlightedRowId = null; // token-row click only, independent of pins
    // Ephemeral — reset on every setPoints() and never persisted or synced
    // to the canvas: this is just a quick local view narrowing, not a
    // selection, so it shouldn't outlive the table.
    this._search = '';
    this._sortKey = null;
    this._sortAsc = true;
    this._page = 0;
    this._showSelectedOnly = false;

    this.container.innerHTML = `
      <div class="dt-top">
        <input type="text" class="dt-search" placeholder="Search, or id=/lang=/text=/pairid=...">
        <label class="dt-show-selected-label"><input type="checkbox" class="dt-show-selected"> Show selected</label>
        <button type="button" class="dt-select-all" title="Pin every row matching the current search, across all pages">Select all</button>
        <button type="button" class="dt-clear-selection">Clear selection</button>
        <span class="dt-count"></span>
      </div>
      <div class="dt-scroll">
        <table>
          <thead>
            <tr>
              <th class="dt-pin-col"></th>
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
    this.container.querySelector('.dt-show-selected').addEventListener('change', e => {
      this._showSelectedOnly = e.target.checked;
      this._page = 0;
      this._render();
      this._onToggleShowSelectedOnly(e.target.checked);
    });
    this.container.querySelector('.dt-select-all').addEventListener('click', () => this._onSelectAll(this._visibleRows()));
    this.container.querySelector('.dt-clear-selection').addEventListener('click', () => this._onClearSelection());
    this.container.querySelector('.dt-clear-selection').disabled = true;
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
  _indexGroups(points) {
    this._groupSizes = new Map();
    for (const p of points) this._groupSizes.set(p.groupKey, (this._groupSizes.get(p.groupKey) || 0) + 1);
  }

  setPoints(points) {
    this.points = points;
    this._indexGroups(points);
    this._pins = [];
    this.container.querySelector('.dt-clear-selection').disabled = true;
    this._highlightedRowId = null;
    this._search = '';
    this._searchInput.value = '';
    this._sortKey = null;
    this._page = 0;
    this._showSelectedOnly = false;
    this.container.querySelector('.dt-show-selected').checked = false;
    this._render();
  }

  // Same dataset, new coordinates: swap the rows but keep pins, search,
  // sort and page.
  updatePoints(points) {
    this.points = points;
    this._indexGroups(points);
    this._render();
  }

  // Highlights a row without touching which sentence's actions are shown
  // — used for token row clicks, since a token has no actions of its own
  // and shouldn't hide its parent sentence's.
  highlightRow(id) {
    this._highlightedRowId = id;
    this._render();
  }

  isShowSelectedOnly() {
    return this._showSelectedOnly;
  }

  // Lets App programmatically override the checkbox (e.g. auto-turning it
  // back off if the pinned set becomes empty while it's on — see
  // App._syncPins) without going through _onToggleShowSelectedOnly again,
  // which would just call back into the same App method.
  setShowSelectedOnly(enabled) {
    this._showSelectedOnly = enabled;
    this.container.querySelector('.dt-show-selected').checked = enabled;
    this._page = 0;
    this._render();
  }

  // pins: Array<{point, color, focused, detail}> — the full current set,
  // every time. Pages/scrolls to whichever pin is focused so a fresh pin
  // (or an explicit focus change) is always visible; harmless no-op if
  // that row is already on-screen.
  setPins(pins) {
    this._pins = pins;
    this.container.querySelector('.dt-clear-selection').disabled = pins.length === 0;
    const focused = pins.find(p => p.focused);
    if (focused) this._pageToRow(focused.point.id);
    this._render();
    this._scrollToFocused();
  }

  _pinFor(id) {
    return this._pins.find(p => p.point.id === id);
  }

  _scrollToFocused() {
    const row = this.tbody.querySelector('tr.dt-row-focused');
    if (row) row.scrollIntoView({ block: 'nearest' });
  }

  _pageToRow(id) {
    if (id === null) return;
    const rows = this._visibleRows();
    const index = rows.findIndex(r => r.id === id);
    if (index >= 0) this._page = Math.floor(index / PAGE_SIZE);
  }

  _visibleRows() {
    let rows;
    const parsed = parseSearchQuery(this._search);
    if (!parsed) {
      rows = this.points;
    } else if (parsed.key) {
      const q = parsed.value.toLowerCase();
      rows = EXACT_MATCH_FILTER_KEYS.has(parsed.key)
        ? this.points.filter(p => String(p[parsed.key] ?? '').toLowerCase() === q)
        : this.points.filter(p => String(p[parsed.key] ?? '').toLowerCase().includes(q));
    } else {
      const q = parsed.value.toLowerCase();
      rows = this.points.filter(p => p.lang.toLowerCase().includes(q) || (p.text || '').toLowerCase().includes(q));
    }
    if (this._showSelectedOnly) {
      const pinnedIds = new Set(this._pins.map(p => p.point.id));
      rows = rows.filter(p => pinnedIds.has(p.id));
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
    this.container.querySelector('.dt-select-all').disabled = total === 0;
    this._pageEl.textContent = `Page ${this._page + 1} / ${totalPages}`;

    this.tbody.innerHTML = '';
    pageRows.forEach(p => {
      this.tbody.appendChild(this._renderSentenceRow(p));
      const pin = this._pinFor(p.id);
      if (pin && pin.detail.tokens) {
        pin.detail.tokenPoints.forEach(t => this.tbody.appendChild(this._renderTokenRow(t)));
      }
    });
  }

  _renderSentenceRow(p) {
    const tr = document.createElement('tr');
    const pin = this._pinFor(p.id);
    const highlighted = p.id === this._highlightedRowId;
    tr.className = 'dt-row'
      + (highlighted ? ' dt-row-highlighted' : '')
      + (pin ? ' dt-row-pinned' : '')
      + (pin && pin.focused ? ' dt-row-focused' : '');
    if (pin) tr.style.setProperty('--pin-color', pin.color);

    tr.innerHTML = `
      <td class="dt-pin-col"></td>
      <td>${escapeHtml(String(p.id))}</td>
      <td>${escapeHtml(p.lang)}</td>
      <td class="dt-text" title="${escapeHtml(p.text)}">${escapeHtml(p.text)}</td>
      <td>${escapeHtml(String(p.pairId ?? ''))}</td>
      <td><span class="dt-type dt-type-sentence">sentence</span></td>
      <td class="dt-text">${p.hasTokens ? escapeHtml(p.tokenTexts.join(', ')) : ''}</td>
      <td class="dt-actions-cell"></td>
    `;

    const pinCheckbox = document.createElement('input');
    pinCheckbox.type = 'checkbox';
    pinCheckbox.className = 'dt-pin-checkbox';
    pinCheckbox.checked = !!pin;
    pinCheckbox.title = pin ? 'Unpin' : 'Pin';
    pinCheckbox.addEventListener('change', e => { e.stopPropagation(); this._onRowClick({ ...p, type: 'sentence' }); });
    tr.querySelector('.dt-pin-col').appendChild(pinCheckbox);

    // The whole row toggles the pin, not just the checkbox — except clicks on
    // the checkbox itself / the Actions menu, and drags that selected text.
    tr.addEventListener('click', e => {
      if (e.target.closest('input, button, .actions-menu')) return;
      if (window.getSelection().toString()) return;
      this._onRowClick({ ...p, type: 'sentence' });
    });

    if (pin) {
      const actionsCell = tr.querySelector('.dt-actions-cell');
      const canTranslate = (this._groupSizes.get(p.groupKey) || 0) > 1;
      actionsCell.appendChild(renderActionsMenu(p, pin.detail, canTranslate, {
        onSelectTranslations: pt => this._onSelectTranslations(pt),
        onToggleTokens: pt => this._onToggleTokens(pt),
      }, 'table'));
    }

    return tr;
  }

  _renderTokenRow(t) {
    const tr = document.createElement('tr');
    const highlighted = t.id === this._highlightedRowId;
    tr.className = 'dt-row dt-row-token' + (highlighted ? ' dt-row-highlighted' : '');
    tr.innerHTML = `
      <td></td>
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
