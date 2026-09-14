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
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export class DataTable {
  constructor({ containerId }, { onRowClick, onToggleTranslations, onToggleTokens, onToggleNeighbors }) {
    this.container = document.getElementById(containerId);
    this._onRowClick = onRowClick;
    this._onToggleTranslations = onToggleTranslations;
    this._onToggleTokens = onToggleTokens;
    this._onToggleNeighbors = onToggleNeighbors;

    this.points = [];
    this._pins = []; // Array<{point, color, focused, detail}>
    this._highlightedRowId = null; // token-row click only, independent of pins
    this._search = '';
    this._sortKey = null;
    this._sortAsc = true;
    this._page = 0;

    this.container.innerHTML = `
      <div class="dt-top">
        <input type="text" class="dt-search" placeholder="Search language or text...">
        <label class="dt-k-label">k: <input type="number" class="dt-neighbors-k" value="5" min="1" max="100"></label>
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
              <th>Distance</th>
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
    this._neighborsKInput = this.container.querySelector('.dt-neighbors-k');
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
    this._pins = [];
    this._highlightedRowId = null;
    this._search = '';
    this._searchInput.value = '';
    this._sortKey = null;
    this._page = 0;
    this._render();
  }

  // Current value of the "k" input — read by App when it's about to fetch
  // neighbors, so the count always reflects whatever the user last typed.
  neighborsK() {
    const n = parseInt(this._neighborsKInput.value, 10);
    return Number.isFinite(n) && n > 0 ? n : 5;
  }

  setNeighborsK(k) {
    this._neighborsKInput.value = k;
  }

  // Highlights a row without touching which sentence's actions are shown
  // — used for token row clicks, since a token has no actions of its own
  // and shouldn't hide its parent sentence's.
  highlightRow(id) {
    this._highlightedRowId = id;
    this._render();
  }

  // pins: Array<{point, color, focused, detail}> — the full current set,
  // every time. Pages/scrolls to whichever pin is focused so a fresh pin
  // (or an explicit focus change) is always visible; harmless no-op if
  // that row is already on-screen.
  setPins(pins) {
    this._pins = pins;
    const focused = pins.find(p => p.focused);
    if (focused) this._pageToRow(focused.point.id);
    this._render();
    this._scrollToFocused();
  }

  _pinFor(id) {
    return this._pins.find(p => p.point.id === id);
  }

  _translationsActivePin() {
    return this._pins.find(p => p.detail.translations) || null;
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
    const transPin = this._translationsActivePin();
    if (transPin) {
      rows = this.points.filter(p => p.groupKey === transPin.point.groupKey);
    } else {
      const q = this._search.trim().toLowerCase();
      rows = q
        ? this.points.filter(p => p.lang.toLowerCase().includes(q) || (p.text || '').toLowerCase().includes(q))
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
      const pin = this._pinFor(p.id);
      if (pin && pin.detail.tokens) {
        pin.detail.tokenPoints.forEach(t => this.tbody.appendChild(this._renderTokenRow(t)));
      }
      if (pin && pin.detail.neighbors) {
        pin.detail.neighborRows.forEach(n => this.tbody.appendChild(this._renderNeighborRow(n)));
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
      <td>${escapeHtml(String(p.id))}</td>
      <td>${escapeHtml(p.lang)}</td>
      <td class="dt-text" title="${escapeHtml(p.text)}">${escapeHtml(p.text)}</td>
      <td>${escapeHtml(String(p.pairId ?? ''))}</td>
      <td><span class="dt-type dt-type-sentence">sentence</span></td>
      <td class="dt-text">${p.hasTokens ? escapeHtml(p.tokenTexts.join(', ')) : ''}</td>
      <td></td>
      <td class="dt-actions-cell"></td>
    `;

    if (pin) {
      const actionsCell = tr.querySelector('.dt-actions-cell');
      const canTranslate = this._pins.length === 1;

      const transBtn = document.createElement('button');
      transBtn.type = 'button';
      transBtn.className = 'dt-action-btn';
      transBtn.disabled = !canTranslate;
      if (!canTranslate) transBtn.title = 'Only available with a single point selected';
      transBtn.textContent = pin.detail.translations ? 'Hide translations' : 'Show translations';
      transBtn.addEventListener('click', e => { e.stopPropagation(); this._onToggleTranslations(p); });
      actionsCell.appendChild(transBtn);

      if (p.hasTokens) {
        const tokBtn = document.createElement('button');
        tokBtn.type = 'button';
        tokBtn.className = 'dt-action-btn';
        tokBtn.textContent = pin.detail.tokens ? 'Hide tokens' : 'Show tokens';
        tokBtn.addEventListener('click', e => { e.stopPropagation(); this._onToggleTokens(p); });
        actionsCell.appendChild(tokBtn);
      }

      const neighBtn = document.createElement('button');
      neighBtn.type = 'button';
      neighBtn.className = 'dt-action-btn';
      neighBtn.textContent = pin.detail.neighbors ? 'Hide neighbors' : 'Show neighbors';
      neighBtn.addEventListener('click', e => { e.stopPropagation(); this._onToggleNeighbors(p); });
      actionsCell.appendChild(neighBtn);
    }

    tr.addEventListener('click', () => this._onRowClick({ ...p, type: 'sentence' }));
    return tr;
  }

  _renderTokenRow(t) {
    const tr = document.createElement('tr');
    const highlighted = t.id === this._highlightedRowId;
    tr.className = 'dt-row dt-row-token' + (highlighted ? ' dt-row-highlighted' : '');
    tr.innerHTML = `
      <td>${escapeHtml(t.id)}</td>
      <td>${escapeHtml(t.lang)}</td>
      <td class="dt-text" title="${escapeHtml(t.text)}">${escapeHtml(t.text)}</td>
      <td></td>
      <td><span class="dt-type dt-type-token">token</span></td>
      <td></td>
      <td></td>
      <td></td>
    `;
    tr.addEventListener('click', () => this._onRowClick({ ...t, type: 'token' }));
    return tr;
  }

  _renderNeighborRow(n) {
    const tr = document.createElement('tr');
    tr.className = 'dt-row dt-row-neighbor';
    tr.innerHTML = `
      <td>${escapeHtml(String(n.id))}</td>
      <td>${escapeHtml(n.lang)}</td>
      <td class="dt-text" title="${escapeHtml(n.text ?? '')}">${escapeHtml(n.text ?? '')}</td>
      <td></td>
      <td><span class="dt-type dt-type-neighbor">neighbor</span></td>
      <td></td>
      <td>${n.distance.toFixed(4)}</td>
      <td></td>
    `;
    tr.addEventListener('click', () => this._onRowClick({ ...n, type: 'neighbor' }));
    return tr;
  }
}
