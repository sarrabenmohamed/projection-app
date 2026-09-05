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
  // aren't clickable on canvas). A page holds up to PAGE_SIZE rows in a
  // fixed-height scroll strip, so paging alone doesn't guarantee the row
  // is in view — scroll it into view too, same as a canvas click "jumping"
  // to the row a user couldn't otherwise see.
  selectRow(id) {
    this._selectedId = id;
    this._selectedSentenceId = id;
    this._pageToRow(id);
    this._render();
    this._scrollToSelected();
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
    this._scrollToSelected();
  }

  hideTranslations() {
    this._translationsGroupKey = null;
    this._pageToRow(this._selectedSentenceId);
    this._render();
    this._scrollToSelected();
  }

  // Scrolls the currently-selected row into view within the table's
  // scroll strip, without jumping the page's own scroll position (only
  // the table's internal .dt-scroll container moves).
  _scrollToSelected() {
    const row = this.tbody.querySelector('tr.dt-row-selected');
    if (row) row.scrollIntoView({ block: 'nearest' });
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
      <td>${escapeHtml(String(p.pairId ?? ''))}</td>
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
