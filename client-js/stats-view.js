// Statistics/alignment panel: aggregate cosine statistics for whatever is
// pinned. Doesn't know about selection or the backend — App hands it either
// a /stats response, a loading state, or nothing. Everything rendered comes
// from uploaded datasets (language codes), so it only ever uses textContent.

const EMPTY_HINT = 'Pin points in the data table or on the plot to see aggregate statistics.';

const PAIR_COLUMNS = [
  { key: 'pair', label: 'Language pair', title: 'Languages being compared' },
  { key: 'n_parallel', label: 'Translations', title: 'Selected sentences that have a translation in both languages (same pairId)' },
  { key: 'parallel_cosine', label: 'Parallel cos', title: 'Mean cosine similarity of true translation pairs' },
  { key: 'non_parallel_cosine', label: 'Non-parallel cos', title: 'Mean cosine similarity of all other selected cross-language sentence pairs' },
  { key: 'gap', label: 'Gap', title: 'Parallel minus non-parallel cosine — how much closer translations sit than unrelated sentences' },
  { key: 'retrieval_accuracy', label: 'Retrieval acc.', title: 'Share of translations that are the mutual nearest neighbor of each other within the selection (MEXA-style, needs at least 2 translations)' },
];

function fmt(value) {
  if (value === null || value === undefined) return '–';
  const text = value.toFixed(3);
  return text === '-0.000' ? '0.000' : text;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export class StatsView {
  constructor({ containerId }) {
    this.container = document.getElementById(containerId);
    this._stats = null;
    this._sortKey = null;
    this._sortAsc = true;
    this.clear();
  }

  clear() {
    this._stats = null;
    this._showMessage(EMPTY_HINT);
  }

  setLoading() {
    this._showMessage('Computing statistics...');
  }

  setError(message) {
    this._showMessage(message);
  }

  setStats(stats) {
    this._stats = stats;
    this._render();
  }

  _showMessage(text) {
    this.container.replaceChildren(el('div', 'stats-message', text));
  }

  _rows() {
    const rows = this._stats.language_pairs.map(lp => ({
      ...lp,
      pair: `${lp.a} – ${lp.b}`,
      gap: lp.parallel_cosine !== null && lp.non_parallel_cosine !== null ? lp.parallel_cosine - lp.non_parallel_cosine : null,
    }));
    const key = this._sortKey;
    if (!key) return rows;
    // Missing values always sink to the bottom, whichever way it's sorted.
    return rows.sort((x, y) => {
      if (x[key] === null && y[key] === null) return 0;
      if (x[key] === null) return 1;
      if (y[key] === null) return -1;
      const v = x[key] < y[key] ? -1 : x[key] > y[key] ? 1 : 0;
      return this._sortAsc ? v : -v;
    });
  }

  _sortBy(key) {
    if (this._sortKey === key) this._sortAsc = !this._sortAsc;
    else { this._sortKey = key; this._sortAsc = true; }
    this._render();
  }

  _render() {
    const s = this._stats;
    const summaryGap = s.summary.parallel_cosine !== null && s.summary.non_parallel_cosine !== null
      ? s.summary.parallel_cosine - s.summary.non_parallel_cosine : null;
    const languageCounts = s.languages.map(l => `${l.lang}: ${l.count}`).join('\n');

    const cards = el('div', 'stats-cards');
    const addCard = (label, value, title) => {
      const card = el('div', 'stats-card');
      card.title = title;
      card.append(el('div', 'stats-card-value', value), el('div', 'stats-card-label', label));
      cards.appendChild(card);
    };
    addCard('Selected', `${s.n_selected} · ${s.n_languages} lang`, languageCounts);
    addCard('Parallel cos', fmt(s.summary.parallel_cosine), 'Mean cosine similarity of true translation pairs, averaged over language pairs');
    addCard('Non-parallel cos', fmt(s.summary.non_parallel_cosine), 'Mean cosine similarity of other cross-language sentence pairs, averaged over language pairs');
    addCard('Gap', fmt(summaryGap), 'Parallel minus non-parallel cosine');
    addCard('Retrieval acc.', fmt(s.summary.retrieval_accuracy), 'Average retrieval accuracy over language pairs with at least 2 translations');
    addCard('Within-lang cos', fmt(s.overall.within_language), 'Mean cosine similarity between selected sentences of the same language');
    addCard('Cross-lang cos', fmt(s.overall.cross_language), 'Mean cosine similarity between selected sentences of different languages');

    const wrap = el('div', 'stats-table-wrap');
    if (s.language_pairs.length === 0) {
      wrap.appendChild(el('div', 'stats-message', 'Pin sentences from at least two languages to compare language pairs.'));
    } else {
      const table = el('table', 'stats-table');
      const headRow = el('tr');
      for (const col of PAIR_COLUMNS) {
        const th = el('th', '', col.label + (this._sortKey === col.key ? (this._sortAsc ? ' ▲' : ' ▼') : ''));
        th.title = col.title;
        th.addEventListener('click', () => this._sortBy(col.key));
        headRow.appendChild(th);
      }
      table.appendChild(el('thead')).appendChild(headRow);
      const tbody = el('tbody');
      for (const row of this._rows()) {
        const tr = el('tr');
        tr.appendChild(el('td', '', row.pair));
        tr.appendChild(el('td', '', String(row.n_parallel)));
        for (const key of ['parallel_cosine', 'non_parallel_cosine', 'gap', 'retrieval_accuracy']) {
          tr.appendChild(el('td', '', fmt(row[key])));
        }
        tbody.appendChild(tr);
      }
      table.appendChild(tbody);
      wrap.appendChild(table);
    }

    this.container.replaceChildren(cards, wrap);
  }
}
