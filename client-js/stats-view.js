// Statistics/alignment panel: aggregate cosine statistics for whatever is
// pinned — a language or set of languages, all translations of a sentence,
// or the points in an area. Doesn't know about selection or the backend —
// App hands it either a /stats response, a loading state, or nothing.
// Everything rendered comes from uploaded datasets (language codes), so it
// only ever uses textContent.

const EMPTY_HINT = 'Pin points in the data table or on the plot to see aggregate statistics.';

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
    this.clear();
  }

  clear() {
    this._showMessage(EMPTY_HINT);
  }

  setLoading() {
    this._showMessage('Computing statistics...');
  }

  setError(message) {
    this._showMessage(message);
  }

  setStats(s) {
    const languageCounts = s.languages.map(l => `${l.lang}: ${l.count}`).join('\n');
    const cards = el('div', 'stats-cards');
    const addCard = (label, value, title) => {
      const card = el('div', 'stats-card');
      card.title = title;
      card.append(el('div', 'stats-card-value', value), el('div', 'stats-card-label', label));
      cards.appendChild(card);
    };
    addCard('Selected', `${s.n_selected} · ${s.n_languages} lang`, languageCounts);
    addCard('Within-lang cos', fmt(s.overall.within_language), 'Mean cosine similarity between selected sentences of the same language');
    addCard('Cross-lang cos', fmt(s.overall.cross_language), 'Mean cosine similarity between selected sentences of different languages');
    addCard('Parallel cos', fmt(s.summary.parallel_cosine), 'Mean cosine similarity of true translation pairs (same pairId), averaged over language pairs');
    addCard('Retrieval acc.', fmt(s.summary.retrieval_accuracy), 'Share of translations that are the mutual nearest neighbor of each other within the selection (MEXA-style); averaged over language pairs with at least 2 translations');
    this.container.replaceChildren(cards);
  }

  _showMessage(text) {
    this.container.replaceChildren(el('div', 'stats-message', text));
  }
}
