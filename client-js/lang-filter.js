// Searchable multi-select for choosing which languages get a real color.
// Replaces a flat legend, which stops being readable well before 200
// categories — type to filter the language list, click a suggestion to add
// it as a chip, click a chip's × to remove it. Reports the current selection
// via onChange(selectedLangs); doesn't know anything about points or canvas.

export class LangFilter {
  constructor({ inputId, suggestionsId, chipsId }, onChange) {
    this.input = document.getElementById(inputId);
    this.suggestionsEl = document.getElementById(suggestionsId);
    this.chipsEl = document.getElementById(chipsId);
    this._onChange = onChange;

    this.allLangs = [];
    this.selected = new Set();

    this.input.addEventListener('input', () => this._renderSuggestions());
    this.input.addEventListener('focus', () => this._renderSuggestions());
    // delay so a mousedown on a suggestion (which blurs the input first)
    // still has a chance to register before the list disappears
    this.input.addEventListener('blur', () => setTimeout(() => this._hideSuggestions(), 150));
  }

  // Called whenever a dataset loads — resets the selection (a filter from
  // the old dataset's language set wouldn't mean anything for the new one)
  // and refreshes what's searchable. After an edit of the same dataset,
  // keepSelection keeps the chosen languages that are still in it.
  setLangs(allLangs, { keepSelection = false } = {}) {
    this.allLangs = [...allLangs].sort();
    this.selected = keepSelection ? new Set([...this.selected].filter(lang => this.allLangs.includes(lang))) : new Set();
    this.input.value = '';
    this._renderChips();
    this._hideSuggestions();
  }

  getSelected() {
    return [...this.selected];
  }

  _renderSuggestions() {
    const query = this.input.value.trim().toLowerCase();
    const matches = this.allLangs
      .filter(lang => !this.selected.has(lang) && lang.toLowerCase().includes(query));

    this.suggestionsEl.innerHTML = '';
    if (matches.length === 0) {
      this._hideSuggestions();
      return;
    }
    this.suggestionsEl.style.display = 'block';
    matches.forEach(lang => {
      const item = document.createElement('div');
      item.className = 'lang-suggestion-item';
      item.textContent = lang;
      // mousedown (not click) + preventDefault so the input doesn't blur
      // and hide the list before the selection registers
      item.addEventListener('mousedown', event => {
        event.preventDefault();
        this._select(lang);
      });
      this.suggestionsEl.appendChild(item);
    });
  }

  _hideSuggestions() {
    this.suggestionsEl.style.display = 'none';
  }

  _select(lang) {
    this.selected.add(lang);
    this.input.value = '';
    this.input.focus();
    this._renderChips();
    this._renderSuggestions();
    this._onChange(this.getSelected());
  }

  _remove(lang) {
    this.selected.delete(lang);
    this._renderChips();
    this._onChange(this.getSelected());
  }

  _renderChips() {
    // Mirrors PointLayer's own d3.scaleOrdinal(selected, d3.schemeTableau10)
    // exactly, so a chip's color always matches its points' color. Kept
    // duplicated rather than shared — it's one line, not worth a coupling.
    const colorScale = d3.scaleOrdinal([...this.selected].sort(), d3.schemeTableau10);

    this.chipsEl.innerHTML = '';
    this.selected.forEach(lang => {
      const chip = document.createElement('div');
      chip.className = 'lang-chip';
      chip.style.background = colorScale(lang);
      chip.textContent = lang;

      const remove = document.createElement('span');
      remove.className = 'lang-chip-remove';
      remove.textContent = '×';
      remove.addEventListener('click', () => this._remove(lang));
      chip.appendChild(remove);

      this.chipsEl.appendChild(chip);
    });
  }
}
