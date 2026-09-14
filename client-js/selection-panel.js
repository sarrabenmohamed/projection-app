// Selection panel: a standalone, always-visible overview of every
// currently pinned point, independent of the data table's search/sort/
// pagination (the pinned set is small by construction). Clicking a row
// changes focus only — it never unpins. This is the one place in the app
// where a click doesn't toggle pin membership: the Selection panel is the
// overview of what you're keeping, so a stray click here shouldn't cost
// you your working set. The explicit "×" button is the only way to unpin
// from this view.

import { renderActionsMenu } from './actions-menu.js';

export class SelectionPanel {
  constructor({ containerId }, { onFocusPin, onUnpin, onToggleTranslations, onToggleTokens, onToggleNeighbors, onChangeNeighborsK }) {
    this.container = document.getElementById(containerId);
    this._onFocusPin = onFocusPin;
    this._onUnpin = onUnpin;
    this._onToggleTranslations = onToggleTranslations;
    this._onToggleTokens = onToggleTokens;
    this._onToggleNeighbors = onToggleNeighbors;
    this._onChangeNeighborsK = onChangeNeighborsK;
    this._pins = [];

    this.container.innerHTML = '<div class="sel-list"></div>';
    this._listEl = this.container.querySelector('.sel-list');
  }

  setPins(pins) {
    this._pins = pins;
    this._render();
  }

  _render() {
    this._listEl.innerHTML = '';
    if (this._pins.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'sel-empty';
      empty.textContent = 'No points pinned — click a point on the plot or in the data table.';
      this._listEl.appendChild(empty);
      return;
    }

    const canTranslate = this._pins.length === 1;
    for (const pin of this._pins) {
      const row = document.createElement('div');
      row.className = 'sel-row' + (pin.focused ? ' sel-row-focused' : '');

      const swatch = document.createElement('span');
      swatch.className = 'sel-swatch';
      swatch.style.background = pin.color;
      row.appendChild(swatch);

      const idEl = document.createElement('span');
      idEl.className = 'sel-id';
      idEl.textContent = String(pin.point.id);
      row.appendChild(idEl);

      const langEl = document.createElement('span');
      langEl.className = 'sel-lang';
      langEl.textContent = pin.point.lang;
      row.appendChild(langEl);

      const textEl = document.createElement('span');
      textEl.className = 'sel-text';
      textEl.textContent = pin.point.text || '';
      textEl.title = pin.point.text || '';
      row.appendChild(textEl);

      row.appendChild(renderActionsMenu(pin.point, pin.detail, canTranslate, {
        onToggleTranslations: pt => this._onToggleTranslations(pt),
        onToggleTokens: pt => this._onToggleTokens(pt),
        onToggleNeighbors: pt => this._onToggleNeighbors(pt),
        onChangeK: (pt, k) => this._onChangeNeighborsK(pt, k),
      }, 'panel'));

      const removeBtn = document.createElement('button');
      removeBtn.type = 'button';
      removeBtn.className = 'sel-remove';
      removeBtn.textContent = '×';
      removeBtn.title = 'Unpin';
      removeBtn.addEventListener('click', e => { e.stopPropagation(); this._onUnpin(pin.point); });
      row.appendChild(removeBtn);

      row.addEventListener('click', () => this._onFocusPin(pin.point.id));
      this._listEl.appendChild(row);
    }
  }
}
