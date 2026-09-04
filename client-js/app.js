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
