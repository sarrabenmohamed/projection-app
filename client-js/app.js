// Embedding Projection — app orchestrator. Owns selection state and wires
// together the projection view, backend client, and detail panel; each of
// those owns its own domain and App never reaches past their interfaces.

import { ProjectionView } from './projection-view.js';
import { BackendClient } from './backend-client.js';
import { DetailPanel } from './detail-panel.js';
import { LangFilter } from './lang-filter.js';

const BACKEND_URL = 'http://localhost:8500';

class App {
  constructor() {
    this.view = null;
    this.backend = new BackendClient(BACKEND_URL);
    this.detail = new DetailPanel({
      placeholderId: 'detail-placeholder',
      langId: 'detail-lang',
      textId: 'detail-text',
      translationsId: 'detail-translations',
    });
    this.langFilter = new LangFilter(
      { inputId: 'lang-search', suggestionsId: 'lang-suggestions', chipsId: 'lang-chips' },
      selected => this.view.setLangFilter(selected)
    );
    this.points = [];
    this._selectedGroupKey = null;
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

    this._selectedGroupKey = null;
    this.view.clearHighlight();
    this.detail.clear();

    const coordsById = new Map(proj.points.map(p => [p.id, p]));
    this.points = meta.points.map(p => {
      const c = coordsById.get(p.id);
      return {
        id: p.id, lang: p.lang, text: p.text,
        // no pairId at all -> each point is its own group, so it isolates alone on click
        groupKey: p.pairId != null ? `pair:${p.pairId}` : `id:${p.id}`,
        x: c.x, y: c.y,
      };
    });

    document.getElementById('status').textContent =
      `${meta.points.length} points · ${meta.lang_codes.length} languages · ${meta.model} (${meta.layer} layer) · ${proj.method}`;

    this.langFilter.setLangs(meta.lang_codes);
    this.view.update(this.points, d => this.onClickPoint(d));
  }

  onClickPoint(d) {
    if (this._selectedGroupKey === d.groupKey) {
      this.onBackgroundClick();
      return;
    }
    this._selectedGroupKey = d.groupKey;
    this.view.highlightParallel(d.groupKey);
    const translations = this.points.filter(p => p.groupKey === d.groupKey && p.id !== d.id);
    this.detail.show(d, translations);
  }

  onBackgroundClick() {
    if (this._selectedGroupKey === null) return;
    this._selectedGroupKey = null;
    this.view.clearHighlight();
    this.detail.clear();
  }
}


// ── Entry point ──────────────────────────────────────────────────────────────
// type="module" scripts are deferred automatically, so the DOM is already
// parsed by the time this runs — no onload handler needed.

new App().init();
