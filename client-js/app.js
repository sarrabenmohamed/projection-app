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
        onToggleNeighbors: point => this.onToggleNeighbors(point),
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
    this._neighborsPointId = null;
  }

  init() {
    this.view = new ProjectionView('projection');
    this.view._onBackgroundClick = () => this.onBackgroundClick();

    document.getElementById('file-input')
      .addEventListener('change', e => this._onFileSelected(e.target.files[0]));
    document.getElementById('method-select')
      .addEventListener('change', () => this._refresh());
    document.getElementById('metric-select')
      .addEventListener('change', () => this._refresh());
    document.getElementById('refit-button')
      .addEventListener('click', () => this._refresh());
    document.getElementById('label-select')
      .addEventListener('change', e => this.view.setLabelBy(e.target.value));
    document.getElementById('format-help-toggle')
      .addEventListener('click', () => this._toggleFormatHelp());
    document.getElementById('projection-fullscreen-btn')
      .addEventListener('click', () => this._toggleFullscreen('projection-panel', 'data-table-panel'));
    document.getElementById('data-table-fullscreen-btn')
      .addEventListener('click', () => this._toggleFullscreen('data-table-panel', 'projection-panel'));
    this._initLayoutConfig();

    this._refresh();
  }

  // Layout config panel: lets the sidebar/projection/table each be hidden
  // independently (unlike fullscreen, which always keeps exactly one view
  // visible), persisted across reloads via localStorage. Never allows both
  // the projection and data-table panels to be hidden at once — that would
  // leave #main empty with no way to get either back.
  _initLayoutConfig() {
    const STORAGE_KEY = 'projection-app-layout';
    const panelIdFor = { sidebar: 'sidebar', projection: 'projection-panel', dataTable: 'data-table-panel' };
    const checkboxIdFor = { sidebar: 'toggle-sidebar', projection: 'toggle-projection', dataTable: 'toggle-data-table' };

    let saved = {};
    try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'); } catch { saved = {}; }
    for (const key of Object.keys(panelIdFor)) {
      const visible = saved[key] !== false;
      document.getElementById(panelIdFor[key]).classList.toggle('hidden', !visible);
      document.getElementById(checkboxIdFor[key]).checked = visible;
    }

    document.getElementById('layout-config-toggle').addEventListener('click', () => {
      document.getElementById('layout-config-panel').classList.toggle('visible');
    });

    for (const key of Object.keys(checkboxIdFor)) {
      document.getElementById(checkboxIdFor[key]).addEventListener('change', e => {
        const wantVisible = e.target.checked;
        if (!wantVisible && key !== 'sidebar') {
          const otherKey = key === 'projection' ? 'dataTable' : 'projection';
          if (!document.getElementById(checkboxIdFor[otherKey]).checked) {
            e.target.checked = true; // refuse to hide both view panels at once
            return;
          }
        }

        const panel = document.getElementById(panelIdFor[key]);
        panel.classList.toggle('hidden', !wantVisible);
        panel.classList.remove('fullscreen'); // avoid a stale fullscreen+hidden combo

        const current = {};
        for (const k of Object.keys(checkboxIdFor)) current[k] = document.getElementById(checkboxIdFor[k]).checked;
        localStorage.setItem(STORAGE_KEY, JSON.stringify(current));

        if (key === 'projection' && this.view) this.view.resize();
      });
    }
  }

  // Toggles `panelId` to fill #main (hiding `otherPanelId`), or restores
  // the normal split if `panelId` is already fullscreen. The projection
  // view's canvas has real pixel dimensions tied to its container, so it
  // needs an explicit resize() after any layout change; the data table is
  // plain flexed HTML and resizes itself via CSS alone.
  _toggleFullscreen(panelId, otherPanelId) {
    const panel = document.getElementById(panelId);
    const other = document.getElementById(otherPanelId);
    const turningOn = !panel.classList.contains('fullscreen');

    panel.classList.toggle('fullscreen', turningOn);
    other.classList.toggle('hidden', turningOn);
    if (turningOn) other.classList.remove('fullscreen');

    this.view.resize();
  }

  _toggleFormatHelp() {
    const el = document.getElementById('format-help');
    el.style.display = el.style.display === 'block' ? 'none' : 'block';
  }

  // Shows/hides the full-screen loading overlay used around every backend
  // call — a visible spinner instead of just the (still-updated) status
  // text, since a UMAP/t-SNE refit or a token projection can take a while.
  _setLoading(isLoading, message) {
    document.getElementById('loading-overlay').classList.toggle('visible', isLoading);
    if (isLoading) document.getElementById('loading-text').textContent = message;
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

      this._setLoading(true, 'Uploading dataset...');
      try {
        await this.backend.postDataset(raw);
      } catch (err) {
        this._setLoading(false);
        document.getElementById('status').textContent = `Invalid file: ${err.message}`;
        return;
      }
      this._setLoading(false);

      this._refresh();
    };
    reader.readAsText(file);
  }

  async _refresh() {
    const method = document.getElementById('method-select').value;
    const metric = document.getElementById('metric-select').value;
    document.getElementById('status').textContent = `Running ${method} (${metric})...`;
    this._setLoading(true, `Running ${method} (${metric})...`);

    let meta, proj;
    try {
      [meta, proj] = await Promise.all([
        this.backend.fetchPoints(),
        this.backend.fetchProjection(method, metric),
      ]);
    } catch (err) {
      this._setLoading(false);
      document.getElementById('status').textContent = err.message;
      return;
    }

    this._setLoading(false);
    this._loadDataset(meta, proj);
  }

  _loadDataset(meta, proj) {
    const hint = document.getElementById('projection-hint');
    if (hint) hint.remove();

    this._selectedPoint = null;
    this._translationsShown = false;
    this._tokenOverlayPointId = null;
    this._neighborsPointId = null;
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
      `${meta.points.length} points · ${meta.lang_codes.length} languages · ${meta.model} (${meta.layer} layer, ${meta.pooling}-pooled) · ${proj.method}`;

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
    if (row.type === 'neighbor') {
      // Neighbor rows carry only {id, lang, text, distance} — jump to the
      // full point object so selection state (groupKey, hasTokens, ...)
      // stays consistent with canvas/table-driven selection.
      const full = this.points.find(p => p.id === row.id);
      if (full) this._selectSentence(full);
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
    this._clearNeighbors();
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

    const metric = document.getElementById('metric-select').value;
    this._setLoading(true, 'Computing token projection...');
    let proj;
    try {
      proj = await this.backend.fetchTokenProjection(point.id, metric);
    } catch (err) {
      this._setLoading(false);
      document.getElementById('status').textContent = err.message;
      return;
    }
    this._setLoading(false);

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

  // Nearest neighbors are computed on the full embedding vectors (see
  // server.py's /neighbors), independent of projection method/metric, so
  // unlike tokens this never needs to force a UMAP switch.
  async onToggleNeighbors(point) {
    if (this._neighborsPointId === point.id) {
      this._clearNeighbors();
      return;
    }

    const metric = document.getElementById('metric-select').value;
    const k = this.table.neighborsK();
    this._setLoading(true, 'Finding nearest neighbors...');
    let res;
    try {
      res = await this.backend.fetchNeighbors(point.id, k, metric);
    } catch (err) {
      this._setLoading(false);
      document.getElementById('status').textContent = err.message;
      return;
    }
    this._setLoading(false);

    this._neighborsPointId = point.id;
    this.table.showNeighborsFor(point.id, res.neighbors);
  }

  _clearNeighbors() {
    if (this._neighborsPointId === null) return;
    this.table.hideNeighborsFor(this._neighborsPointId);
    this._neighborsPointId = null;
  }
}


// ── Entry point ──────────────────────────────────────────────────────────────
// type="module" scripts are deferred automatically, so the DOM is already
// parsed by the time this runs — no onload handler needed.

new App().init();
