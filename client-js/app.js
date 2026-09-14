// Embedding Projection — app orchestrator. Owns selection state and wires
// together the projection view, backend client, data table, and language
// filter; each of those owns its own domain and App never reaches past
// their interfaces.

import { ProjectionView } from './projection-view.js';
import { BackendClient } from './backend-client.js';
import { DataTable } from './data-table.js';
import { LangFilter } from './lang-filter.js';
import { readUrlState, writeUrlState } from './url-state.js';

const BACKEND_URL = 'http://localhost:8500';
const PIN_PALETTE = d3.schemeTableau10;

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
    this._selection = [];           // ordered array of pinned ids, insertion order
    this._focusedId = null;
    this._detailState = new Map();  // id -> {translations, tokens, tokenPoints, neighbors, neighborsK, neighborRows}
    this._pinColors = new Map();    // id -> color string
    this._pendingSelection = null;  // still the old single-id shape here — Task 6 renames this to _pendingSelectionIds when it rewrites URL restore end-to-end
  }

  async init() {
    this.view = new ProjectionView('projection');
    this.view._onBackgroundClick = () => this.onBackgroundClick();

    // Restore shareable state from the URL (see url-state.js) before the
    // first fetch, so method/metric are right from the initial load and a
    // selection can be re-applied once points exist.
    const urlState = readUrlState();
    if (urlState.method) document.getElementById('method-select').value = urlState.method;
    if (urlState.metric) document.getElementById('metric-select').value = urlState.metric;
    if (urlState.labelBy) {
      document.getElementById('label-select').value = urlState.labelBy;
      this.view.setLabelBy(urlState.labelBy);
    }
    if (urlState.id) {
      this._pendingSelection = {
        id: urlState.id,
        translations: urlState.translations === '1',
        tokens: urlState.tokens === '1',
        neighborsK: urlState.neighbors,
      };
    }

    document.getElementById('file-input')
      .addEventListener('change', e => this._onFileSelected(e.target.files[0]));
    document.getElementById('method-select')
      .addEventListener('change', () => this._refresh());
    document.getElementById('metric-select')
      .addEventListener('change', () => this._refresh());
    document.getElementById('refit-button')
      .addEventListener('click', () => this._refresh());
    document.getElementById('label-select')
      .addEventListener('change', e => { this.view.setLabelBy(e.target.value); this._syncUrl(); });
    document.getElementById('format-help-toggle')
      .addEventListener('click', () => this._toggleFormatHelp());
    document.getElementById('projection-fullscreen-btn')
      .addEventListener('click', () => this._toggleFullscreen('projection-panel', 'data-table-panel'));
    document.getElementById('data-table-fullscreen-btn')
      .addEventListener('click', () => this._toggleFullscreen('data-table-panel', 'projection-panel'));
    document.getElementById('copy-link-button')
      .addEventListener('click', () => this._copyLink());
    this._initLayoutConfig();

    await this._refresh();
    await this._applyPendingSelection();
  }

  // Reflects the currently shareable state (method, metric, label-by, and
  // single-point selection) into the URL query string. Called at the end
  // of every state-changing action rather than continuously, so the URL
  // always matches what "Copy link" would hand someone else.
  _syncUrl() {
    const state = {
      method: document.getElementById('method-select').value,
      metric: document.getElementById('metric-select').value,
      labelBy: document.getElementById('label-select').value,
    };
    if (this._selectedPoint) {
      state.id = String(this._selectedPoint.id);
      if (this._translationsShown) state.translations = '1';
      if (this._tokenOverlayPointId === this._selectedPoint.id) state.tokens = '1';
      if (this._neighborsPointId === this._selectedPoint.id) state.neighbors = String(this.table.neighborsK());
    }
    writeUrlState(state);
  }

  // Re-applies a selection restored from the URL, once points from the
  // initial load are available. Only runs once — subsequent refreshes
  // (method/metric change, refit, upload) reset selection deliberately and
  // shouldn't resurrect a stale link's point.
  async _applyPendingSelection() {
    if (!this._pendingSelection) return;
    this._pendingSelection = null;
    // URL-based restore is rewritten for multi-pin selection in Task 6;
    // until then this deliberately does nothing rather than calling into
    // methods this task and Task 3 remove.
  }

  async _copyLink() {
    const feedback = document.getElementById('copy-link-feedback');
    try {
      await navigator.clipboard.writeText(window.location.href);
      feedback.textContent = 'Copied!';
    } catch {
      feedback.textContent = 'Copy failed';
    }
    setTimeout(() => { feedback.textContent = ''; }, 1500);
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

    this._selection = [];
    this._focusedId = null;
    this._detailState.clear();
    this._pinColors.clear();
    this.view.clearHighlight();
    this.view.setPinnedPoints([]);
    this.view.setTokenOverlays(new Map());

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
    this._syncUrl();
  }

  // Rebuilds every consumer's view of the pinned set from scratch each
  // time, rather than patching incrementally — simplest way to guarantee
  // canvas rings/overlays and the table's rendered actions never drift
  // from _selection/_detailState.
  _syncPins() {
    const pins = this._selection.map(id => ({
      point: this.points.find(p => p.id === id),
      color: this._pinColors.get(id),
      focused: id === this._focusedId,
      detail: this._detailState.get(id),
    }));

    this.view.setPinnedPoints(pins.map(p => ({ id: p.point.id, color: p.color, focused: p.focused })));

    const overlaysByPin = new Map();
    for (const p of pins) {
      if (p.detail.tokens && p.detail.tokenPoints.length) overlaysByPin.set(p.point.id, p.detail.tokenPoints);
    }
    this.view.setTokenOverlays(overlaysByPin);

    this.table.setPins(pins);
  }

  // Canvas points are always sentences — token overlay points aren't
  // clickable on canvas (see projection-view.js's _onCanvasClick).
  onCanvasClickPoint(d) {
    this._togglePin(d);
  }

  onSelectRow(row) {
    if (row.type === 'token') {
      this.table.highlightRow(row.id);
      return;
    }
    if (row.type === 'neighbor') {
      // Neighbor rows carry only {id, lang, text, distance} — jump to the
      // full point object so pin state (groupKey, hasTokens, ...) stays
      // consistent with canvas/table-driven pinning.
      const full = this.points.find(p => p.id === row.id);
      if (full) this._togglePin(full);
      return;
    }
    this._togglePin(row);
  }

  // Toggles `point`'s membership in the selection set. Pinning a *second*
  // point while the sole existing pin has translations shown auto-clears
  // that flag first — translations can only ever be active for exactly
  // one pinned point (see onToggleTranslations), so going from 1 -> 2
  // pins must never leave it active.
  _togglePin(point) {
    const idx = this._selection.indexOf(point.id);
    if (idx === -1) {
      if (this._selection.length === 1) {
        const soleDetail = this._detailState.get(this._selection[0]);
        if (soleDetail.translations) {
          soleDetail.translations = false;
          this.view.clearHighlight();
        }
      }
      this._pinColors.set(point.id, PIN_PALETTE[this._selection.length % PIN_PALETTE.length]);
      this._selection.push(point.id);
      this._detailState.set(point.id, {
        translations: false, tokens: false, tokenPoints: [],
        neighbors: false, neighborsK: 5, neighborRows: [],
      });
      this._focusedId = point.id;
    } else {
      this._selection.splice(idx, 1);
      this._detailState.delete(point.id);
      this._pinColors.delete(point.id);
      if (this._focusedId === point.id) {
        this._focusedId = this._selection.length ? this._selection[this._selection.length - 1] : null;
      }
    }
    this._syncPins();
    this._syncUrl();
  }

  onBackgroundClick() {
    if (this._selection.length === 0) return;
    this._selection = [];
    this._focusedId = null;
    this._detailState.clear();
    this._pinColors.clear();
    this.view.clearHighlight();
    this._syncPins();
    this._syncUrl();
  }

  // Translations only exist for exactly one pinned point at a time — see
  // SELECTION-STATE.md. Callers (the row's button/dropdown) are expected
  // to already be disabled when _selection.length !== 1, but this method
  // defends against being reached anyway rather than corrupting state.
  onToggleTranslations(point) {
    if (this._selection.length !== 1 || this._selection[0] !== point.id) return;
    const detail = this._detailState.get(point.id);
    detail.translations = !detail.translations;
    if (detail.translations) this.view.highlightParallel(point.groupKey);
    else this.view.clearHighlight();
    this._syncPins();
    this._syncUrl();
  }

  // Tokens are independent per pinned point. Token projection only exists
  // in the shared UMAP space (see server.py's _get_umap_reducer), so if
  // the main view is on t-SNE this switches it to UMAP first — that runs
  // _refresh(), which (like any _refresh()) wipes the entire selection —
  // and then explicitly re-pins just this one point afterward, the one
  // deliberate exception to "any _refresh() clears everything."
  async onToggleTokens(point) {
    const detail = this._detailState.get(point.id);
    if (!detail) return;
    if (detail.tokens) {
      detail.tokens = false;
      detail.tokenPoints = [];
      this._syncPins();
      this._syncUrl();
      return;
    }

    const methodSelect = document.getElementById('method-select');
    if (methodSelect.value !== 'umap') {
      methodSelect.value = 'umap';
      await this._refresh();
      const reselected = this.points.find(p => p.id === point.id);
      if (!reselected) return;
      this._togglePin(reselected);
      point = reselected;
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

    const liveDetail = this._detailState.get(point.id);
    if (!liveDetail) return; // unpinned while the request was in flight
    liveDetail.tokens = true;
    liveDetail.tokenPoints = proj.tokens.map((t, i) => ({
      id: `${point.id}::${i}`, lang: point.lang, text: t.token, x: t.x, y: t.y,
    }));
    this._syncPins();
    this._syncUrl();
  }

  // Neighbors are independent per pinned point. Computed on the full
  // embedding vectors (server.py's /neighbors), so unlike tokens this
  // never needs to force a UMAP switch.
  async onToggleNeighbors(point) {
    const detail = this._detailState.get(point.id);
    if (!detail) return;
    if (detail.neighbors) {
      detail.neighbors = false;
      detail.neighborRows = [];
      this._syncPins();
      this._syncUrl();
      return;
    }

    const metric = document.getElementById('metric-select').value;
    const k = this.table.neighborsK();
    detail.neighborsK = k;
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

    const liveDetail = this._detailState.get(point.id);
    if (!liveDetail) return;
    liveDetail.neighbors = true;
    liveDetail.neighborRows = res.neighbors;
    this._syncPins();
    this._syncUrl();
  }
}


// ── Entry point ──────────────────────────────────────────────────────────────
// type="module" scripts are deferred automatically, so the DOM is already
// parsed by the time this runs — no onload handler needed.

new App().init();
