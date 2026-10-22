// Embedding Projection — app orchestrator. Owns selection state and wires
// together the projection view, backend client, data table, and language
// filter; each of those owns its own domain and App never reaches past
// their interfaces.

import { ProjectionView } from './projection-view.js';
import { BackendClient } from './backend-client.js';
import { DataTable } from './data-table.js';
import { LangFilter } from './lang-filter.js';
import { installTooltips } from './help-tips.js';
import { StatsView } from './stats-view.js';
import { installActionsMenuOutsideClickHandler } from './actions-menu.js';

const PANEL_IDS = ['sidebar', 'projection-panel', 'data-table-panel', 'stats-panel'];

// Same origin by default (server.py serves this page); ?backend=http://host:port overrides it.
const BACKEND_URL = new URLSearchParams(location.search).get('backend') ?? '';
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
        onToggleShowSelectedOnly: enabled => this._setShowSelectedOnly(enabled),
        onClearSelection: () => this.clearSelection(),
        onSelectAll: points => this.selectPoints(points),
      }
    );
    this.stats = new StatsView({ containerId: 'stats' });
    this.langFilter = new LangFilter(
      { inputId: 'lang-search', suggestionsId: 'lang-suggestions', chipsId: 'lang-chips' },
      selected => this.view.setLangFilter(selected)
    );
    this.points = [];
    this._selection = [];           // ordered array of pinned ids, insertion order
    this._focusedId = null;
    this._detailState = new Map();  // id -> {translations, tokens, tokenPoints}
    this._pinColors = new Map();    // id -> color string
    this._statsRequest = 0;         // bumps on every selection change so a stale /stats response is dropped
    this._statsTimer = null;
  }

  async init() {
    this.view = new ProjectionView('projection');

    document.getElementById('file-input')
      .addEventListener('change', e => this._onFileSelected(e.target.files[0]));
    document.getElementById('flores-load-button')
      .addEventListener('click', () => this._onLoadFlores());
    document.getElementById('method-select')
      .addEventListener('change', () => { this._syncParamVisibility(); this._refresh(); });
    this._syncParamVisibility();
    document.getElementById('metric-select')
      .addEventListener('change', () => this._refresh());
    // n_neighbors/min_dist/perplexity/learning_rate deliberately do NOT
    // auto-refresh on change — editing more than one before blurring away
    // would fire a _refresh() per field, and an in-flight response's
    // "reflect the effective params back" step (see _loadDataset) could
    // then clobber a value the user typed into another field afterward.
    // Refit is the one explicit trigger for all of them.
    document.getElementById('refit-button')
      .addEventListener('click', () => this._refresh());
    document.getElementById('label-select')
      .addEventListener('change', e => this.view.setLabelBy(e.target.value));
    document.getElementById('fisheye-toggle')
      .addEventListener('change', e => this._setFisheye(e.target.checked));
    document.getElementById('fisheye-size')
      .addEventListener('input', e => {
        document.getElementById('fisheye-size-value').textContent = e.target.value;
        this.view.setFisheyeRadius(Number(e.target.value));
      });
    document.getElementById('fisheye-style-select')
      .addEventListener('change', e => this.view.setFisheyeStyle(e.target.value));
    document.getElementById('family-color-toggle')
      .addEventListener('change', e => {
        this.view.setFamilyColorMode(e.target.checked);
        document.getElementById('lang-search').disabled = e.target.checked;
        this._renderFamilyLegend();
      });
    document.getElementById('lang-search').disabled = document.getElementById('family-color-toggle').checked;
    document.getElementById('format-help-toggle')
      .addEventListener('click', () => this._toggleFormatHelp());
    document.getElementById('projection-caveats-toggle')
      .addEventListener('click', () => this._toggleProjectionCaveats());
    document.getElementById('projection-fullscreen-btn')
      .addEventListener('click', () => this._toggleFullscreen('projection-panel'));
    document.getElementById('data-table-fullscreen-btn')
      .addEventListener('click', () => this._toggleFullscreen('data-table-panel'));
    document.getElementById('stats-fullscreen-btn')
      .addEventListener('click', () => this._toggleFullscreen('stats-panel'));
    installTooltips();
    this._initPanels();
    this._initResizeHandle();
    installActionsMenuOutsideClickHandler();

    await this._refresh();
  }

  // UMAP and t-SNE take disjoint hyperparameters, so only the selected
  // method's fields are shown.
  _syncParamVisibility() {
    const umap = document.getElementById('method-select').value === 'umap';
    document.getElementById('umap-params').hidden = !umap;
    document.getElementById('tsne-params').hidden = umap;
  }

  _umapParams() {
    return {
      n_neighbors: document.getElementById('umap-n-neighbors').value,
      min_dist: document.getElementById('umap-min-dist').value,
    };
  }

  _tsneParams() {
    return {
      perplexity: document.getElementById('tsne-perplexity').value,
      learning_rate: document.getElementById('tsne-learning-rate').value,
    };
  }

  // The lens size/style controls only matter while the lens is on.
  _setFisheye(enabled) {
    this.view.setFisheyeEnabled(enabled);
    document.getElementById('fisheye-options').hidden = !enabled;
  }

  // Every view (sidebar, projection, data table, statistics) can be
  // collapsed to just its header bar; the choice persists in localStorage.
  // Collapsing never removes the header, so there is always a way back.
  _initPanels() {
    const STORAGE_KEY = 'projection-app-collapsed';
    let saved = [];
    try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); } catch { saved = []; }
    this._collapsed = new Set(Array.isArray(saved) ? saved.filter(id => PANEL_IDS.includes(id)) : []);
    this._saveCollapsed = () => localStorage.setItem(STORAGE_KEY, JSON.stringify([...this._collapsed]));

    for (const btn of document.querySelectorAll('[data-collapse]')) {
      btn.addEventListener('click', () => {
        const id = btn.dataset.collapse;
        if (this._collapsed.has(id)) this._collapsed.delete(id); else this._collapsed.add(id);
        this._saveCollapsed();
        this._applyLayout();
      });
    }
    this._applyLayout();
  }

  // Single place that turns collapsed/hidden state into classes: the
  // projection fills spare space when expanded, otherwise the topmost
  // expanded view does, so a collapsed panel never leaves a blank gap.
  _applyLayout() {
    const isOpen = id => !this._collapsed.has(id) && !document.getElementById(id).classList.contains('hidden');
    for (const id of PANEL_IDS) {
      const collapsed = this._collapsed.has(id);
      document.getElementById(id).classList.toggle('collapsed', collapsed);
      const btn = document.querySelector(`[data-collapse="${id}"]`);
      btn.textContent = collapsed ? '▸' : (id === 'sidebar' ? '◂' : '▾');
      btn.title = collapsed ? 'Expand' : 'Collapse';
      btn.setAttribute('aria-expanded', String(!collapsed));
    }
    const viewIds = ['projection-panel', 'data-table-panel', 'stats-panel'];
    const growId = viewIds.find(isOpen);
    for (const id of viewIds) {
      document.getElementById(id).classList.toggle('grow', id === growId && id !== 'projection-panel');
    }
    this._updateResizeHandleVisibility();
    if (this.view && isOpen('projection-panel')) this.view.resize();
  }

  // Toggles `panelId` to fill #main (hiding every other view-panel), or
  // brings the others back (each in whatever collapsed state it had).
  _toggleFullscreen(panelId) {
    const viewIds = ['projection-panel', 'data-table-panel', 'stats-panel'];
    const panel = document.getElementById(panelId);
    const turningOn = !panel.classList.contains('fullscreen');

    for (const id of viewIds) {
      if (id !== panelId) document.getElementById(id).classList.toggle('hidden', turningOn);
    }
    panel.classList.toggle('fullscreen', turningOn);
    if (turningOn && this._collapsed.delete(panelId)) this._saveCollapsed();
    this._applyLayout();
  }

  // The resize handle only makes sense when both panels it sits between
  // are actually on screen to trade space with each other.
  _updateResizeHandleVisibility() {
    const open = id => !this._collapsed.has(id) && !document.getElementById(id).classList.contains('hidden');
    document.getElementById('data-table-resize-handle').classList.toggle('hidden', !(open('projection-panel') && open('data-table-panel')));
  }

  // Drag-to-resize between the projection and data-table panels. Sets an
  // explicit inline height on #data-table-panel (overriding its CSS
  // default), persisted across reloads like the collapsed state.
  // Dragging the handle down shrinks the table (and lets the flex:1
  // projection panel above it grow into the freed space); dragging up
  // grows it — the opposite of the mouse's own screen-Y direction, since
  // the table sits below the handle.
  _initResizeHandle() {
    const STORAGE_KEY = 'projection-app-table-height';
    const handle = document.getElementById('data-table-resize-handle');
    const panel = document.getElementById('data-table-panel');
    const MIN_HEIGHT = 100;

    const saved = parseInt(localStorage.getItem(STORAGE_KEY), 10);
    if (Number.isFinite(saved) && saved > 0) panel.style.height = `${saved}px`;

    let startY = 0, startHeight = 0;
    const onMouseMove = e => {
      const delta = e.clientY - startY;
      panel.style.height = `${Math.max(MIN_HEIGHT, startHeight - delta)}px`;
      this.view.resize();
    };
    const onMouseUp = () => {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      handle.classList.remove('dragging');
      localStorage.setItem(STORAGE_KEY, parseInt(panel.style.height, 10));
    };
    handle.addEventListener('mousedown', e => {
      startY = e.clientY;
      startHeight = panel.getBoundingClientRect().height;
      handle.classList.add('dragging');
      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
      e.preventDefault();
    });
  }

  _toggleFormatHelp() {
    const el = document.getElementById('format-help');
    el.style.display = el.style.display === 'block' ? 'none' : 'block';
  }

  _toggleProjectionCaveats() {
    const el = document.getElementById('projection-caveats');
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
      const ext = file.name.split('.').pop().toLowerCase();
      const format = { jsonl: 'jsonl', ndjson: 'jsonl', csv: 'csv', tsv: 'csv' }[ext] || 'json';

      this._setLoading(true, 'Uploading dataset...');
      try {
        await this.backend.uploadDataset(reader.result, format);
      } catch (err) {
        this._setLoading(false);
        this._setStatus(`Invalid file: ${err.message}`, true);
        return;
      }
      this._setLoading(false);

      this._refresh();
    };
    reader.readAsText(file);
  }

  _setStatus(text, isError = false) {
    const el = document.getElementById('status');
    el.textContent = text;
    el.classList.toggle('error', isError);
  }

  // Asks the backend to embed the chosen FLORES-200 subset (or serve it from
  // its cache) and then runs the same fetch-points-and-project flow every
  // other dataset change goes through. The file input is cleared so
  // re-picking the same file still fires a change event.
  async _onLoadFlores() {
    const lang_codes = document.getElementById('flores-langs').value
      .split(/[\s,;]+/).filter(Boolean);
    const config = {
      lang_codes,
      n_sentences: parseInt(document.getElementById('flores-n').value, 10),
      model: document.getElementById('flores-model').value.trim(),
      pooling: document.getElementById('flores-pooling').value,
    };
    const button = document.getElementById('flores-load-button');
    const errorBox = document.getElementById('flores-error');
    errorBox.hidden = true;
    button.disabled = true;
    this._setLoading(true, `Embedding ${lang_codes.length} languages × ${config.n_sentences} sentences (first run can take minutes)...`);
    try {
      await this.backend.loadFlores(config);
    } catch (err) {
      this._setLoading(false);
      button.disabled = false;
      errorBox.textContent = err.message;
      errorBox.hidden = false;
      this._setStatus('FLORES-200 load failed.', true);
      return;
    }
    button.disabled = false;
    this._setLoading(false);
    document.getElementById('file-input').value = '';
    this._refresh();
  }

  async _refresh() {
    const method = document.getElementById('method-select').value;
    const metric = document.getElementById('metric-select').value;
    const hyperparams = method === 'umap' ? this._umapParams() : this._tsneParams();
    this._setStatus(`Running ${method} (${metric})...`);
    this._setLoading(true, `Running ${method} (${metric})...`);

    let meta, proj;
    try {
      if (!(await this.backend.fetchDatasetInfo()).loaded) {
        this._setLoading(false);
        this._setStatus('No dataset loaded.');
        return;
      }
      [meta, proj] = await Promise.all([
        this.backend.fetchPoints(),
        this.backend.fetchProjection(method, metric, hyperparams),
      ]);
    } catch (err) {
      this._setLoading(false);
      this._setStatus(err.message, true);
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
    this._refreshStats();
    this._setShowSelectedOnly(false); // DataTable.setPoints() below resets its own checkbox directly, bypassing the callback — this re-enables the fisheye checkbox too

    const coordsById = new Map(proj.points.map(p => [p.id, p]));
    this.points = meta.points.map(p => {
      const c = coordsById.get(p.id);
      return {
        id: p.id, lang: p.lang, text: p.text, pairId: p.pairId, family: p.family,
        hasTokens: p.hasTokens, tokenTexts: p.tokenTexts,
        // no pairId at all -> each point is its own group, so it isolates alone
        groupKey: p.pairId != null ? `pair:${p.pairId}` : `id:${p.id}`,
        x: c.x, y: c.y,
      };
    });

    document.getElementById('status').textContent =
      `${meta.points.length} points · ${meta.lang_codes.length} languages · ${meta.model} (${meta.layer} layer, ${meta.pooling}-pooled) · ${proj.method}`;

    // The backend clamps out-of-range hyperparams (e.g. n_neighbors/perplexity
    // past the dataset's point count) — reflect whatever it actually used back
    // into the inputs so they never silently disagree with the plot.
    if (proj.params) {
      if (proj.method === 'umap') {
        document.getElementById('umap-n-neighbors').value = proj.params.n_neighbors;
        document.getElementById('umap-min-dist').value = proj.params.min_dist;
      } else {
        document.getElementById('tsne-perplexity').value = proj.params.perplexity;
        document.getElementById('tsne-learning-rate').value = proj.params.learning_rate === 'auto' ? '' : proj.params.learning_rate;
      }
    }

    this.langFilter.setLangs(meta.lang_codes);
    this.table.setPoints(this.points);
    this.view.update(this.points, d => this.onCanvasClickPoint(d));
    this._renderFamilyLegend();
  }

  // Rebuilds the sidebar's family-color legend from whatever families are
  // actually present in the current dataset — empty (and hidden) whenever
  // family mode is off, since the swatches would be meaningless then.
  _renderFamilyLegend() {
    const el = document.getElementById('family-legend');
    el.innerHTML = '';
    if (!document.getElementById('family-color-toggle').checked) return;
    for (const { family, color } of this.view.getFamilyLegend()) {
      const item = document.createElement('span');
      item.className = 'family-legend-item';
      const swatch = document.createElement('span');
      swatch.className = 'family-legend-swatch';
      swatch.style.background = color;
      item.appendChild(swatch);
      item.appendChild(document.createTextNode(family));
      el.appendChild(item);
    }
  }

  // Rebuilds every consumer's view of the pinned set from scratch each
  // time, rather than patching incrementally — simplest way to guarantee
  // canvas rings/overlays and the table's rendered actions never drift
  // from _selection/_detailState.
  _syncPins() {
    const pointsById = new Map(this.points.map(p => [p.id, p]));
    const pins = this._selection.map(id => ({
      point: pointsById.get(id),
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

    // If unpinning (via canvas click, background click, or the table
    // checkbox) just emptied the selection while "show selected" was
    // isolating the canvas to it, turn that isolate back off instead of
    // leaving it isolated to an empty set — which would otherwise fade
    // every single point to invisible and look like "everything
    // disappeared" rather than "nothing is pinned anymore."
    if (this._selection.length === 0 && this.table.isShowSelectedOnly()) {
      this.table.setShowSelectedOnly(false);
      this._setShowSelectedOnly(false);
    }
  }

  // Single place that turns "show selected" on/off for the canvas side,
  // so every caller (the table's checkbox, and the auto-recovery in
  // _syncPins above) gets the same side effect: the fisheye lens is
  // mutually exclusive with isolating to the pinned set (turning show-
  // selected on forces the lens off and disables its checkbox; turning
  // it back off just re-enables the checkbox, it doesn't re-check it).
  _setShowSelectedOnly(enabled) {
    this.view.setShowPinnedOnly(enabled);
    const fisheyeToggle = document.getElementById('fisheye-toggle');
    fisheyeToggle.disabled = enabled;
    if (enabled) {
      fisheyeToggle.checked = false;
      this._setFisheye(false);
    }
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
    this._refreshStats();
  }

  // Pins every point in `points` that isn't pinned yet (the data table's
  // Select all, which passes every row matching its current filter, across
  // all pages). Same rules as _togglePin's pin branch, applied in one batch
  // so the canvas/table/stats sync runs once instead of per point.
  selectPoints(points) {
    const toAdd = points.filter(p => !this._detailState.has(p.id));
    if (toAdd.length === 0) return;
    // Translations only exist while exactly one point is pinned (see
    // _togglePin), and this always takes the selection past one.
    if (this._selection.length === 1) {
      const soleDetail = this._detailState.get(this._selection[0]);
      if (soleDetail.translations) {
        soleDetail.translations = false;
        this.view.clearHighlight();
      }
    }
    for (const point of toAdd) {
      this._pinColors.set(point.id, PIN_PALETTE[this._selection.length % PIN_PALETTE.length]);
      this._selection.push(point.id);
      this._detailState.set(point.id, { translations: false, tokens: false, tokenPoints: [] });
    }
    // First, not last: the table pages to the focused row, and a batch
    // select-all shouldn't jump you to the end of the list.
    this._focusedId = toAdd[0].id;
    this._syncPins();
    this._refreshStats();
  }

  clearSelection() {
    if (this._selection.length === 0) return;
    this._selection = [];
    this._focusedId = null;
    this._detailState.clear();
    this._pinColors.clear();
    this.view.clearHighlight();
    this._syncPins();
    this._refreshStats();
  }

  // Recomputes the statistics panel for the current selection. Debounced so
  // a burst of pin toggles costs one request, and guarded by a request
  // counter so a slow response for an older selection never overwrites a
  // newer one.
  _refreshStats() {
    clearTimeout(this._statsTimer);
    const request = ++this._statsRequest;
    const ids = [...this._selection];
    if (ids.length === 0) {
      this.stats.clear();
      return;
    }
    this.stats.setLoading();
    this._statsTimer = setTimeout(async () => {
      try {
        const result = await this.backend.fetchStats(ids);
        if (request === this._statsRequest) this.stats.setStats(result);
      } catch (err) {
        if (request === this._statsRequest) this.stats.setError(err.message);
      }
    }, 150);
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
      return;
    }

    const methodSelect = document.getElementById('method-select');
    if (methodSelect.value !== 'umap') {
      methodSelect.value = 'umap';
      this._syncParamVisibility();
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
      proj = await this.backend.fetchTokenProjection(point.id, metric, this._umapParams());
    } catch (err) {
      this._setLoading(false);
      this._setStatus(err.message, true);
      return;
    }
    this._setLoading(false);

    const liveDetail = this._detailState.get(point.id);
    if (!liveDetail) return; // unpinned while the request was in flight
    liveDetail.tokens = true;
    liveDetail.tokenPoints = proj.tokens.map((t, i) => ({
      id: `${point.id}::${i}`, lang: point.lang, family: point.family, text: t.token, x: t.x, y: t.y,
    }));
    this._syncPins();
  }

}


// ── Entry point ──────────────────────────────────────────────────────────────
// type="module" scripts are deferred automatically, so the DOM is already
// parsed by the time this runs — no onload handler needed.

new App().init();
