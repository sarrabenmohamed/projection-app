// Embedding Projection — app orchestrator. Owns selection state and wires
// together the projection view, backend client, data table, and language
// filter; each of those owns its own domain and App never reaches past
// their interfaces.

import { ProjectionView } from './projection-view.js';
import { BackendClient } from './backend-client.js';
import { DataTable } from './data-table.js';
import { FloresPanel } from './flores-panel.js';
import { LangFilter } from './lang-filter.js';
import { installTooltips } from './help-tips.js';
import { StatsView } from './stats-view.js';
import { installActionsMenuOutsideClickHandler } from './actions-menu.js';
import { makeResizable } from './resizer.js';

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
        onSelectTranslations: point => this.onSelectTranslations(point),
        onToggleTokens: point => this.onToggleTokens(point),
        translationsOn: point => this._translationGroups.has(point.groupKey),
        tokensUnavailable: () => (this._method() === 'umap' ? null : 'Tokens are placed into a UMAP layout; t-SNE has no way to add points to one.'),
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
    this.floresPanel = new FloresPanel(request => this._onFloresRequest(request));
    this.points = [];
    this._fitId = null;             // which UMAP fit the plotted layout came from (token overlays are only valid within one)
    this._overlaidCount = 0;        // points placed into the layout rather than fitted
    this._selection = [];           // ordered array of pinned ids, insertion order
    this._translationGroups = new Set();  // groupKeys whose translations are selected; points added to them later join the selection
    this._focusedId = null;
    this._detailState = new Map();  // id -> {tokens, tokenPoints}
    this._pinColors = new Map();    // id -> color string
    this._statsRequest = 0;         // bumps on every selection change so a stale /stats response is dropped
    this._statsTimer = null;
  }

  async init() {
    this.view = new ProjectionView('projection');

    document.getElementById('file-input')
      .addEventListener('change', e => this._onFileSelected(e.target.files[0]));
    document.getElementById('method-select')
      .addEventListener('change', () => { this._syncParamVisibility(); this._refresh({ keepState: true }); });
    this._syncParamVisibility();
    document.getElementById('metric-select')
      .addEventListener('change', () => { this._syncMetricDescription(); this._refresh({ keepState: true }); });
    this._syncMetricDescription();
    // n_neighbors/min_dist/perplexity/learning_rate deliberately do NOT
    // auto-refresh on change — editing more than one before blurring away
    // would fire a _refresh() per field, and an in-flight response's
    // "reflect the effective params back" step (see _loadDataset) could
    // then clobber a value the user typed into another field afterward.
    // Refit is the one explicit trigger for all of them.
    document.getElementById('refit-button')
      .addEventListener('click', () => this._refit());
    document.getElementById('overlay-refit')
      .addEventListener('click', () => this._refit());
    document.getElementById('overlay-fit-view')
      .addEventListener('click', () => this.view.fitView());
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
    document.getElementById('color-mode-select')
      .addEventListener('change', e => this._setColorMode(e.target.value));
    this._setColorMode(document.getElementById('color-mode-select').value);
    document.getElementById('view3d-toggle')
      .addEventListener('change', e => this._setView3D(e.target.checked));
    document.getElementById('view3d-reset')
      .addEventListener('click', () => this.view.resetView());
    this.view.setAreaSelectHandler(points => this.selectPoints(points));
    document.getElementById('loading-cancel')
      .addEventListener('click', () => this._cancelHandler?.());
    window.addEventListener('keydown', e => {
      if (e.key === 'Escape') this._cancelHandler?.();
    });
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
    const resizeView = () => this.view.resize();
    const resizable = (handleId, panelId, storageKey, options) => makeResizable({
      handle: document.getElementById(handleId), panel: document.getElementById(panelId),
      storageKey, onResize: resizeView, ...options,
    });
    resizable('data-table-resize-handle', 'data-table-panel', 'projection-app-table-height', { axis: 'y', sign: -1, min: 100 });
    resizable('stats-resize-handle', 'stats-panel', 'projection-app-stats-height', { axis: 'y', sign: -1, min: 100 });
    resizable('sidebar-resize-handle', 'sidebar', 'projection-app-sidebar-width', { axis: 'x', sign: 1, min: 200, max: 600 });
    installActionsMenuOutsideClickHandler();

    await this._refresh();
  }

  _method() {
    return document.getElementById('method-select').value;
  }

  // UMAP and t-SNE take disjoint hyperparameters, so only the selected
  // method's fields are shown. Editing the loaded dataset needs UMAP's
  // out-of-sample placement; t-SNE has none, so it would refit and move everything.
  _syncParamVisibility() {
    const umap = this._method() === 'umap';
    document.getElementById('umap-params').hidden = !umap;
    document.getElementById('tsne-params').hidden = umap;
    this.floresPanel.setEditBlocked(umap ? null : 't-SNE can’t add points to an existing layout, so every point would move. Switch to UMAP to edit, or load a new selection.');
  }

  // The option's own title doubles as an always-visible explanation, since
  // hover text on <option> isn't shown by every browser.
  _syncMetricDescription() {
    const select = document.getElementById('metric-select');
    document.getElementById('metric-description').textContent = select.selectedOptions[0].title;
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

  // Language/family color every point; "custom" only colors the languages
  // picked in the language filter, so the picker is only shown for it.
  _setColorMode(mode) {
    this.view.setColorMode(mode);
    document.getElementById('lang-picker').hidden = mode !== 'custom';
    this._renderColorLegend();
  }

  _is3D() {
    return document.getElementById('view3d-toggle').checked;
  }

  // Switching dimensionality re-projects the loaded dataset (keeping the
  // selection and colours), since 2D and 3D are separate fits.
  _setView3D(enabled) {
    document.getElementById('view3d-options').hidden = !enabled;
    this._refresh({ keepState: true });
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
    const fixedHeight = id => open(id) && !document.getElementById(id).classList.contains('grow');
    document.getElementById('data-table-resize-handle').classList.toggle('hidden', !(open('projection-panel') && fixedHeight('data-table-panel')));
    document.getElementById('sidebar-resize-handle').classList.toggle('hidden', this._collapsed.has('sidebar'));
    document.getElementById('stats-resize-handle').classList.toggle('hidden', !((open('projection-panel') || open('data-table-panel')) && fixedHeight('stats-panel')));
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
  // onCancel, when given, shows a Cancel button (and arms Escape) that calls it.
  _setLoading(isLoading, message, onCancel = null) {
    document.getElementById('loading-overlay').classList.toggle('visible', isLoading);
    if (isLoading) document.getElementById('loading-text').textContent = message;
    this._cancelHandler = isLoading ? onCancel : null;
    document.getElementById('loading-cancel').hidden = !this._cancelHandler;
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

  // Asks the backend to embed the requested FLORES-200 subset (or serve it
  // from its cache), then runs the same fetch-points-and-project flow every
  // other dataset change goes through. "edit" changes the loaded dataset and
  // keeps the plot's state (its new points are placed into the layout);
  // "load" replaces it. The file input is cleared so re-picking the same file
  // still fires a change event.
  async _onFloresRequest({ mode, config }) {
    const editing = mode === 'edit';
    const verb = editing ? 'update' : 'load';
    this.floresPanel.clearError();
    this.floresPanel.setBusy(true);
    // Aborting the request frees the UI at once; the backend call stops the
    // embedding job itself at its next sentence.
    const controller = new AbortController();
    const cancel = () => {
      controller.abort();
      this.backend.cancelFlores().catch(() => {});
    };
    this._setLoading(true, editing ? 'Updating dataset...' : 'Embedding...', cancel);
    try {
      await (editing ? this.backend.updateFlores(config, controller.signal) : this.backend.loadFlores(config, controller.signal));
    } catch (err) {
      this._setLoading(false);
      this.floresPanel.setBusy(false);
      if (err.name === 'AbortError') {
        this._setStatus(`FLORES-200 ${verb} cancelled.`);
        return;
      }
      this.floresPanel.showError(err.message);
      this._setStatus(`FLORES-200 ${verb} failed.`, true);
      return;
    }
    this.floresPanel.setBusy(false);
    this._setLoading(false);
    document.getElementById('file-input').value = '';
    this._refresh({ keepState: editing });
  }

  // keepState is for the dataset that's already loaded, re-projected (method,
  // metric, params) or edited (languages, sentences): selection, pins and the
  // language colouring survive for the points still there. Loading a
  // different dataset always starts clean. refit fits UMAP on every point
  // again instead of placing the points added since the last fit into its layout.
  async _refresh({ keepState = false, refit = false } = {}) {
    const method = document.getElementById('method-select').value;
    const metric = document.getElementById('metric-select').value;
    const hyperparams = { ...(method === 'umap' ? this._umapParams() : this._tsneParams()), n_components: this._is3D() ? 3 : 2 };
    if (refit) hyperparams.refit = true;
    this._setStatus(`Running ${method}...`);
    this._setLoading(true, `Running ${method}...`);

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
    this._loadDataset(meta, proj, keepState);
  }

  _loadDataset(meta, proj, keepState = false) {
    const hint = document.getElementById('projection-hint');
    if (hint) hint.remove();

    // keepState only holds for the same dataset (re-projected or edited), which
    // still has some of the points.
    const previousIds = new Set(this.points.map(p => p.id));
    const keep = keepState && meta.points.some(p => previousIds.has(p.id));
    const sameLayout = keep && proj.fit_id != null && proj.fit_id === this._fitId;
    let droppedPins = false;

    if (keep) {
      droppedPins = this._dropVanishedPins(new Set(meta.points.map(p => p.id)));
      if (proj.fit_id == null || proj.fit_id !== this._fitId) {
        // Token points were projected into the previous fit's space.
        for (const detail of this._detailState.values()) {
          detail.tokens = false;
          detail.tokenPoints = [];
        }
      }
    } else {
      this._selection = [];
      this._translationGroups.clear();
      this._focusedId = null;
      this._detailState.clear();
      this._pinColors.clear();
      this.view.setPinnedPoints([]);
      this.view.setTokenOverlays(new Map());
      this._refreshStats();
      this._setShowSelectedOnly(false); // DataTable.setPoints() below resets its own checkbox directly, bypassing the callback — this re-enables the fisheye checkbox too
    }

    const coordsById = new Map(proj.points.map(p => [p.id, p]));
    this.points = meta.points.map(p => {
      const c = coordsById.get(p.id);
      return {
        id: p.id, lang: p.lang, text: p.text, pairId: p.pairId, family: p.family,
        hasTokens: p.hasTokens, tokenTexts: p.tokenTexts,
        // no pairId at all -> each point is its own group, so it isolates alone
        groupKey: p.pairId != null ? `pair:${p.pairId}` : `id:${p.id}`,
        x: c.x, y: c.y, z: c.z, overlaid: Boolean(c.overlaid),
      };
    });
    this._fitId = proj.fit_id;
    this._overlaidCount = proj.n_overlaid || 0;

    document.getElementById('status').textContent =
      `${meta.points.length} points · ${meta.lang_codes.length} languages · ${meta.model} (${meta.layer} layer, ${meta.pooling}-pooled) · ${proj.method}${proj.n_components === 3 ? ' (3D)' : ''}`;
    this._renderOverlayNote(meta.points.length);
    this.floresPanel.setDataset(meta);

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

    if (keep) {
      this.langFilter.setLangs(meta.lang_codes, { keepSelection: true });
      this.table.updatePoints(this.points);
      this.view.update(this.points, d => this.onCanvasClickPoint(d), { keepColors: true, keepView: sameLayout });
      // PointLayer.setPoints() resets its own per-dataset state; re-apply it.
      this.view.setLangFilter(this.langFilter.getSelected());
      this.view.setShowPinnedOnly(this.table.isShowSelectedOnly());
      this._syncPins();
      if (droppedPins) this._refreshStats();
      this._followTranslations();
    } else {
      this.langFilter.setLangs(meta.lang_codes);
      this.table.setPoints(this.points);
      this.view.update(this.points, d => this.onCanvasClickPoint(d));
    }
    this._renderColorLegend();
  }

  // Rebuilds the sidebar's color legend from whatever languages/families are
  // actually present in the current dataset — empty in "custom" mode, where
  // the language chips are the key.
  _renderColorLegend() {
    const el = document.getElementById('color-legend');
    el.innerHTML = '';
    for (const { label, color } of this.view.getColorLegend()) {
      const item = document.createElement('span');
      item.className = 'legend-item';
      const swatch = document.createElement('span');
      swatch.className = 'legend-swatch';
      swatch.style.background = color;
      item.appendChild(swatch);
      item.appendChild(document.createTextNode(label));
      el.appendChild(item);
    }
  }

  // Tells the user that some points were placed into the existing layout
  // rather than fitted with the rest, and offers the refit that includes them.
  _renderOverlayNote(total) {
    const overlaid = this._overlaidCount;
    document.getElementById('overlay-note').hidden = overlaid === 0;
    if (overlaid === 0) return;
    let text = `${overlaid} of ${total} points were placed into the existing layout instead of refitting it; nothing already plotted moved.`;
    if (overlaid > total - overlaid) text += ' Most points are overlaid; a refit gives a more faithful layout.';
    document.getElementById('overlay-note-text').textContent = text;
  }

  // Fits the projection on every point again, with the current settings.
  // Only forced when points were overlaid; otherwise the backend refits
  // exactly when a setting changed.
  _refit() {
    return this._refresh({ keepState: true, refit: this._overlaidCount > 0 });
  }

  // After an edit: forgets the pins (and their tokens, colours, focus) of
  // points that are no longer in the dataset. True if there were any.
  _dropVanishedPins(presentIds) {
    const before = this._selection.length;
    this._selection = this._selection.filter(id => presentIds.has(id));
    for (const id of [...this._detailState.keys()]) {
      if (presentIds.has(id)) continue;
      this._detailState.delete(id);
      this._pinColors.delete(id);
    }
    if (!presentIds.has(this._focusedId)) {
      this._focusedId = this._selection.length ? this._selection[this._selection.length - 1] : null;
    }
    return this._selection.length !== before;
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

  // Toggles `point`'s membership in the selection set.
  _togglePin(point) {
    const idx = this._selection.indexOf(point.id);
    if (idx === -1) {
      this._pinColors.set(point.id, PIN_PALETTE[this._selection.length % PIN_PALETTE.length]);
      this._selection.push(point.id);
      this._detailState.set(point.id, { tokens: false, tokenPoints: [] });
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
    for (const point of toAdd) {
      this._pinColors.set(point.id, PIN_PALETTE[this._selection.length % PIN_PALETTE.length]);
      this._selection.push(point.id);
      this._detailState.set(point.id, { tokens: false, tokenPoints: [] });
    }
    // First, not last: the table pages to the focused row, and a batch
    // select-all shouldn't jump you to the end of the list.
    this._focusedId = toAdd[0].id;
    this._syncPins();
    this._refreshStats();
  }

  clearSelection() {
    this._translationGroups.clear();
    if (this._selection.length === 0) return;
    this._selection = [];
    this._focusedId = null;
    this._detailState.clear();
    this._pinColors.clear();
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

  // Toggles the selection of every translation of `point` (all points
  // sharing its pairId). Turning it on also isolates the plot to the
  // selection; the group stays followed, so a language added later joins it.
  onSelectTranslations(point) {
    const key = point.groupKey;
    const group = this.points.filter(p => p.groupKey === key);
    if (this._translationGroups.delete(key)) {
      for (const p of group) if (this._detailState.has(p.id)) this._togglePin(p);
      return;
    }
    this._translationGroups.add(key);
    this.selectPoints(group);
    this.table.setShowSelectedOnly(true);
    this._setShowSelectedOnly(true);
  }

  // After an edit: the points added to a followed group join the selection;
  // a group that no longer has translations is forgotten.
  _followTranslations() {
    for (const key of [...this._translationGroups]) {
      const group = this.points.filter(p => p.groupKey === key);
      if (group.length < 2) this._translationGroups.delete(key);
      else this.selectPoints(group);
    }
  }

  // Tokens are independent per pinned point. Token projection only exists
  // in the shared UMAP space (see backend/projection.py), so the action is
  // unavailable while the main view is on t-SNE.
  async onToggleTokens(point) {
    const detail = this._detailState.get(point.id);
    if (!detail) return;
    if (detail.tokens) {
      detail.tokens = false;
      detail.tokenPoints = [];
      this._syncPins();
      return;
    }

    const metric = document.getElementById('metric-select').value;
    this._setLoading(true, 'Computing token projection...');
    let proj;
    try {
      proj = await this.backend.fetchTokenProjection(point.id, metric, { ...this._umapParams(), n_components: this._is3D() ? 3 : 2 });
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
      id: `${point.id}::${i}`, lang: point.lang, family: point.family, text: t.token, x: t.x, y: t.y, z: t.z,
    }));
    this._syncPins();
  }

}


// ── Entry point ──────────────────────────────────────────────────────────────
// type="module" scripts are deferred automatically, so the DOM is already
// parsed by the time this runs — no onload handler needed.

new App().init();
