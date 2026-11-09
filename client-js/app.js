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
import { PanelLayout } from './panel-layout.js';

// Same origin by default (server.py serves this page); ?backend=http://host:port overrides it.
const BACKEND_URL = new URLSearchParams(location.search).get('backend') ?? '';
const PIN_PALETTE = d3.schemeTableau10;

const byId = id => document.getElementById(id);
const on = (id, event, handler) => byId(id).addEventListener(event, handler);
const VIEW3D_KEY = 'projection-app-3d';
const COLOR_MODE_KEY = 'projection-app-color-mode';
const COLOR_LANGS_KEY = 'projection-app-color-langs';
const FISHEYE_KEY = 'projection-app-fisheye';
const SELECTION_KEY = 'projection-app-selection';
const TOKENS_NEED_UMAP = 'Tokens are placed into a UMAP layout; t-SNE has no way to add points to one.';
const EDIT_NEEDS_UMAP = 't-SNE can’t add points to an existing layout, so every point would move. Switch to UMAP to edit, or load a new selection.';

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
        tokensUnavailable: () => (this._method() === 'umap' ? null : TOKENS_NEED_UMAP),
        onToggleShowSelectedOnly: enabled => this._setShowSelectedOnly(enabled),
        onClearSelection: () => this.clearSelection(),
        onSelectAll: points => this.selectPoints(points),
      }
    );
    this.stats = new StatsView({ containerId: 'stats' });
    this.langFilter = new LangFilter(
      { inputId: 'lang-search', suggestionsId: 'lang-suggestions', chipsId: 'lang-chips' },
      selected => {
        localStorage.setItem(COLOR_LANGS_KEY, JSON.stringify(selected));
        this.view.setLangFilter(selected);
      }
    );
    this.floresPanel = new FloresPanel(request => this._onFloresRequest(request));
    this.points = [];
    this._colorLangsToRestore = this._storedColorLangs(); // applied once, to the first dataset after a page load
    this._selectionToRestore = this._storedSelection();   // likewise
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
    this.layout = new PanelLayout({ onProjectionResized: () => this.view?.resize() });
    this._restoreView3D();
    this._restoreColorMode();
    this._restoreFisheye();
    this._blockTrackpadPageZoom();

    this._bindDatasetControls();
    this._bindViewControls();
    this._bindLoadingOverlay();
    installTooltips();
    this.layout.init();
    installActionsMenuOutsideClickHandler();

    await this._refresh();
  }

  _bindDatasetControls() {
    on('file-input', 'change', e => this._onFileSelected(e.target.files[0]));
    on('method-select', 'change', () => { this._syncParamVisibility(); this._refresh({ keepState: true }); });
    this._syncParamVisibility();
    on('metric-select', 'change', () => { this._syncMetricDescription(); this._refresh({ keepState: true }); });
    this._syncMetricDescription();
    // n_neighbors/min_dist/perplexity/learning_rate deliberately do NOT
    // auto-refresh on change — editing more than one before blurring away
    // would fire a _refresh() per field, and an in-flight response's
    // "reflect the effective params back" step (see _reflectParams) could
    // then clobber a value the user typed into another field afterward.
    // Refit is the one explicit trigger for all of them.
    on('refit-button', 'click', () => this._refit());
    on('overlay-refit', 'click', () => this._refit());
    on('overlay-fit-view', 'click', () => this.view.fitView());
    on('view3d-toggle', 'change', e => this._setView3D(e.target.checked));
    on('format-help-toggle', 'click', () => this._toggleDisplay('format-help'));
    on('projection-caveats-toggle', 'click', () => this._toggleDisplay('projection-caveats'));
  }

  _bindViewControls() {
    on('label-select', 'change', e => this.view.setLabelBy(e.target.value));
    on('fisheye-toggle', 'change', e => {
      this._setFisheye(e.target.checked);
      this._saveFisheye();
    });
    on('fisheye-size', 'input', e => {
      byId('fisheye-size-value').textContent = e.target.value;
      this.view.setFisheyeRadius(Number(e.target.value));
      this._saveFisheye();
    });
    on('fisheye-style-select', 'change', e => {
      this.view.setFisheyeStyle(e.target.value);
      this._saveFisheye();
    });
    on('color-mode-select', 'change', e => this._setColorMode(e.target.value));
    this._setColorMode(byId('color-mode-select').value);
    on('view3d-reset', 'click', () => this.view.resetView());
    this.view.setAreaSelectHandler(points => this.selectPoints(points));
  }

  _bindLoadingOverlay() {
    on('loading-cancel', 'click', () => this._cancelHandler?.());
    window.addEventListener('keydown', e => {
      if (e.key === 'Escape') this._cancelHandler?.();
    });
  }

  _method() {
    return byId('method-select').value;
  }

  // UMAP and t-SNE take disjoint hyperparameters, so only the selected
  // method's fields are shown. Editing the loaded dataset needs UMAP's
  // out-of-sample placement; t-SNE has none, so it would refit and move everything.
  _syncParamVisibility() {
    const umap = this._method() === 'umap';
    byId('umap-params').hidden = !umap;
    byId('tsne-params').hidden = umap;
    this.floresPanel.setEditBlocked(umap ? null : EDIT_NEEDS_UMAP);
  }

  // The option's own title doubles as an always-visible explanation, since
  // hover text on <option> isn't shown by every browser.
  _syncMetricDescription() {
    const select = byId('metric-select');
    byId('metric-description').textContent = select.selectedOptions[0].title;
  }

  _umapParams() {
    return {
      n_neighbors: byId('umap-n-neighbors').value,
      min_dist: byId('umap-min-dist').value,
    };
  }

  _tsneParams() {
    return {
      perplexity: byId('tsne-perplexity').value,
      learning_rate: byId('tsne-learning-rate').value,
    };
  }

  // Language/family color every point; "custom" only colors the languages
  // picked in the language filter, so the picker is only shown for it.
  _setColorMode(mode) {
    localStorage.setItem(COLOR_MODE_KEY, mode);
    this.view.setColorMode(mode);
    byId('lang-picker').hidden = mode !== 'custom';
    this._renderColorLegend();
  }

  // The first projection after a page load must already use the dimension the
  // user left on, so it is read from storage rather than from the checkbox,
  // which the browser may restore late or not at all.
  _restoreView3D() {
    const enabled = localStorage.getItem(VIEW3D_KEY) === '1';
    byId('view3d-toggle').checked = enabled;
    byId('view3d-options').hidden = !enabled;
  }

  _restoreColorMode() {
    const select = byId('color-mode-select');
    const stored = localStorage.getItem(COLOR_MODE_KEY);
    if ([...select.options].some(o => o.value === stored)) select.value = stored;
  }

  // The lens settings are applied to the view explicitly: the browser may
  // restore the checkbox on its own, which would leave it checked with no lens.
  _restoreFisheye() {
    let saved = {};
    try {
      saved = JSON.parse(localStorage.getItem(FISHEYE_KEY) || '{}') || {};
    } catch { /* an unreadable value means defaults */ }
    const size = byId('fisheye-size');
    const style = byId('fisheye-style-select');
    if (Number.isFinite(saved.radius)) size.value = saved.radius;
    if ([...style.options].some(o => o.value === saved.style)) style.value = saved.style;
    byId('fisheye-size-value').textContent = size.value;
    this.view.setFisheyeRadius(Number(size.value));
    this.view.setFisheyeStyle(style.value);
    byId('fisheye-toggle').checked = saved.enabled === true;
    this._setFisheye(saved.enabled === true);
  }

  // Saved from the user's own changes only: showing the selection turns the
  // lens off for a while, which shouldn't forget that it was wanted.
  _saveFisheye() {
    localStorage.setItem(FISHEYE_KEY, JSON.stringify({
      enabled: byId('fisheye-toggle').checked,
      radius: Number(byId('fisheye-size').value),
      style: byId('fisheye-style-select').value,
    }));
  }

  // A pinch on a touchpad reaches the page as a Ctrl+wheel with a fractional
  // delta; left alone it magnifies the whole UI instead of the browser's zoom.
  // Mouse Ctrl+wheel (whole-number deltas) still changes the browser zoom, and
  // the canvas does its own zoom for both.
  _blockTrackpadPageZoom() {
    window.addEventListener('wheel', e => {
      if (e.ctrlKey && !Number.isInteger(e.deltaY)) e.preventDefault();
    }, { passive: false });
  }

  _storedSelection() {
    try {
      const ids = JSON.parse(localStorage.getItem(SELECTION_KEY) || '[]');
      return Array.isArray(ids) ? ids.filter(id => typeof id === 'string') : [];
    } catch {
      return [];
    }
  }

  _storedColorLangs() {
    try {
      const langs = JSON.parse(localStorage.getItem(COLOR_LANGS_KEY) || '[]');
      return Array.isArray(langs) ? langs.filter(l => typeof l === 'string') : [];
    } catch {
      return [];
    }
  }

  _is3D() {
    return byId('view3d-toggle').checked;
  }

  // Switching dimensionality re-projects the loaded dataset (keeping the
  // selection and colours), since 2D and 3D are separate fits.
  _setView3D(enabled) {
    byId('view3d-options').hidden = !enabled;
    localStorage.setItem(VIEW3D_KEY, enabled ? '1' : '0');
    this._refresh({ keepState: true });
  }

  // The lens size/style controls only matter while the lens is on.
  _setFisheye(enabled) {
    this.view.setFisheyeEnabled(enabled);
    byId('fisheye-options').hidden = !enabled;
  }

  _toggleDisplay(id) {
    const el = byId(id);
    el.style.display = el.style.display === 'block' ? 'none' : 'block';
  }

  // Shows/hides the full-screen loading overlay used around every backend
  // call — a visible spinner instead of just the (still-updated) status
  // text, since a UMAP/t-SNE refit or a token projection can take a while.
  // onCancel, when given, shows a Cancel button (and arms Escape) that calls it.
  _setLoading(isLoading, message, onCancel = null) {
    byId('loading-overlay').classList.toggle('visible', isLoading);
    if (isLoading) byId('loading-text').textContent = message;
    this._cancelHandler = isLoading ? onCancel : null;
    byId('loading-cancel').hidden = !this._cancelHandler;
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
    const el = byId('status');
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
    byId('file-input').value = '';
    this._refresh({ keepState: editing });
  }

  // keepState is for the dataset that's already loaded, re-projected (method,
  // metric, params) or edited (languages, sentences): selection, pins and the
  // language colouring survive for the points still there. Loading a
  // different dataset always starts clean. refit fits UMAP on every point
  // again instead of placing the points added since the last fit into its layout.
  async _refresh({ keepState = false, refit = false } = {}) {
    const method = byId('method-select').value;
    const metric = byId('metric-select').value;
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
    byId('projection-hint')?.remove();

    // keepState only holds for the same dataset (re-projected or edited), which
    // still has some of the points.
    const previousIds = new Set(this.points.map(p => p.id));
    const keep = keepState && meta.points.some(p => previousIds.has(p.id));
    const sameLayout = keep && proj.fit_id != null && proj.fit_id === this._fitId;
    let droppedPins = false;

    if (keep) {
      droppedPins = this._dropVanishedPins(new Set(meta.points.map(p => p.id)));
      if (!sameLayout) this._forgetTokenOverlays(); // token points were projected into the previous fit's space
    } else {
      this._resetSelection();
    }

    this.points = this._buildPoints(meta, proj);
    this._fitId = proj.fit_id;
    this._overlaidCount = proj.n_overlaid || 0;

    byId('status').textContent =
      `${meta.points.length} points · ${meta.lang_codes.length} languages · ${meta.model} (${meta.layer} layer, ${meta.pooling}-pooled) · ${proj.method}${proj.n_components === 3 ? ' (3D)' : ''}`;
    this._renderOverlayNote(meta.points.length);
    this.floresPanel.setDataset(meta);
    this._reflectParams(proj);

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
      if (this._colorLangsToRestore.length) {
        this.langFilter.setSelected(this._colorLangsToRestore);
        this.view.setLangFilter(this.langFilter.getSelected());
      }
      this._restoreSelection(meta);
    }
    this._colorLangsToRestore = [];
    this._selectionToRestore = [];
    this._renderColorLegend();
  }

  _buildPoints(meta, proj) {
    const coordsById = new Map(proj.points.map(p => [p.id, p]));
    return meta.points.map(p => {
      const c = coordsById.get(p.id);
      return {
        id: p.id, lang: p.lang, text: p.text, pairId: p.pairId, family: p.family,
        hasTokens: p.hasTokens, tokenTexts: p.tokenTexts,
        // no pairId at all -> each point is its own group, so it isolates alone
        groupKey: p.pairId != null ? `pair:${p.pairId}` : `id:${p.id}`,
        x: c.x, y: c.y, z: c.z, overlaid: Boolean(c.overlaid),
      };
    });
  }

  _forgetTokenOverlays() {
    for (const detail of this._detailState.values()) {
      detail.tokens = false;
      detail.tokenPoints = [];
    }
  }

  // Pins the points selected before the page was reloaded that this dataset still has.
  _restoreSelection(meta) {
    const present = new Set(meta.points.map(p => p.id));
    const pointsById = new Map(this.points.map(p => [p.id, p]));
    const ids = this._selectionToRestore.filter(id => present.has(id));
    if (ids.length === 0) return;
    for (const id of ids) this._addPin(pointsById.get(id));
    this._focusedId = this._lastPinnedId();
    this._syncPins();
    this._refreshStats();
  }

  // A different dataset starts with nothing selected.
  _resetSelection() {
    this._selection = [];
    this._translationGroups.clear();
    this._focusedId = null;
    this._detailState.clear();
    this._pinColors.clear();
    localStorage.setItem(SELECTION_KEY, '[]');
    this.view.setPinnedPoints([]);
    this.view.setTokenOverlays(new Map());
    this._refreshStats();
    this._setShowSelectedOnly(false); // DataTable.setPoints() resets its own checkbox directly, bypassing the callback — this re-enables the fisheye checkbox too
  }

  // The backend clamps out-of-range hyperparams (e.g. n_neighbors/perplexity
  // past the dataset's point count) — reflect whatever it actually used back
  // into the inputs so they never silently disagree with the plot.
  _reflectParams(proj) {
    if (!proj.params) return;
    if (proj.method === 'umap') {
      byId('umap-n-neighbors').value = proj.params.n_neighbors;
      byId('umap-min-dist').value = proj.params.min_dist;
    } else {
      byId('tsne-perplexity').value = proj.params.perplexity;
      byId('tsne-learning-rate').value = proj.params.learning_rate === 'auto' ? '' : proj.params.learning_rate;
    }
  }

  // Rebuilds the sidebar's color legend from whatever languages/families are
  // actually present in the current dataset — empty in "custom" mode, where
  // the language chips are the key.
  _renderColorLegend() {
    const el = byId('color-legend');
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
    byId('overlay-note').hidden = overlaid === 0;
    if (overlaid === 0) return;
    let text = `${overlaid} of ${total} points were placed into the existing layout instead of refitting it; nothing already plotted moved.`;
    if (overlaid > total - overlaid) text += ' Most points are overlaid; a refit gives a more faithful layout.';
    byId('overlay-note-text').textContent = text;
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
      this._focusedId = this._lastPinnedId();
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
    localStorage.setItem(SELECTION_KEY, JSON.stringify(this._selection));

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
    const fisheyeToggle = byId('fisheye-toggle');
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
      this._addPin(point);
      this._focusedId = point.id;
    } else {
      this._selection.splice(idx, 1);
      this._detailState.delete(point.id);
      this._pinColors.delete(point.id);
      if (this._focusedId === point.id) this._focusedId = this._lastPinnedId();
    }
    this._syncPins();
    this._refreshStats();
  }

  _addPin(point) {
    this._pinColors.set(point.id, PIN_PALETTE[this._selection.length % PIN_PALETTE.length]);
    this._selection.push(point.id);
    this._detailState.set(point.id, { tokens: false, tokenPoints: [] });
  }

  _lastPinnedId() {
    return this._selection.length ? this._selection[this._selection.length - 1] : null;
  }

  // Pins every point in `points` that isn't pinned yet (the data table's
  // Select all, which passes every row matching its current filter, across
  // all pages). Same rules as _togglePin's pin branch, applied in one batch
  // so the canvas/table/stats sync runs once instead of per point.
  selectPoints(points) {
    const toAdd = points.filter(p => !this._detailState.has(p.id));
    if (toAdd.length === 0) return;
    for (const point of toAdd) this._addPin(point);
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

    const metric = byId('metric-select').value;
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
