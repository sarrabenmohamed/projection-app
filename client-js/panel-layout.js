// Collapsing, full-screening and resizing of the four panels (sidebar,
// projection, data table, statistics). Owns the layout state only; it never
// touches app data.

import { makeResizable } from './resizer.js';

const PANEL_IDS = ['sidebar', 'projection-panel', 'data-table-panel', 'stats-panel'];
const VIEW_IDS = ['projection-panel', 'data-table-panel', 'stats-panel'];
const COLLAPSED_KEY = 'projection-app-collapsed';

const byId = id => document.getElementById(id);

export class PanelLayout {
  // onProjectionResized fires whenever the projection's size may have changed.
  constructor({ onProjectionResized }) {
    this._onProjectionResized = onProjectionResized;
    this._collapsed = this._loadCollapsed();
  }

  // Every view can be collapsed to just its header bar; the choice persists
  // in localStorage. Collapsing never removes the header, so there is always
  // a way back.
  init() {
    for (const btn of document.querySelectorAll('[data-collapse]')) {
      btn.addEventListener('click', () => {
        const id = btn.dataset.collapse;
        if (this._collapsed.has(id)) this._collapsed.delete(id); else this._collapsed.add(id);
        this._saveCollapsed();
        this._apply();
      });
    }
    byId('projection-fullscreen-btn').addEventListener('click', () => this._toggleFullscreen('projection-panel'));
    byId('data-table-fullscreen-btn').addEventListener('click', () => this._toggleFullscreen('data-table-panel'));
    byId('stats-fullscreen-btn').addEventListener('click', () => this._toggleFullscreen('stats-panel'));

    const resizable = (handleId, panelId, storageKey, options) => makeResizable({
      handle: byId(handleId), panel: byId(panelId),
      storageKey, onResize: this._onProjectionResized, ...options,
    });
    resizable('data-table-resize-handle', 'data-table-panel', 'projection-app-table-height', { axis: 'y', sign: -1, min: 100 });
    resizable('stats-resize-handle', 'stats-panel', 'projection-app-stats-height', { axis: 'y', sign: -1, min: 100 });
    resizable('sidebar-resize-handle', 'sidebar', 'projection-app-sidebar-width', { axis: 'x', sign: 1, min: 200, max: 600 });

    this._apply();
  }

  _loadCollapsed() {
    let saved = [];
    try { saved = JSON.parse(localStorage.getItem(COLLAPSED_KEY) || '[]'); } catch { saved = []; }
    return new Set(Array.isArray(saved) ? saved.filter(id => PANEL_IDS.includes(id)) : []);
  }

  _saveCollapsed() {
    localStorage.setItem(COLLAPSED_KEY, JSON.stringify([...this._collapsed]));
  }

  _isOpen(id) {
    return !this._collapsed.has(id) && !byId(id).classList.contains('hidden');
  }

  // Single place that turns collapsed/hidden state into classes: the
  // projection fills spare space when expanded, otherwise the topmost
  // expanded view does, so a collapsed panel never leaves a blank gap.
  _apply() {
    for (const id of PANEL_IDS) {
      const collapsed = this._collapsed.has(id);
      byId(id).classList.toggle('collapsed', collapsed);
      const btn = document.querySelector(`[data-collapse="${id}"]`);
      btn.textContent = collapsed ? '▸' : (id === 'sidebar' ? '◂' : '▾');
      btn.title = collapsed ? 'Expand' : 'Collapse';
      btn.setAttribute('aria-expanded', String(!collapsed));
    }
    const growId = VIEW_IDS.find(id => this._isOpen(id));
    for (const id of VIEW_IDS) {
      byId(id).classList.toggle('grow', id === growId && id !== 'projection-panel');
    }
    this._updateResizeHandles();
    if (this._isOpen('projection-panel')) this._onProjectionResized();
  }

  // Toggles `panelId` to fill #main (hiding every other view-panel), or
  // brings the others back (each in whatever collapsed state it had).
  _toggleFullscreen(panelId) {
    const panel = byId(panelId);
    const turningOn = !panel.classList.contains('fullscreen');
    for (const id of VIEW_IDS) {
      if (id !== panelId) byId(id).classList.toggle('hidden', turningOn);
    }
    panel.classList.toggle('fullscreen', turningOn);
    if (turningOn && this._collapsed.delete(panelId)) this._saveCollapsed();
    this._apply();
  }

  // A resize handle only makes sense when both panels it sits between are
  // actually on screen to trade space with each other.
  _updateResizeHandles() {
    const fixedHeight = id => this._isOpen(id) && !byId(id).classList.contains('grow');
    const hide = (id, hidden) => byId(id).classList.toggle('hidden', hidden);
    hide('data-table-resize-handle', !(this._isOpen('projection-panel') && fixedHeight('data-table-panel')));
    hide('sidebar-resize-handle', this._collapsed.has('sidebar'));
    hide('stats-resize-handle', !((this._isOpen('projection-panel') || this._isOpen('data-table-panel')) && fixedHeight('stats-panel')));
  }
}
