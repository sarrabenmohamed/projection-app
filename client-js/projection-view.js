// Projection view — projection view only, D3 OO structure.
//
// Renders points on a single <canvas> instead of one SVG <circle> per point —
// at thousands of points, restyling that many DOM nodes for a dim/undim
// transition is the actual bottleneck (measured directly), not the 200ms
// transition itself. Canvas redraws of the whole point set are cheap even at
// 60fps, so click/hover behavior stays identical, and hit-testing (click/
// hover) uses a d3.quadtree instead of DOM listeners.
//
// Composes three small collaborators (PointLayer, ZoomableAxes, Tooltip) but
// exposes the same interface App has always used for click-isolate — update()
// plus highlightParallel/clearHighlight — so that split never had to touch
// App at all. setLangFilter() is the one new method, driven by LangFilter
// (owned by App, same as the file input and method select).

import { PointLayer } from './point-layer.js';
import { ZoomableAxes } from './zoomable-axes.js';
import { Tooltip } from './tooltip.js';

export class ProjectionView {
  constructor(containerId) {
    // bottom/left grown to fit axis tick labels; the canvas + axis SVG both
    // use this same margin so their coordinate spaces line up exactly.
    this.margin = { top: 10, right: 10, bottom: 30, left: 45 };
    this.el = document.getElementById(containerId);
    this.el.style.position = 'relative';
    this.width  = this.el.clientWidth  - this.margin.left - this.margin.right;
    this.height = this.el.clientHeight - this.margin.top  - this.margin.bottom;

    const dpr = window.devicePixelRatio || 1;
    this.canvas = document.createElement('canvas');
    this.canvas.width  = this.el.clientWidth  * dpr;
    this.canvas.height = this.el.clientHeight * dpr;
    this.canvas.style.width  = `${this.el.clientWidth}px`;
    this.canvas.style.height = `${this.el.clientHeight}px`;
    this.el.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');
    this.ctx.scale(dpr, dpr);

    this.pointLayer = new PointLayer(() => this._draw());
    this.axes = new ZoomableAxes(this.el, this.canvas, this.width, this.height, this.margin, () => {
      this._draw();
      this._updateAxes();
    });
    this.tooltip = new Tooltip(this.el);

    this.quadtree = null;
    this._hoveredId = null;

    this.canvas.addEventListener('click', event => this._onCanvasClick(event));
    this.canvas.addEventListener('mousemove', event => this._onCanvasMouseMove(event));
    this.canvas.addEventListener('mouseleave', () => this._hideTooltip());
  }

  update(points, onClickPoint) {
    this._onClickPoint = onClickPoint;

    const baseXScale = d3.scaleLinear(d3.extent(points, d => d.x), [20, this.width - 20]);
    const baseYScale = d3.scaleLinear(d3.extent(points, d => d.y), [this.height - 20, 20]);
    this.pointLayer.setPoints(points, baseXScale, baseYScale);

    this.quadtree = d3.quadtree()
      .x(d => baseXScale(d.x))
      .y(d => baseYScale(d.y))
      .addAll(this.pointLayer.points);

    // fires the axes' onZoom callback synchronously, which draws the first frame
    this.axes.reset();
  }

  highlightParallel(groupKey) { this.pointLayer.highlightGroup(groupKey); }
  clearHighlight() { this.pointLayer.clear(); }
  setLangFilter(langs) { this.pointLayer.setLangFilter(langs); }
  setTokenOverlay(tokenPoints) { this.pointLayer.setTokenOverlay(tokenPoints); }
  clearTokenOverlay() { this.pointLayer.clearTokenOverlay(); }
  setSelectedId(id) { this.pointLayer.setSelectedId(id); }
  clearSelectedId() { this.pointLayer.clearSelectedId(); }

  _draw() {
    this.ctx.clearRect(0, 0, this.el.clientWidth, this.el.clientHeight);
    this.pointLayer.draw(this.ctx, this.margin, this.axes.transform, this._hoveredId);
  }

  _updateAxes() {
    this.axes.update(this.pointLayer.baseXScale, this.pointLayer.baseYScale);
  }

  // Finds the point under the mouse (if any). The quadtree is indexed in
  // unzoomed (base) coordinates, so the mouse position — in current, zoomed
  // screen space — has to be run back through the inverse zoom transform
  // first; the hit radius is likewise scaled by 1/k so it represents the same
  // on-screen distance at any zoom level.
  //
  // While a group is isolated, everything outside it is invisible (opacity
  // 0) — the quadtree still spatially indexes those points, so without this
  // check a click/hover could land on a point the user can't see. Treating
  // that as no-hit means such a click falls through to the background
  // handler, clearing the isolation instead of silently selecting something
  // hidden.
  _pointAt(event) {
    const rect = this.canvas.getBoundingClientRect();
    const screenX = event.clientX - rect.left - this.margin.left;
    const screenY = event.clientY - rect.top - this.margin.top;
    const [x, y] = this.axes.transform.invert([screenX, screenY]);
    const found = this.quadtree.find(x, y, 8 / this.axes.transform.k);

    const activeGroupKey = this.pointLayer.getActiveGroupKey();
    if (activeGroupKey !== null && found && found.groupKey !== activeGroupKey) {
      return undefined;
    }
    return found;
  }

  _overlayPointAt(event) {
    const rect = this.canvas.getBoundingClientRect();
    const screenX = event.clientX - rect.left - this.margin.left;
    const screenY = event.clientY - rect.top - this.margin.top;
    const [x, y] = this.axes.transform.invert([screenX, screenY]);
    return this.pointLayer.tokenOverlayPointAt(x, y, 8 / this.axes.transform.k);
  }

  _onCanvasClick(event) {
    if (this._overlayPointAt(event)) return; // token overlay points aren't clickable, just hoverable
    const d = this._pointAt(event);
    if (d) {
      this._onClickPoint?.(d);
    } else {
      this._onBackgroundClick?.();
    }
  }

  _onCanvasMouseMove(event) {
    const overlayHit = this._overlayPointAt(event);
    if (overlayHit) {
      const rect = this.el.getBoundingClientRect();
      this.tooltip.show(event.clientX - rect.left, event.clientY - rect.top, overlayHit.text);
      this.canvas.style.cursor = 'default';
      if (this._hoveredId !== null) {
        this._hoveredId = null;
        this._draw();
      }
      return;
    }

    const d = this._pointAt(event);
    const hoveredId = d ? d.id : null;
    if (hoveredId !== this._hoveredId) {
      this._hoveredId = hoveredId;
      this._draw();
    }
    if (d) {
      const rect = this.el.getBoundingClientRect();
      this.tooltip.show(event.clientX - rect.left, event.clientY - rect.top, `${d.lang}: ${d.text}`);
      this.canvas.style.cursor = 'pointer';
    } else {
      this._hideTooltip();
    }
  }

  _hideTooltip() {
    this.tooltip.hide();
    this.canvas.style.cursor = 'default';
    if (this._hoveredId !== null) {
      this._hoveredId = null;
      this._draw();
    }
  }
}
