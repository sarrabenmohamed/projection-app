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
// exposes the interface App has always used — update() plus the setters
// below — so that split never had to touch App at all. setLangFilter() is the one new method, driven by LangFilter
// (owned by App, same as the file input and method select).

import { PointLayer } from './point-layer.js';
import { ZoomableAxes } from './zoomable-axes.js';
import { Tooltip } from './tooltip.js';

// Fisheye lens size (screen pixels, constant regardless of zoom) and
// strength — fixed for now rather than exposed as UI controls. The lens is
// a disk the user drags by its centre grip or its rim.
const DEFAULT_FISHEYE_RADIUS = 110;
const FISHEYE_DISTORTION = 3;
const LENS_GRIP_RADIUS = 14;
const LENS_RIM_GRAB = 8;
const ROTATE_SPEED = 0.008; // radians per dragged pixel
const DRAG_THRESHOLD = 3;   // px of movement before a press counts as a drag, not a click

export class ProjectionView {
  constructor(containerId) {
    // bottom/left grown to fit axis tick labels; the canvas + axis SVG both
    // use this same margin so their coordinate spaces line up exactly.
    this.margin = { top: 10, right: 10, bottom: 30, left: 45 };
    this.el = document.getElementById(containerId);
    this.el.style.position = 'relative';

    this.canvas = document.createElement('canvas');
    this.el.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');
    this._applySize();

    this.pointLayer = new PointLayer(() => this._draw());
    this.axes = new ZoomableAxes(this.canvas, this.width, this.height, () => this._draw(), event => this._blocksPan(event));
    this.tooltip = new Tooltip(this.el);

    this.quadtree = null;
    this._hoveredId = null;
    this._labelBy = 'text';
    this._fisheyeEnabled = false;
    this._fisheyeStyle = 'd3'; // 'd3' | 'geometric' — see point-layer.js's _fisheyeOffset
    this._fisheyeRadius = DEFAULT_FISHEYE_RADIUS; // screen px
    this._lensCenter = null;    // {x, y} screen px within the plot area (margin excluded)
    this._lensDrag = null;
    this._rotateDrag = null;
    this._areaDrag = null;      // {x0, y0, x1, y1} in plot-area screen px while a box-select is being dragged
    this._onAreaSelect = null;
    this._quadtreeDirty = false;
    this._suppressClick = false;
    this._drawRafPending = false;

    this.canvas.addEventListener('click', event => this._onCanvasClick(event));
    this.canvas.addEventListener('mousemove', event => this._onCanvasMouseMove(event));
    this.canvas.addEventListener('mouseleave', () => this._onCanvasMouseLeave());
    this.canvas.addEventListener('mousedown', event => this._onLensMouseDown(event));
    this.canvas.addEventListener('mousedown', event => this._onRotateMouseDown(event));
    this.canvas.addEventListener('mousedown', event => this._onAreaMouseDown(event));
    this.canvas.addEventListener('contextmenu', event => event.preventDefault()); // right-drag is the box-select
  }

  // keepColors: `points` is an edited version of the current set, so
  // languages already colored keep their colors (see PointLayer.setPoints).
  // keepView: the layout is the same one with points added or removed, so
  // the axes and zoom stay as they are — everything already plotted stays
  // exactly where it is on screen (new points may be off-screen until fitView()).
  update(points, onClickPoint, { keepColors = false, keepView = false } = {}) {
    this._onClickPoint = onClickPoint;

    const dims = this.pointLayer.dims;
    keepView = keepView && this.pointLayer.baseXScale && points.length > 0 && (points[0].z !== undefined ? 3 : 2) === dims;
    this.pointLayer.setPoints(points, { keepColors, keepSphere: keepView });
    if (keepView) {
      this._rebuildQuadtree();
      this._draw();
      return;
    }
    this._fitScales();

    // fires the axes' onZoom callback synchronously, which draws the first frame
    this.axes.reset();
  }

  // Frames every point (after an edit kept the old view).
  fitView() {
    this._fitScales();
    this.axes.reset();
  }

  // Called with (points, event) when the user finishes a box-select.
  setAreaSelectHandler(handler) { this._onAreaSelect = handler; }

  // The base (pre-zoom) scales and the hit-test quadtree that go with them.
  // 2D stretches the data extent over the plot area; 3D keeps a fixed
  // equal-aspect scale, so rotating never rescales or squashes the cloud.
  _fitScales() {
    const layer = this.pointLayer;
    if (layer.dims === 3) {
      const r = layer.radius;
      const half = Math.max(1, Math.min(this.width, this.height) / 2 - 20);
      const cx = this.width / 2, cy = this.height / 2;
      layer.setScales(d3.scaleLinear([-r, r], [cx - half, cx + half]), d3.scaleLinear([-r, r], [cy + half, cy - half]));
    } else {
      layer.setScales(
        d3.scaleLinear(d3.extent(layer.points, d => d.x), [20, this.width - 20]),
        d3.scaleLinear(d3.extent(layer.points, d => d.y), [this.height - 20, 20]),
      );
    }
    this._rebuildQuadtree();
  }

  // Rotating moves every point, so the quadtree is rebuilt lazily, on the
  // first hit-test after a rotation rather than on every drag frame.
  _rebuildQuadtree() {
    const layer = this.pointLayer;
    this.quadtree = d3.quadtree()
      .x(d => layer.baseXScale(d.x))
      .y(d => layer.baseYScale(d.y))
      .addAll(layer.points);
    this._quadtreeDirty = false;
  }

  resetView() {
    this.pointLayer.resetRotation();
    this._quadtreeDirty = true;
    this.axes.reset();
  }

  get is3D() { return this.pointLayer.dims === 3; }

  setLangFilter(langs) { this.pointLayer.setLangFilter(langs); }
  setColorMode(mode) { this.pointLayer.setColorMode(mode); }
  getColorLegend() { return this.pointLayer.getColorLegend(); }
  setPinnedPoints(pins) { this.pointLayer.setPinnedPoints(pins); }
  setShowPinnedOnly(enabled) { this.pointLayer.setShowPinnedOnly(enabled); }
  setTokenOverlays(overlaysByPin) { this.pointLayer.setTokenOverlays(overlaysByPin); }
  setLabelBy(field) { this._labelBy = field; }

  // Enabling drops the lens disk in the middle of the plot; the user then
  // drags it wherever they want to magnify.
  setFisheyeEnabled(enabled) {
    this._fisheyeEnabled = enabled;
    this._lensCenter = enabled ? (this._lensCenter || { x: this.width / 2, y: this.height / 2 }) : null;
    this._draw();
  }

  setFisheyeRadius(radius) {
    this._fisheyeRadius = radius;
    this._draw();
  }

  setFisheyeStyle(style) {
    this._fisheyeStyle = style;
    this._draw();
  }

  // Sets this.width/height (plot area, margin excluded) and the canvas's
  // backing-store size (device-pixel-ratio aware) from the container's
  // current clientWidth/clientHeight. Shared by the constructor and
  // resize() so both stay in sync with exactly the same sizing logic.
  _applySize() {
    this.width  = this.el.clientWidth  - this.margin.left - this.margin.right;
    this.height = this.el.clientHeight - this.margin.top  - this.margin.bottom;

    const dpr = window.devicePixelRatio || 1;
    this.canvas.width  = this.el.clientWidth  * dpr;
    this.canvas.height = this.el.clientHeight * dpr;
    this.canvas.style.width  = `${this.el.clientWidth}px`;
    this.canvas.style.height = `${this.el.clientHeight}px`;
    this.ctx.scale(dpr, dpr);
  }

  // Recomputes canvas/axes sizing for a new container size (e.g. after a
  // collapse/fullscreen layout toggle) and rebuilds the base scales +
  // quadtree against the current points, then redraws.
  resize() {
    this._applySize();
    this.axes.resize(this.width, this.height);

    if (this.pointLayer.points.length) this._fitScales();
    if (this._lensCenter) this._lensCenter = this._clampLens(this._lensCenter.x, this._lensCenter.y);
    this._draw();
  }

  _draw() {
    this.ctx.clearRect(0, 0, this.el.clientWidth, this.el.clientHeight);
    this.pointLayer.draw(this.ctx, this.margin, this.axes.transform, this._hoveredId, this._fisheyeFocus());
    if (this._lensCenter) this._drawLens();
    if (this._areaDrag) this._drawAreaBox();
  }

  _drawAreaBox() {
    const { x, y, w, h } = this._areaRect();
    const ctx = this.ctx;
    ctx.save();
    ctx.translate(this.margin.left, this.margin.top);
    ctx.fillStyle = 'rgba(26, 115, 232, 0.12)';
    ctx.fillRect(x, y, w, h);
    ctx.setLineDash([5, 3]);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = '#1a73e8';
    ctx.strokeRect(x, y, w, h);
    ctx.restore();
  }

  _areaRect() {
    const a = this._areaDrag;
    return { x: Math.min(a.x0, a.x1), y: Math.min(a.y0, a.y1), w: Math.abs(a.x1 - a.x0), h: Math.abs(a.y1 - a.y0) };
  }

  // The lens lives in screen space (so panning/zooming doesn't move the
  // disk); the point layer wants it in its pre-zoom "local" space, with the
  // radius shrunk by the zoom so it stays a constant on-screen size.
  _fisheyeFocus() {
    if (!this._lensCenter) return null;
    const [x, y] = this.axes.transform.invert([this._lensCenter.x, this._lensCenter.y]);
    return {
      x, y, radius: this._fisheyeRadius / this.axes.transform.k, distortion: FISHEYE_DISTORTION,
      style: this._fisheyeStyle,
    };
  }

  // Visible boundary: a light tint, a dark outline with a white halo (so it
  // reads over both dense clusters and empty space), and a centre grip.
  _drawLens() {
    const ctx = this.ctx;
    const cx = this.margin.left + this._lensCenter.x;
    const cy = this.margin.top + this._lensCenter.y;
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, this._fisheyeRadius, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(70, 120, 200, 0.07)';
    ctx.fill();
    ctx.lineWidth = 5;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
    ctx.stroke();
    ctx.lineWidth = 2;
    ctx.strokeStyle = this._lensDrag ? '#1d4e9e' : '#3b6fc4';
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(cx, cy, LENS_GRIP_RADIUS, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.fillStyle = '#3b6fc4';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('✥', cx, cy + 1);
    ctx.restore();
  }

  _clampLens(x, y) {
    return {
      x: Math.min(Math.max(x, 0), this.width),
      y: Math.min(Math.max(y, 0), this.height),
    };
  }

  // Mouse position in the same plot-area screen space as _lensCenter.
  _plotPosition(event) {
    const rect = this.canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left - this.margin.left, y: event.clientY - rect.top - this.margin.top };
  }

  // Only the grip and the rim grab the lens; the interior stays free so
  // panning and point hover/click still work under it.
  _lensHit(event) {
    if (!this._lensCenter) return false;
    const p = this._plotPosition(event);
    const dist = Math.hypot(p.x - this._lensCenter.x, p.y - this._lensCenter.y);
    return dist <= LENS_GRIP_RADIUS || Math.abs(dist - this._fisheyeRadius) <= LENS_RIM_GRAB;
  }

  _onLensMouseDown(event) {
    if (event.button !== 0 || !this._lensHit(event)) return;
    const p = this._plotPosition(event);
    this._lensDrag = { dx: this._lensCenter.x - p.x, dy: this._lensCenter.y - p.y };
    this.canvas.style.cursor = 'grabbing';
    this.tooltip.hide();
    event.preventDefault();

    const onMove = e => {
      const q = this._plotPosition(e);
      this._lensCenter = this._clampLens(q.x + this._lensDrag.dx, q.y + this._lensDrag.dy);
      this._requestDraw();
    };
    const onUp = () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      this._lensDrag = null;
      this._suppressClick = true; // the click that follows the drop shouldn't select a point
      setTimeout(() => { this._suppressClick = false; }, 0);
      this.canvas.style.cursor = 'grab';
      this._draw();
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }

  // In 3D a plain left-drag rotates, so panning moves to shift+drag.
  _blocksPan(event) {
    return this._lensHit(event) || (this.is3D && !event.shiftKey);
  }

  // Counts as a click-suppressing drag once the pointer moved far enough.
  _endDragGesture(moved) {
    if (!moved) return;
    this._suppressClick = true; // the click that follows the release shouldn't select a point
    setTimeout(() => { this._suppressClick = false; }, 0);
  }

  _onRotateMouseDown(event) {
    if (!this.is3D || event.button !== 0 || event.ctrlKey || event.shiftKey || this._lensHit(event)) return;
    event.preventDefault();
    this.tooltip.hide();
    this._hoveredId = null;
    this._rotateDrag = { x: event.clientX, y: event.clientY, startX: event.clientX, startY: event.clientY, moved: false };
    this.canvas.style.cursor = 'grabbing';

    const onMove = e => {
      const drag = this._rotateDrag;
      if (!drag.moved && Math.hypot(e.clientX - drag.startX, e.clientY - drag.startY) > DRAG_THRESHOLD) drag.moved = true;
      this.pointLayer.rotateBy((e.clientX - drag.x) * ROTATE_SPEED, (e.clientY - drag.y) * ROTATE_SPEED);
      drag.x = e.clientX;
      drag.y = e.clientY;
      this._quadtreeDirty = true;
      this._requestDraw();
    };
    const onUp = () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      this._endDragGesture(this._rotateDrag.moved);
      this._rotateDrag = null;
      this.canvas.style.cursor = 'grab';
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }

  // Right-drag (or ctrl+drag, for trackpads) draws a box; every selectable
  // point inside it is handed to the area-select handler on release.
  _onAreaMouseDown(event) {
    const isAreaGesture = event.button === 2 || (event.button === 0 && event.ctrlKey);
    if (!isAreaGesture || !this._onAreaSelect) return;
    event.preventDefault();
    this.tooltip.hide();
    const start = this._plotPosition(event);
    this._areaDrag = { x0: start.x, y0: start.y, x1: start.x, y1: start.y };

    const onMove = e => {
      const q = this._clampLens(this._plotPosition(e).x, this._plotPosition(e).y);
      this._areaDrag.x1 = q.x;
      this._areaDrag.y1 = q.y;
      this._requestDraw();
    };
    const onUp = () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      const { x, y, w, h } = this._areaRect();
      this._areaDrag = null;
      this._endDragGesture(true);
      this._draw();
      if (w < DRAG_THRESHOLD && h < DRAG_THRESHOLD) return;
      const [lx0, ly0] = this.axes.transform.invert([x, y]);
      const [lx1, ly1] = this.axes.transform.invert([x + w, y + h]);
      const inside = this.pointLayer.pointsInRect(lx0, ly0, lx1, ly1);
      if (inside.length) this._onAreaSelect(inside);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }

  // Coalesces redraws to at most one per animation frame. mousemove can
  // fire far more often than the browser paints, and with the fisheye
  // lens active every point's position is recomputed on every redraw
  // (unlike a plain hover-highlight redraw, which is rare) — without this,
  // the combination of high-frequency events and per-point work drops
  // frames visibly wherever points are actually moving under the lens.
  _requestDraw() {
    if (this._drawRafPending) return;
    this._drawRafPending = true;
    requestAnimationFrame(() => {
      this._drawRafPending = false;
      this._draw();
    });
  }

  // Converts a mouse event's screen position into the same "local"
  // (pre-zoom) coordinate space baseXScale/baseYScale output — i.e. what
  // the quadtree is indexed in and what fisheyeFocus is expressed in.
  _screenToLocal(event) {
    const rect = this.canvas.getBoundingClientRect();
    const screenX = event.clientX - rect.left - this.margin.left;
    const screenY = event.clientY - rect.top - this.margin.top;
    return this.axes.transform.invert([screenX, screenY]);
  }

  // Finds the point under the mouse (if any). The quadtree is indexed in
  // unzoomed (base) coordinates, so the mouse position — in current, zoomed
  // screen space — has to be run back through the inverse zoom transform
  // first; the hit radius is likewise scaled by 1/k so it represents the same
  // on-screen distance at any zoom level.
  //
  // While the "show pinned only" isolate is
  // active, everything outside it is invisible (opacity 0) — the
  // quadtree still spatially indexes those points, so without this check
  // a click/hover could land on a point the user can't see. Treating
  // that as no-hit means such a click falls through to the background
  // handler, clearing the isolation instead of silently selecting
  // something hidden.
  _pointAt(event) {
    const [x, y] = this._screenToLocal(event);
    const radius = 8 / this.axes.transform.k;
    if (this.pointLayer.fisheyeActive) return this.pointLayer.pointAtDisplayed(x, y, radius);
    if (this._quadtreeDirty) this._rebuildQuadtree();
    const found = this.quadtree.find(x, y, radius);
    if (found && this.pointLayer.isIsolatedOut(found)) return undefined;
    return found;
  }

  _overlayPointAt(event) {
    const [x, y] = this._screenToLocal(event);
    return this.pointLayer.tokenOverlayPointAt(x, y, 8 / this.axes.transform.k);
  }

  _onCanvasClick(event) {
    if (this._suppressClick || this._lensHit(event)) return;
    if (this._overlayPointAt(event)) return; // token overlay points aren't clickable, just hoverable
    const d = this._pointAt(event);
    if (d) this._onClickPoint?.(d);
  }

  // A token point always shows its own token; the selected hover label
  // (e.g. language) is appended as context rather than replacing it.
  _overlayLabel(point) {
    const token = String(point.text);
    if (this._labelBy === 'text') return token;
    return `${token} (${point[this._labelBy]})`;
  }

  _onCanvasMouseMove(event) {
    if (this._lensDrag || this._rotateDrag || this._areaDrag) return;
    if (this._lensHit(event)) {
      this._hoveredId = null;
      this.tooltip.hide();
      this.canvas.style.cursor = 'grab';
      this._requestDraw();
      return;
    }

    const overlayHit = this._overlayPointAt(event);
    if (overlayHit) {
      const rect = this.el.getBoundingClientRect();
      this.tooltip.show(event.clientX - rect.left, event.clientY - rect.top, this._overlayLabel(overlayHit));
      this.canvas.style.cursor = 'default';
      this._hoveredId = null;
      this._requestDraw();
      return;
    }

    const d = this._pointAt(event);
    this._hoveredId = d ? d.id : null;
    this._requestDraw();
    if (d) {
      const rect = this.el.getBoundingClientRect();
      const label = String(d[this._labelBy]);
      this.tooltip.show(event.clientX - rect.left, event.clientY - rect.top, d.overlaid ? `${label} · overlaid` : label);
      this.canvas.style.cursor = 'pointer';
    } else {
      this.tooltip.hide();
      this.canvas.style.cursor = this.is3D ? 'grab' : 'default';
    }
  }

  _onCanvasMouseLeave() {
    this.tooltip.hide();
    if (!this._lensDrag && !this._rotateDrag) this.canvas.style.cursor = 'default';
    if (this._hoveredId !== null) {
      this._hoveredId = null;
      this._draw();
    }
  }
}
