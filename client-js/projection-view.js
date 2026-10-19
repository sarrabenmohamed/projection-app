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

// Fisheye lens size (screen pixels, constant regardless of zoom) and
// strength — fixed for now rather than exposed as UI controls.
const FISHEYE_RADIUS = 180;
const FISHEYE_DISTORTION = 3;

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
    this.axes = new ZoomableAxes(this.canvas, this.width, this.height, () => this._draw());
    this.tooltip = new Tooltip(this.el);

    this.quadtree = null;
    this._hoveredId = null;
    this._labelBy = 'text';
    this._fisheyeEnabled = false;
    this._fisheyeStyle = 'd3'; // 'd3' | 'geometric' — see point-layer.js's _fisheyeOffset
    this._fisheyeFocus = null;
    this._drawRafPending = false;

    this.canvas.addEventListener('click', event => this._onCanvasClick(event));
    this.canvas.addEventListener('mousemove', event => this._onCanvasMouseMove(event));
    this.canvas.addEventListener('mouseleave', () => this._onCanvasMouseLeave());
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
  setFamilyColorMode(enabled) { this.pointLayer.setFamilyColorMode(enabled); }
  getFamilyLegend() { return this.pointLayer.getFamilyLegend(); }
  setPinnedPoints(pins) { this.pointLayer.setPinnedPoints(pins); }
  setShowPinnedOnly(enabled) { this.pointLayer.setShowPinnedOnly(enabled); }
  setTokenOverlays(overlaysByPin) { this.pointLayer.setTokenOverlays(overlaysByPin); }
  setLabelBy(field) { this._labelBy = field; }

  // Turning the lens off clears its focus immediately (rather than waiting
  // for the next mousemove) so the plot snaps back to its undistorted
  // layout as soon as the toggle is unchecked, even if the mouse doesn't move.
  setFisheyeEnabled(enabled) {
    this._fisheyeEnabled = enabled;
    if (!enabled) {
      this._fisheyeFocus = null;
      this._draw();
    }
  }

  // Changing style while the lens is already active re-renders with the
  // new curve immediately, at the same focus position, rather than
  // waiting for the next mousemove.
  setFisheyeStyle(style) {
    this._fisheyeStyle = style;
    if (this._fisheyeFocus) {
      this._fisheyeFocus = { ...this._fisheyeFocus, style };
      this._draw();
    }
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

    if (this.pointLayer.points.length) {
      const baseXScale = d3.scaleLinear(d3.extent(this.pointLayer.points, d => d.x), [20, this.width - 20]);
      const baseYScale = d3.scaleLinear(d3.extent(this.pointLayer.points, d => d.y), [this.height - 20, 20]);
      this.pointLayer.baseXScale = baseXScale;
      this.pointLayer.baseYScale = baseYScale;
      this.quadtree = d3.quadtree()
        .x(d => baseXScale(d.x))
        .y(d => baseYScale(d.y))
        .addAll(this.pointLayer.points);
    }
    this._draw();
  }

  _draw() {
    this.ctx.clearRect(0, 0, this.el.clientWidth, this.el.clientHeight);
    this.pointLayer.draw(this.ctx, this.margin, this.axes.transform, this._hoveredId, this._fisheyeFocus);
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
  // While an isolate (translations group, or "show pinned only") is
  // active, everything outside it is invisible (opacity 0) — the
  // quadtree still spatially indexes those points, so without this check
  // a click/hover could land on a point the user can't see. Treating
  // that as no-hit means such a click falls through to the background
  // handler, clearing the isolation instead of silently selecting
  // something hidden.
  _pointAt(event) {
    const [x, y] = this._screenToLocal(event);
    const found = this.quadtree.find(x, y, 8 / this.axes.transform.k);
    if (found && this.pointLayer.isIsolatedOut(found)) return undefined;
    return found;
  }

  _overlayPointAt(event) {
    const [x, y] = this._screenToLocal(event);
    return this.pointLayer.tokenOverlayPointAt(x, y, 8 / this.axes.transform.k);
  }

  // Hit-testing (_pointAt/_overlayPointAt above) always uses the real,
  // undistorted mouse position — the lens is a visual preview only, so
  // clicking exactly where a point visually moved to (rather than its real
  // position) won't select it. Radius is in screen pixels divided by the
  // current zoom level, same pattern as the hit-test radius above, so the
  // lens is a constant on-screen size regardless of zoom.
  _updateFisheyeFocus(event) {
    if (!this._fisheyeEnabled) {
      this._fisheyeFocus = null;
      return;
    }
    const [x, y] = this._screenToLocal(event);
    this._fisheyeFocus = {
      x, y, radius: FISHEYE_RADIUS / this.axes.transform.k, distortion: FISHEYE_DISTORTION,
      style: this._fisheyeStyle,
    };
  }

  _onCanvasClick(event) {
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
    this._updateFisheyeFocus(event);

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
      this.tooltip.show(event.clientX - rect.left, event.clientY - rect.top, String(d[this._labelBy]));
      this.canvas.style.cursor = 'pointer';
    } else {
      // Just hovering empty canvas space, not leaving it — hide the
      // tooltip only. Do NOT clear _fisheyeFocus here: the lens should
      // stay active over gaps between points, not just exactly on one;
      // _onCanvasMouseLeave (actual mouseleave) is the only place that
      // clears it.
      this.tooltip.hide();
      this.canvas.style.cursor = 'default';
    }
  }

  _onCanvasMouseLeave() {
    this.tooltip.hide();
    this.canvas.style.cursor = 'default';
    const hadFisheyeFocus = this._fisheyeFocus !== null;
    this._fisheyeFocus = null;
    if (this._hoveredId !== null || hadFisheyeFocus) {
      this._hoveredId = null;
      this._draw();
    }
  }
}
