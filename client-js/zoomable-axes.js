// Owns the d3.zoom behavior attached to targetEl (the canvas) — no visual
// axis/tick chrome, just the pan/zoom state. Exposes the current transform
// for the owner to use when drawing points and hit-testing; calls onZoom()
// after every zoom/pan/reset so the owner knows to redraw.

export class ZoomableAxes {
  constructor(targetEl, width, height, onZoom) {
    this._targetEl = targetEl;
    this.transform = d3.zoomIdentity;

    this.zoom = d3.zoom()
      .scaleExtent([0.5, 40])
      .extent([[0, 0], [width, height]])
      .on('zoom', event => {
        this.transform = event.transform;
        onZoom();
      });
    d3.select(targetEl).call(this.zoom);
  }

  // A fresh dataset starts unzoomed. Resetting through the zoom behavior
  // itself (rather than just this.transform = d3.zoomIdentity) keeps its
  // internal state in sync, so the next pan/zoom gesture starts correctly,
  // and fires onZoom() synchronously so the caller's first draw is in sync.
  reset() {
    d3.select(this._targetEl).call(this.zoom.transform, d3.zoomIdentity);
  }

  // Recomputes the zoom behavior's pannable/zoomable extent for a new
  // container size — used when the plot's container is resized (e.g. a
  // collapse/fullscreen toggle elsewhere in the layout).
  resize(width, height) {
    this.zoom.extent([[0, 0], [width, height]]);
  }
}
