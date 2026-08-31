// Owns the SVG axis overlay and the d3.zoom behavior attached to targetEl
// (the canvas). Exposes the current transform for the owner to use when
// drawing points and hit-testing; calls onZoom() after every zoom/pan/reset
// so the owner knows to redraw.

export class ZoomableAxes {
  constructor(containerEl, targetEl, width, height, margin, onZoom) {
    this._targetEl = targetEl;
    this.transform = d3.zoomIdentity;

    // Axes are a handful of tick marks, not one element per data point, so
    // a small SVG overlay for just the axis chrome is fine
    // TODO: use d3.axisBottom() and d3.axisLeft() to draw the axes?
    this.svg = d3.select(containerEl).append('svg')
      .attr('class', 'axis-layer')
      .attr('width', containerEl.clientWidth)
      .attr('height', containerEl.clientHeight);
    this.xAxisG = this.svg.append('g').attr('class', 'axis x-axis')
      .attr('transform', `translate(${margin.left},${height + margin.top})`);
    this.yAxisG = this.svg.append('g').attr('class', 'axis y-axis')
      .attr('transform', `translate(${margin.left},${margin.top})`);

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

  update(baseXScale, baseYScale) {
    const zx = this.transform.rescaleX(baseXScale);
    const zy = this.transform.rescaleY(baseYScale);
    this.xAxisG.call(d3.axisBottom(zx).ticks(6));
    this.yAxisG.call(d3.axisLeft(zy).ticks(6));
  }
}
