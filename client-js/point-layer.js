// Owns the point data, their colors and per-point opacity, and paints them
// onto a given canvas context. Hides the whole d3.timer-driven opacity
// animation behind highlightGroup/clear/restoreActive/setLangFilter.
//
// Color is filter-driven, not per-language-always-on: past a handful of
// languages, giving every one its own hue stops being readable (colors
// repeat, legends become unusable). So by default every point is neutral
// gray; only languages explicitly selected via the language filter get a
// real color, and everything else dims to context. With nothing selected,
// every point is neutral gray at full opacity — the resting state shows
// shape, not identity.

const NEUTRAL_COLOR = '#bbb';
const FILTERED_OUT_OPACITY = 0.3;
const ISOLATED_OUT_OPACITY = 0;

export class PointLayer {
  constructor(onChange) {
    this._onChange = onChange; // called after every opacity update, so the owner redraws
    this.points = [];
    this.baseXScale = null;
    this.baseYScale = null;
    this._timer = null;
    this._activeGroupKey = null;
    this._selectedLangs = new Set();
    this._langColorScale = null;
  }

  setPoints(points, baseXScale, baseYScale) {
    this._activeGroupKey = null;
    this._selectedLangs = new Set();
    this._langColorScale = null;
    this.points = points.map(p => ({ ...p, opacity: 1 }));
    this.baseXScale = baseXScale;
    this.baseYScale = baseYScale;
  }

  // langs = the currently selected subset from the language filter (may be
  // empty). Colors are assigned fresh among just this subset, so 3 selected
  // languages get 3 clearly different hues regardless of how many languages
  // exist overall.
  setLangFilter(langs) {
    this._selectedLangs = new Set(langs);
    this._langColorScale = d3.scaleOrdinal([...this._selectedLangs].sort(), d3.schemeTableau10);
    if (this._activeGroupKey !== null) {
      this._animateTo(d => d.groupKey === this._activeGroupKey ? 1 : ISOLATED_OUT_OPACITY);
    } else {
      this._animateTo(d => this._baseOpacity(d));
    }
  }

  _baseOpacity(d) {
    if (this._selectedLangs.size === 0) return 1;
    return this._selectedLangs.has(d.lang) ? 1 : FILTERED_OUT_OPACITY;
  }

  _colorFor(d) {
    return this._selectedLangs.has(d.lang) ? this._langColorScale(d.lang) : NEUTRAL_COLOR;
  }

  draw(ctx, margin, transform, hoveredId) {
    ctx.save();
    ctx.translate(margin.left, margin.top);
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.k, transform.k);
    for (const d of this.points) {
      const cx = this.baseXScale(d.x), cy = this.baseYScale(d.y);
      const hovered = d.id === hoveredId;
      ctx.beginPath();
      ctx.arc(cx, cy, 5 / transform.k, 0, 2 * Math.PI);
      ctx.globalAlpha = d.opacity * 0.8;
      ctx.fillStyle = this._colorFor(d);
      ctx.fill();
      ctx.globalAlpha = d.opacity;
      ctx.lineWidth = (hovered ? 1.5 : 0.5) / transform.k;
      ctx.strokeStyle = hovered ? '#333' : '#fff';
      ctx.stroke();
    }
    ctx.restore();
    ctx.globalAlpha = 1;
  }

  // Animates every point's opacity toward targetOpacityFn(d) — the canvas
  // equivalent of a D3 selection.transition().style(...).
  //
  // Captures the points array itself (not just re-reading this.points inside
  // the timer): the timer callback fires on a later animation frame, and if
  // setPoints() replaces this.points with a new (different-length) array
  // before then, indexing into the freshly-captured startOpacities/targets
  // arrays using the NEW array's length reads past their end -> undefined ->
  // NaN opacity for every point. Operating on the array reference captured
  // here means a stale in-flight animation only ever touches its own array.
  _animateTo(targetOpacityFn, duration = 200) {
    if (this._timer) this._timer.stop();
    const points = this.points;
    const startOpacities = points.map(d => d.opacity);
    const targets = points.map(targetOpacityFn);
    this._timer = d3.timer(elapsed => {
      const t = Math.min(1, elapsed / duration);
      const e = d3.easeCubic(t);
      for (let i = 0; i < points.length; i++) {
        points[i].opacity = startOpacities[i] + (targets[i] - startOpacities[i]) * e;
      }
      this._onChange();
      if (t >= 1) this._timer.stop();
    });
  }

  // Isolate every point in the same group as the clicked one (same groupKey)
  // — ex4's highlightNeighborhood, with "neighbour" redefined as "same
  // sentence, other language" instead of graph-adjacency — by fading
  // everything else out completely. Wins over the language filter's base
  // state while active; clear()/restoreActive() below return to that base
  // state rather than to full opacity.
  highlightGroup(groupKey) {
    this._activeGroupKey = groupKey;
    this._animateTo(d => d.groupKey === groupKey ? 1 : ISOLATED_OUT_OPACITY);
  }

  clear() {
    this._activeGroupKey = null;
    this._animateTo(d => this._baseOpacity(d));
  }

  restoreActive() {
    if (this._activeGroupKey !== null) {
      this.highlightGroup(this._activeGroupKey);
    } else {
      this.clear();
    }
  }

  // So the owner can tell whether an isolation is active without reaching
  // into a private field — used to keep hit-testing off points that are
  // currently invisible (opacity 0, isolated out).
  getActiveGroupKey() {
    return this._activeGroupKey;
  }
}
