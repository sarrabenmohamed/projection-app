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
//
// Pinned points get a colored ring; `pins` is the full current list every
// time (not incremental), computed by App from its selection set — so a
// point that stops being pinned just isn't in the next call's array.

const NEUTRAL_COLOR = '#bbb';
const FILTERED_OUT_OPACITY = 0.3;
const ISOLATED_OUT_OPACITY = 0;

// Circular "magnifying glass" distortion: a point at offset (dx, dy) from
// the lens focus, within `radius`, is pushed further from the focus (k > 1
// near the center) while points near the edge of the lens compress back
// toward their real position (k -> 1 exactly at d = radius, so there's no
// visible seam where the lens ends). Points beyond `radius` are untouched.
function _fisheyeOffset(dx, dy, radius, distortion) {
  const d = Math.sqrt(dx * dx + dy * dy);
  if (d >= radius) return [dx, dy];
  const k = (radius * (distortion + 1)) / (distortion * d + radius);
  return [dx * k, dy * k];
}

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
    this._tokenOverlayEntries = [];
    this._pins = [];
  }

  setPoints(points, baseXScale, baseYScale) {
    this._activeGroupKey = null;
    this._selectedLangs = new Set();
    this._langColorScale = null;
    this._tokenOverlayEntries = [];
    this._pins = [];
    this.points = points.map(p => ({ ...p, opacity: 1 }));
    this.baseXScale = baseXScale;
    this.baseYScale = baseYScale;
  }

  // Pinned points get a colored ring; `pins` is the full current list every
  // time (not incremental), computed by App from its selection set — so a
  // point that stops being pinned just isn't in the next call's array.
  setPinnedPoints(pins) {
    this._pins = pins; // Array<{id, color, focused}>
    this._onChange();
  }

  // Token points for pinned sentences, embedded into the
  // same UMAP space as `this.points` via the backend's shared reducer —
  // drawn on the same axes but tracked separately so they're unaffected by
  // the isolate-fade opacity animation and don't participate in click-to-
  // isolate grouping. Each pin's tokens are rendered in that pin's color.
  setTokenOverlays(overlaysByPin) {
    this._tokenOverlayEntries = [];
    for (const [pinId, points] of overlaysByPin) {
      const pin = this._pins.find(p => p.id === pinId);
      const color = pin ? pin.color : '#e63946';
      for (const point of points) {
        this._tokenOverlayEntries.push({ ...point, pinId, color });
      }
    }
    this._onChange();
  }

  tokenOverlayPointAt(x, y, radius) {
    for (const d of this._tokenOverlayEntries) {
      const dx = this.baseXScale(d.x) - x;
      const dy = this.baseYScale(d.y) - y;
      if (Math.hypot(dx, dy) <= radius) return d;
    }
    return undefined;
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

  // A point is drawn "on top" of the plain background if it's pinned or
  // actively colored by the language filter — otherwise a gray point later
  // in the array would sit on top of a colored/pinned point earlier in it
  // (canvas paints in draw order, not by any notion of importance) and
  // visually bury it. Two passes — background first, on-top points last —
  // guarantee that never happens, regardless of the points' original order.
  _isOnTop(d) {
    return this._selectedLangs.has(d.lang) || this._pins.some(p => p.id === d.id);
  }

  // fisheyeFocus, when active, is {x, y, radius, distortion} in the same
  // "local" (pre-zoom) coordinate space as baseXScale/baseYScale output —
  // see ProjectionView._updateFisheyeFocus for how it's computed.
  draw(ctx, margin, transform, hoveredId, fisheyeFocus) {
    ctx.save();
    ctx.translate(margin.left, margin.top);
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.k, transform.k);

    const background = [], foreground = [];
    for (const d of this.points) (this._isOnTop(d) ? foreground : background).push(d);
    for (const d of background) this._drawPoint(ctx, d, transform, hoveredId, fisheyeFocus);
    for (const d of foreground) this._drawPoint(ctx, d, transform, hoveredId, fisheyeFocus);

    for (const d of this._tokenOverlayEntries) {
      let cx = this.baseXScale(d.x), cy = this.baseYScale(d.y);
      if (fisheyeFocus) {
        const [ox, oy] = _fisheyeOffset(cx - fisheyeFocus.x, cy - fisheyeFocus.y, fisheyeFocus.radius, fisheyeFocus.distortion);
        cx = fisheyeFocus.x + ox;
        cy = fisheyeFocus.y + oy;
      }
      const size = 5 / transform.k;
      ctx.beginPath();
      ctx.moveTo(cx, cy - size);
      ctx.lineTo(cx + size, cy);
      ctx.lineTo(cx, cy + size);
      ctx.lineTo(cx - size, cy);
      ctx.closePath();
      ctx.globalAlpha = 0.9;
      ctx.fillStyle = d.color;
      ctx.fill();
      ctx.lineWidth = 0.75 / transform.k;
      ctx.strokeStyle = '#fff';
      ctx.stroke();
    }
    ctx.restore();
    ctx.globalAlpha = 1;
  }

  _drawPoint(ctx, d, transform, hoveredId, fisheyeFocus) {
    let cx = this.baseXScale(d.x), cy = this.baseYScale(d.y);
    if (fisheyeFocus) {
      const [ox, oy] = _fisheyeOffset(cx - fisheyeFocus.x, cy - fisheyeFocus.y, fisheyeFocus.radius, fisheyeFocus.distortion);
      cx = fisheyeFocus.x + ox;
      cy = fisheyeFocus.y + oy;
    }
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
    const pin = this._pins.find(p => p.id === d.id);
    if (pin) {
      ctx.beginPath();
      ctx.arc(cx, cy, 8.5 / transform.k, 0, 2 * Math.PI);
      ctx.globalAlpha = d.opacity;
      ctx.lineWidth = (pin.focused ? 2.6 : 1.8) / transform.k;
      ctx.strokeStyle = pin.color;
      ctx.stroke();
    }
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
