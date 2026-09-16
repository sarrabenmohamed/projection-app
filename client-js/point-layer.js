// Owns the point data, their colors and per-point opacity, and paints them
// onto a given canvas context. Hides the whole d3.timer-driven opacity
// animation behind highlightGroup/clear/restoreActive/setLangFilter.
//
// Two coloring modes, mutually exclusive (see setFamilyColorMode):
// - Family mode (default): every point is colored by its language's
//   linguistic family — always on, no filtering. `family` is a field the
//   backend already attaches to every point (server.py's family_of()),
//   not something this class computes.
// - Language-filter mode: the pre-existing behavior. Every point is
//   neutral gray by default; only languages explicitly selected via the
//   language filter get a real color, and everything else dims to
//   context. Filter-driven rather than per-language-always-on because
//   past a handful of languages, giving every one its own hue stops
//   being readable (colors repeat, legends become unusable).
//
// Pinned points get a colored ring; `pins` is the full current list every
// time (not incremental), computed by App from its selection set — so a
// point that stops being pinned just isn't in the next call's array.

const NEUTRAL_COLOR = '#bbb';
const FILTERED_OUT_OPACITY = 0.3;
const ISOLATED_OUT_OPACITY = 0;
const POINT_RADIUS = 5; // local-unit render radius before the 1/transform.k scale-compensation
const FISHEYE_RELAX_ITERATIONS = 4;

// Circular "magnifying glass" distortion: a point at offset (dx, dy) from
// the lens focus, within `radius`, is pushed further from the focus (k > 1
// near the center) while points near the edge of the lens compress back
// toward their real position (k -> 1 exactly at d = radius, so there's no
// visible seam where the lens ends). Points beyond `radius` are untouched.
//
// This alone can't prevent two points from still overlapping after
// magnification: two points equidistant from the focus get pushed by the
// identical factor k, so they stay exactly as overlapped, just farther
// from center. See _computeFisheyePositions for the pass that fixes that.
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
    this._familyModeEnabled = true; // a view preference, not per-dataset — setPoints() below doesn't reset it
    this._familyColorScale = null;
    this._showPinnedOnly = false;
  }

  setPoints(points, baseXScale, baseYScale) {
    this._activeGroupKey = null;
    this._selectedLangs = new Set();
    this._langColorScale = null;
    this._tokenOverlayEntries = [];
    this._pins = [];
    this._showPinnedOnly = false;
    this.points = points.map(p => ({ ...p, opacity: 1 }));
    const families = Array.from(new Set(this.points.map(p => p.family))).sort();
    this._familyColorScale = d3.scaleOrdinal(families, d3.schemeTableau10);
    this.baseXScale = baseXScale;
    this.baseYScale = baseYScale;
  }

  // Family mode and the language filter are mutually exclusive views of
  // color — switching modes re-triggers whatever opacity target is
  // currently active so the fade animates consistently with every other
  // opacity change.
  setFamilyColorMode(enabled) {
    this._familyModeEnabled = enabled;
    this._animateTo(d => this._opacityTarget(d));
  }

  // The family -> color mapping actually in use for the current dataset
  // (only the families present, not the full fixed palette) — used by
  // App to render a legend. Empty until setPoints() has built the scale.
  getFamilyLegend() {
    if (!this._familyColorScale) return [];
    return this._familyColorScale.domain().map(family => ({ family, color: this._familyColorScale(family) }));
  }

  // Isolates the canvas to just the pinned points — takes precedence over
  // an active translations isolate (see _opacityTarget), matching
  // DataTable's own show_selected filter's narrowing behavior.
  setShowPinnedOnly(enabled) {
    this._showPinnedOnly = enabled;
    this._animateTo(d => this._opacityTarget(d));
  }

  // Pinned points get a colored ring; `pins` is the full current list every
  // time (not incremental), computed by App from its selection set — so a
  // point that stops being pinned just isn't in the next call's array.
  // Which points count as "pinned" can itself affect who's isolated-out
  // (see setShowPinnedOnly), so this re-triggers the opacity animation
  // rather than just a plain redraw whenever that mode is active.
  setPinnedPoints(pins) {
    this._pins = pins; // Array<{id, color, focused}>
    if (this._showPinnedOnly) this._animateTo(d => this._opacityTarget(d));
    else this._onChange();
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
    this._animateTo(d => this._opacityTarget(d));
  }

  _baseOpacity(d) {
    if (this._familyModeEnabled) return 1; // every point has a real family color, nothing to dim
    if (this._selectedLangs.size === 0) return 1;
    return this._selectedLangs.has(d.lang) ? 1 : FILTERED_OUT_OPACITY;
  }

  _colorFor(d) {
    if (this._familyModeEnabled) return this._familyColorScale(d.family);
    return this._selectedLangs.has(d.lang) ? this._langColorScale(d.lang) : NEUTRAL_COLOR;
  }

  _isPinned(d) {
    return this._pins.some(p => p.id === d.id);
  }

  // Whether `d` is actively colored (as opposed to plain background gray)
  // — every point in family mode, or the language-filter's selected subset
  // otherwise.
  _isColored(d) {
    return this._familyModeEnabled || this._selectedLangs.has(d.lang);
  }

  _isolateActive() {
    return this._activeGroupKey !== null || this._showPinnedOnly;
  }

  // "Show pinned only" takes top precedence over a translations isolate
  // (narrows rather than unions) — matching DataTable's own show_selected
  // filter, which likewise narrows whatever rows the translations view
  // was already showing down to just the pinned ones, not the other way
  // around. With neither active, falls back to the base coloring-mode
  // opacity (family mode: always 1; language-filter mode: dim non-
  // selected languages).
  _opacityTarget(d) {
    if (this._showPinnedOnly) return this._isPinned(d) ? 1 : ISOLATED_OUT_OPACITY;
    if (this._activeGroupKey !== null) return d.groupKey === this._activeGroupKey ? 1 : ISOLATED_OUT_OPACITY;
    return this._baseOpacity(d);
  }

  // Whether `d` is currently faded to fully invisible by an active
  // isolate — used by ProjectionView to keep hit-testing off points the
  // user can't see (the quadtree still spatially indexes them).
  isIsolatedOut(d) {
    return this._isolateActive() && this._opacityTarget(d) === ISOLATED_OUT_OPACITY;
  }

  // fisheyeFocus, when active, is {x, y, radius, distortion} in the same
  // "local" (pre-zoom) coordinate space as baseXScale/baseYScale output —
  // see ProjectionView._updateFisheyeFocus for how it's computed.
  draw(ctx, margin, transform, hoveredId, fisheyeFocus) {
    ctx.save();
    ctx.translate(margin.left, margin.top);
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.k, transform.k);

    const positions = fisheyeFocus ? this._computeFisheyePositions(fisheyeFocus, transform) : null;

    // Three-tier z-order so canvas draw order (dataset order, not by any
    // notion of importance) never buries something the user should be
    // able to see: plain gray points first, colored ones next, pinned
    // points always last (a pin should win even over a colored point at
    // the same screen location).
    const background = [], colored = [], pinned = [];
    for (const d of this.points) {
      if (this._isPinned(d)) pinned.push(d);
      else if (this._isColored(d)) colored.push(d);
      else background.push(d);
    }
    for (const d of background) this._drawPoint(ctx, d, transform, hoveredId, positions);
    for (const d of colored) this._drawPoint(ctx, d, transform, hoveredId, positions);
    for (const d of pinned) this._drawPoint(ctx, d, transform, hoveredId, positions);

    for (const d of this._tokenOverlayEntries) {
      const [cx, cy] = positions ? positions.get(d.id) : [this.baseXScale(d.x), this.baseYScale(d.y)];
      const size = POINT_RADIUS / transform.k;
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

  // Radial distortion alone (_fisheyeOffset) can't stop two points from
  // still overlapping after magnification — it moves each point only as
  // a function of its own distance from the focus, blind to every other
  // point. This adds a second pass: after the radial distortion, points
  // that ended up within one render-diameter of each other are pushed
  // apart along the line between them, split evenly, for a few
  // iterations (a full physical simulation isn't needed — a handful of
  // passes visibly declutters without the cost or instability of solving
  // to convergence). Sentence points and token-overlay points share one
  // relaxation pass since both can visually collide under the lens; each
  // carries its own unique id, so one Map holds both without clashing.
  _computeFisheyePositions(fisheyeFocus, transform) {
    const positions = new Map();
    const affected = [];
    const allPoints = this._tokenOverlayEntries.length ? this.points.concat(this._tokenOverlayEntries) : this.points;
    for (const d of allPoints) {
      const cx0 = this.baseXScale(d.x), cy0 = this.baseYScale(d.y);
      const dx = cx0 - fisheyeFocus.x, dy = cy0 - fisheyeFocus.y;
      if (Math.sqrt(dx * dx + dy * dy) >= fisheyeFocus.radius) {
        positions.set(d.id, [cx0, cy0]);
        continue;
      }
      const [ox, oy] = _fisheyeOffset(dx, dy, fisheyeFocus.radius, fisheyeFocus.distortion);
      positions.set(d.id, [fisheyeFocus.x + ox, fisheyeFocus.y + oy]);
      affected.push(d.id);
    }

    const minDist = (2 * POINT_RADIUS) / transform.k;
    for (let iter = 0; iter < FISHEYE_RELAX_ITERATIONS; iter++) {
      for (let i = 0; i < affected.length; i++) {
        const pi = positions.get(affected[i]);
        for (let j = i + 1; j < affected.length; j++) {
          const pj = positions.get(affected[j]);
          const dx = pj[0] - pi[0], dy = pj[1] - pi[1];
          const dist = Math.sqrt(dx * dx + dy * dy) || 0.0001; // avoid a divide-by-zero if two points land exactly on top of each other
          if (dist >= minDist) continue;
          const push = (minDist - dist) / 2;
          const ux = dx / dist, uy = dy / dist;
          pi[0] -= ux * push; pi[1] -= uy * push;
          pj[0] += ux * push; pj[1] += uy * push;
        }
      }
    }
    return positions;
  }

  _drawPoint(ctx, d, transform, hoveredId, fisheyePositions) {
    const [cx, cy] = fisheyePositions ? fisheyePositions.get(d.id) : [this.baseXScale(d.x), this.baseYScale(d.y)];
    const hovered = d.id === hoveredId;
    ctx.beginPath();
    ctx.arc(cx, cy, POINT_RADIUS / transform.k, 0, 2 * Math.PI);
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
  // everything else out completely (unless it's also pinned while "show
  // pinned only" is active — see _opacityTarget). clear()/restoreActive()
  // below return to that base state rather than to full opacity.
  highlightGroup(groupKey) {
    this._activeGroupKey = groupKey;
    this._animateTo(d => this._opacityTarget(d));
  }

  clear() {
    this._activeGroupKey = null;
    this._animateTo(d => this._opacityTarget(d));
  }

  restoreActive() {
    if (this._activeGroupKey !== null) {
      this.highlightGroup(this._activeGroupKey);
    } else {
      this.clear();
    }
  }
}
