// Owns the point data, their colors and per-point opacity, and paints them
// onto a given canvas context. Hides the whole d3.timer-driven opacity
// animation behind setLangFilter/setShowPinnedOnly.
//
// Three coloring modes, mutually exclusive (see setColorMode):
// - 'language' (default): every language gets its own hue, always on.
//   Past a couple of dozen languages the palette repeats; the legend and
//   hover labels still say which is which.
// - 'family': every point is colored by its language's linguistic family.
//   `family` is a field the backend already attaches to every point
//   (backend/languages.py's point_family()), not something this class computes.
// - 'custom': the language-filter behavior. Every point is neutral gray
//   by default; only languages explicitly selected via the language filter
//   get a real color, and everything else dims to context.
//
// Points are either 2D ({x, y}) or 3D ({x, y, z}). A 3D set is kept as raw
// coordinates in `p3` and rotated (rotateBy) into the same screen-plane
// `x`/`y` the 2D pipeline already uses for scales, fisheye and hit-testing,
// plus a `depth` in [-1, 1] (1 = nearest) that drives size, opacity and draw
// order.
//
// Pinned points get a colored ring; `pins` is the full current list every
// time (not incremental), computed by App from its selection set — so a
// point that stops being pinned just isn't in the next call's array.

const NEUTRAL_COLOR = '#bbb';
const FILTERED_OUT_OPACITY = 0.3;
const ISOLATED_OUT_OPACITY = 0.3;
// d3.schemeTableau10 with its trailing gray (#bab0ac) swapped for a cyan —
// that gray reads as "dimmed/background" (see NEUTRAL_COLOR above), so a
// family that landed on it was indistinguishable from an isolated-out point.
const FAMILY_COLORS = [...d3.schemeTableau10.slice(0, 9), '#17becf'];
// One hue per language: the family palette, then Paired and Dark2 for variety.
const LANG_COLORS = [...FAMILY_COLORS, ...d3.schemePaired, ...d3.schemeDark2];
const DEPTH_SIZE_RANGE = 0.35;  // 3D: nearest points drawn up to 35% bigger, farthest 35% smaller
const HULL_GROWTH = 1.08;       // 3D: how far past the drawn points a press still rotates
const DEPTH_FADE = 0.45;        // 3D: farthest points lose up to 45% opacity
const DEFAULT_YAW = 0.6;
const DEFAULT_PITCH = 0.35;
const POINT_RADIUS = 5; // local-unit render radius before the 1/transform.k scale-compensation
const FISHEYE_RELAX_ITERATIONS = 4;

// Two interchangeable circular "magnifying glass" distortions — both take
// a point's offset (dx, dy) from the lens focus and return a new offset,
// pushed further out near the center (k > 1) and compressing back to the
// real position at the edge (k -> 1 exactly at d = radius, so there's no
// visible seam either way). App picks which one via the sidebar's
// "Fisheye style" select, threaded through as fisheyeFocus.style.
//
// Neither alone can prevent two points from still overlapping after
// magnification: two points equidistant from the focus get pushed by the
// identical factor k, so they stay exactly as overlapped, just farther
// from center. See _computeFisheyePositions for the relaxation pass that
// fixes that — it runs on top of whichever style is selected.

// This project's own original formula: a simple rational falloff.
function _fisheyeOffsetGeometric(dx, dy, radius, distortion) {
  const d = Math.sqrt(dx * dx + dy * dy);
  if (d >= radius) return [dx, dy];
  const k = (radius * (distortion + 1)) / (distortion * d + radius);
  return [dx * k, dy * k];
}

// Ported from the classic d3-plugins `d3.fisheye.circular()` (Bostock;
// MIT-licensed, archived at
// https://github.com/d3/d3-plugins/blob/master/fisheye/fisheye.js), which
// predates D3's v4+ modularization and was never folded into core D3, so
// it isn't available from the `d3@7` bundle this project loads and had
// to be ported by hand instead of imported. Uses an exponential falloff
// rather than the geometric formula's rational one.
function _fisheyeOffsetD3(dx, dy, radius, distortion) {
  const d = Math.sqrt(dx * dx + dy * dy);
  if (!d || d >= radius) return [dx, dy];
  let k0 = Math.exp(distortion);
  k0 = (k0 / (k0 - 1)) * radius;
  const k1 = distortion / radius;
  const k = ((k0 * (1 - Math.exp(-d * k1))) / d) * 0.75 + 0.25;
  return [dx * k, dy * k];
}

function _fisheyeOffset(dx, dy, radius, distortion, style) {
  return style === 'geometric'
    ? _fisheyeOffsetGeometric(dx, dy, radius, distortion)
    : _fisheyeOffsetD3(dx, dy, radius, distortion);
}

// An ordinal color scale over `values` (sorted) whose colors survive the set
// of values changing, so editing a dataset never recolors the points already
// on screen: a value colored before keeps its color (even if it was gone for
// a while), and new values take the next unused ones. `indexes` (value ->
// palette index) is what to pass back as `previous` next time; with nothing
// previous this is just d3.scaleOrdinal(sortedValues, palette).
function stableColorScale(values, palette, previous = new Map()) {
  const indexes = new Map(previous);
  let next = previous.size ? Math.max(...previous.values()) + 1 : 0;
  const sorted = [...values].sort();
  for (const value of sorted) {
    if (!indexes.has(value)) indexes.set(value, next++);
  }
  const scale = d3.scaleOrdinal(sorted, sorted.map(value => palette[indexes.get(value) % palette.length]));
  return { scale, indexes };
}

// Token diamonds are larger than sentence circles, a lightened pin color
// with a dark outline, so they read as a different kind of mark.
const TOKEN_SIZE_FACTOR = 1.6;
const TOKEN_LIGHTEN = 0.35;

export class PointLayer {
  constructor(onChange) {
    this._onChange = onChange; // called after every opacity update, so the owner redraws
    this.points = [];
    this.baseXScale = null;
    this.baseYScale = null;
    this._timer = null;
    this._selectedLangs = new Set();
    this._langColorScale = null;
    this._tokenOverlayEntries = [];
    this._pinsById = new Map();
    this._colorMode = 'language'; // a view preference, not per-dataset — setPoints() below doesn't reset it
    this._familyColorScale = null;
    this._languageColorScale = null;
    this._familyColorIndexes = new Map();
    this._languageColorIndexes = new Map();
    this._showPinnedOnly = false;
    this.dims = 2;
    this._yaw = DEFAULT_YAW;     // view preferences too: a reprojection keeps the rotation
    this._pitch = DEFAULT_PITCH;
    this._center = [0, 0, 0];
    this.radius = 1;             // 3D: farthest distance from _center, sets the screen scale
  }

  // keepColors: the points are an edited version of the current ones, so
  // languages and families that were already colored keep their colors.
  // Scales are set separately because a 3D set's screen scale depends on the
  // radius that setPoints() computes.
  // keepSphere (3D): keep the previous center and radius so the cloud isn't rescaled.
  setPoints(points, { keepColors = false, keepSphere = false } = {}) {
    const sphere = { center: this._center, radius: this.radius };
    this._selectedLangs = new Set();
    this._langColorScale = null;
    this._tokenOverlayEntries = [];
    this._pinsById = new Map();
    this._showPinnedOnly = false;
    this._displayPositions = null;
    this.points = points.map(p => ({ ...p, opacity: 1 }));
    const families = stableColorScale(new Set(this.points.map(p => p.family)), FAMILY_COLORS, keepColors ? this._familyColorIndexes : undefined);
    this._familyColorScale = families.scale;
    this._familyColorIndexes = families.indexes;
    const langs = stableColorScale(new Set(this.points.map(p => p.lang)), LANG_COLORS, keepColors ? this._languageColorIndexes : undefined);
    this._languageColorScale = langs.scale;
    this._languageColorIndexes = langs.indexes;

    this.dims = this.points.length && this.points[0].z !== undefined ? 3 : 2;
    if (this.dims === 3) {
      for (const d of this.points) d.p3 = [d.x, d.y, d.z];
      this._fitSphere();
      if (keepSphere) [this._center, this.radius] = [sphere.center, sphere.radius];
      this._applyRotation();
    }
  }

  setScales(baseXScale, baseYScale) {
    this.baseXScale = baseXScale;
    this.baseYScale = baseYScale;
  }

  // The color modes are mutually exclusive views of color — switching
  // re-triggers whatever opacity target is currently active so the fade
  // animates consistently with every other opacity change.
  setColorMode(mode) {
    this._colorMode = mode;
    this._animateTo(d => this._opacityTarget(d));
  }

  // The label -> color mapping the current mode uses (only the values
  // present in this dataset, not the full palette) — used by App to render
  // a legend. Empty in 'custom' mode, where the language chips are the key.
  getColorLegend() {
    const scale = { language: this._languageColorScale, family: this._familyColorScale }[this._colorMode];
    if (!scale) return [];
    return scale.domain().map(label => ({ label, color: scale(label) }));
  }

  // ── 3D rotation ──────────────────────────────────────────────────────────
  _fitSphere() {
    const n = this.points.length;
    this._center = [0, 1, 2].map(i => this.points.reduce((sum, d) => sum + d.p3[i], 0) / n);
    this.radius = Math.max(1e-9, ...this.points.map(d =>
      Math.hypot(d.p3[0] - this._center[0], d.p3[1] - this._center[1], d.p3[2] - this._center[2])));
  }

  // Rotates d.p3 about the center (yaw about the vertical axis, then pitch
  // about the horizontal one) into the screen-plane x/y and depth.
  _project(d) {
    const { x, y, depth } = this._rotate(d.p3);
    d.x = x;
    d.y = y;
    d.depth = depth;
  }

  _rotate(p3) {
    const [cx, cy, cz] = this._center;
    const x = p3[0] - cx, y = p3[1] - cy, z = p3[2] - cz;
    const cosY = Math.cos(this._yaw), sinY = Math.sin(this._yaw);
    const cosP = Math.cos(this._pitch), sinP = Math.sin(this._pitch);
    const x1 = cosY * x + sinY * z;
    const z1 = -sinY * x + cosY * z;
    return { x: x1, y: cosP * y - sinP * z1, depth: (sinP * y + cosP * z1) / this.radius };
  }

  _bounds() {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const d of this.points) for (let i = 0; i < 3; i++) {
      lo[i] = Math.min(lo[i], d.p3[i]);
      hi[i] = Math.max(hi[i], d.p3[i]);
    }
    return { lo, hi };
  }

  // Screen-plane (pre-zoom) position and depth of a raw 3D coordinate.
  _screenOf(p3) {
    const r = this._rotate(p3);
    return { sx: this.baseXScale(r.x), sy: this.baseYScale(r.y), depth: r.depth };
  }

  // Outline of what is drawn in 3D (points and axes), slightly grown, for deciding
  // whether a press is on the cloud (rotate) or beside it (pan). Pre-zoom space.
  boxHull() {
    if (this.dims !== 3 || !this.points.length || !this.baseXScale) return null;
    const { lo, hi } = this._bounds();
    const shown = this.points.map(d => [this.baseXScale(d.x), this.baseYScale(d.y)]);
    for (const [a, b] of this._axisEnds(lo, hi)) shown.push([a.sx, a.sy], [b.sx, b.sy]);
    const hull = d3.polygonHull(shown);
    if (!hull) return null;
    const [cx, cy] = d3.polygonCentroid(hull);
    return hull.map(([x, y]) => [cx + (x - cx) * HULL_GROWTH, cy + (y - cy) * HULL_GROWTH]);
  }

  _axisEnds(lo, hi) {
    return [0, 1, 2].map(i => {
      const end = [lo[0], lo[1], lo[2]];
      end[i] = hi[i];
      return [this._screenOf([lo[0], lo[1], lo[2]]), this._screenOf(end)];
    });
  }

  _drawAxes(ctx, transform) {
    const { lo, hi } = this._bounds();
    const k = transform.k;
    ctx.save();
    ctx.lineWidth = 2 / k;
    ['#d93025', '#188038', '#1a73e8'].forEach((color, i) => {
      const [a, b] = this._axisEnds(lo, hi)[i];
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(a.sx, a.sy);
      ctx.lineTo(b.sx, b.sy);
      ctx.stroke();
    });
    ctx.restore();
  }

  _applyRotation() {
    for (const d of this.points) this._project(d);
    for (const d of this._tokenOverlayEntries) if (d.p3) this._project(d);
  }

  rotateBy(dYaw, dPitch) {
    if (this.dims !== 3) return;
    this._yaw += dYaw;
    this._pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this._pitch + dPitch));
    this._applyRotation();
  }

  resetRotation() {
    this._yaw = DEFAULT_YAW;
    this._pitch = DEFAULT_PITCH;
    if (this.dims === 3) this._applyRotation();
  }

  // Near points are drawn bigger and more opaque than far ones (3D only).
  _depthScale(d) {
    return this.dims === 3 ? 1 + DEPTH_SIZE_RANGE * d.depth : 1;
  }

  _depthAlpha(d) {
    return this.dims === 3 ? 1 - DEPTH_FADE * (1 - d.depth) / 2 : 1;
  }

  // Isolates the canvas to just the pinned points.
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
    // Keyed by id: select-all can pin thousands of points, and draw() looks
    // every point up on every frame.
    this._pinsById = new Map(pins.map(p => [p.id, p])); // Array<{id, color, focused}>
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
      const pin = this._pinsById.get(pinId);
      // Lighter than the pin's own color, so a token never blends into the
      // (same-palette) sentence points around it.
      const color = d3.interpolateRgb(pin ? pin.color : '#e63946', '#ffffff')(TOKEN_LIGHTEN);
      for (const point of points) {
        const entry = { ...point, pinId, color };
        if (this.dims === 3 && point.z !== undefined) {
          entry.p3 = [point.x, point.y, point.z];
          this._project(entry);
        }
        this._tokenOverlayEntries.push(entry);
      }
    }
    this._onChange();
  }

  tokenOverlayPointAt(x, y, radius) {
    for (const d of this._tokenOverlayEntries) {
      const [px, py] = this._displayedPosition(d);
      if (Math.hypot(px - x, py - y) <= radius) return d;
    }
    return undefined;
  }

  // Where d was last drawn: the lens-displaced position while the fisheye is
  // on, its plain position otherwise.
  _displayedPosition(d) {
    return this._displayPositions?.get(d.id) || [this.baseXScale(d.x), this.baseYScale(d.y)];
  }

  // Nearest visible point to (x, y) within `radius`, using the positions the
  // fisheye actually drew (the quadtree only knows the undistorted ones).
  // Returns undefined when the lens isn't active, so callers fall back to it.
  pointAtDisplayed(x, y, radius) {
    if (!this._displayPositions) return undefined;
    let best, bestDist = radius;
    for (const d of this.points) {
      if (this.isIsolatedOut(d)) continue;
      const [px, py] = this._displayedPosition(d);
      const dist = Math.hypot(px - x, py - y);
      if (dist <= bestDist) { best = d; bestDist = dist; }
    }
    return best;
  }

  get fisheyeActive() { return this._displayPositions != null; }

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
    if (this._colorMode !== 'custom') return 1; // every point has a real color, nothing to dim
    if (this._selectedLangs.size === 0) return 1;
    return this._selectedLangs.has(d.lang) ? 1 : FILTERED_OUT_OPACITY;
  }

  _colorFor(d) {
    if (this.isIsolatedOut(d)) return NEUTRAL_COLOR;
    if (this._colorMode === 'family') return this._familyColorScale(d.family);
    if (this._colorMode === 'language') return this._languageColorScale(d.lang);
    return this._selectedLangs.has(d.lang) ? this._langColorScale(d.lang) : NEUTRAL_COLOR;
  }

  _isPinned(d) {
    return this._pinsById.has(d.id);
  }

  // Whether `d` is actively colored (as opposed to plain background gray)
  // — every point in language/family mode, or the language-filter's
  // selected subset in custom mode.
  _isColored(d) {
    if (this.isIsolatedOut(d)) return false;
    return this._colorMode !== 'custom' || this._selectedLangs.has(d.lang);
  }

  // Whether a box-select may take `d`: not faded out by the isolate, and,
  // when the language filter is narrowing the view, one of its languages.
  _isSelectable(d) {
    if (this.isIsolatedOut(d)) return false;
    return this._colorMode !== 'custom' || this._selectedLangs.size === 0 || this._selectedLangs.has(d.lang);
  }

  // Points whose drawn position lies inside the rectangle, given in the
  // same local (pre-zoom) coordinates as baseXScale/baseYScale output.
  pointsInRect(x0, y0, x1, y1) {
    return this.points.filter(d => {
      if (!this._isSelectable(d)) return false;
      const [px, py] = this._displayedPosition(d);
      return px >= x0 && px <= x1 && py >= y0 && py <= y1;
    });
  }

  _isolateActive() {
    return this._showPinnedOnly;
  }

  // With "show pinned only" on, everything unpinned is faded out; otherwise
  // the base coloring-mode opacity applies (language/family mode: always 1;
  // custom mode: dim non-selected languages).
  _opacityTarget(d) {
    if (this._showPinnedOnly) return this._isPinned(d) ? 1 : ISOLATED_OUT_OPACITY;
    return this._baseOpacity(d);
  }

  // Whether `d` is currently faded to background gray by an active
  // isolate — used by ProjectionView to keep hit-testing off points the
  // user isn't focused on (the quadtree still spatially indexes them),
  // and by _colorFor/_isColored to render them as dimmed background
  // rather than in their normal color.
  isIsolatedOut(d) {
    return this._isolateActive() && this._opacityTarget(d) === ISOLATED_OUT_OPACITY;
  }

  // fisheyeFocus, when active, is {x, y, radius, distortion} in the same
  // "local" (pre-zoom) coordinate space as baseXScale/baseYScale output —
  // see ProjectionView._fisheyeFocus for how it's computed.
  draw(ctx, margin, transform, hoveredId, fisheyeFocus) {
    ctx.save();
    ctx.translate(margin.left, margin.top);
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.k, transform.k);

    if (this.dims === 3) this._drawAxes(ctx, transform);

    const positions = fisheyeFocus ? this._computeFisheyePositions(fisheyeFocus, transform) : null;
    this._displayPositions = positions;

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
    if (this.dims === 3) for (const tier of [background, colored, pinned]) tier.sort((a, b) => a.depth - b.depth);
    for (const d of background) this._drawPoint(ctx, d, transform, hoveredId, positions);
    for (const d of colored) this._drawPoint(ctx, d, transform, hoveredId, positions);
    for (const d of pinned) this._drawPoint(ctx, d, transform, hoveredId, positions);

    for (const d of this._tokenOverlayEntries) {
      const [cx, cy] = positions ? positions.get(d.id) : [this.baseXScale(d.x), this.baseYScale(d.y)];
      const size = (TOKEN_SIZE_FACTOR * POINT_RADIUS * this._depthScale(d)) / transform.k;
      ctx.beginPath();
      ctx.moveTo(cx, cy - size);
      ctx.lineTo(cx + size, cy);
      ctx.lineTo(cx, cy + size);
      ctx.lineTo(cx - size, cy);
      ctx.closePath();
      ctx.globalAlpha = 1;
      ctx.fillStyle = d.color;
      ctx.fill();
      ctx.lineWidth = 1.25 / transform.k;
      ctx.strokeStyle = '#222';
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
      const [ox, oy] = _fisheyeOffset(dx, dy, fisheyeFocus.radius, fisheyeFocus.distortion, fisheyeFocus.style);
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
    const scale = this._depthScale(d);
    const alpha = d.opacity * this._depthAlpha(d);
    ctx.arc(cx, cy, (POINT_RADIUS * scale) / transform.k, 0, 2 * Math.PI);
    ctx.globalAlpha = alpha * 0.8;
    ctx.fillStyle = this._colorFor(d);
    ctx.fill();
    ctx.globalAlpha = alpha;
    ctx.lineWidth = (hovered ? 1.5 : 0.5) / transform.k;
    ctx.strokeStyle = hovered ? '#333' : '#fff';
    ctx.stroke();
    const pin = this._pinsById.get(d.id);
    if (pin) {
      ctx.beginPath();
      ctx.arc(cx, cy, (8.5 * scale) / transform.k, 0, 2 * Math.PI);
      ctx.globalAlpha = alpha;
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
}
