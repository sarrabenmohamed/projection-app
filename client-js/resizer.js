// Drag handle that resizes one panel. Pointer capture keeps the drag alive
// outside the handle, and the (expensive) onResize callback is coalesced to
// one call per animation frame so dragging stays smooth.
//
// sign: +1 when the panel grows as the pointer moves right/down (panel is
// before the handle), -1 when it grows as the pointer moves left/up (panel is
// after the handle).
export function makeResizable({ handle, panel, storageKey, axis, sign, min, max = Infinity, onResize }) {
  const horizontal = axis === 'x';
  const prop = horizontal ? 'width' : 'height';

  const apply = size => {
    const px = `${Math.round(Math.min(max, Math.max(min, size)))}px`;
    panel.style[prop] = px;
    if (horizontal) panel.style.minWidth = px;
  };

  const saved = parseInt(localStorage.getItem(storageKey), 10);
  if (Number.isFinite(saved) && saved > 0) apply(saved);

  let start = 0, startSize = 0, framePending = false;
  const notify = () => {
    if (framePending) return;
    framePending = true;
    requestAnimationFrame(() => {
      framePending = false;
      onResize?.();
    });
  };

  handle.addEventListener('pointerdown', e => {
    start = horizontal ? e.clientX : e.clientY;
    startSize = panel.getBoundingClientRect()[prop];
    handle.setPointerCapture(e.pointerId);
    handle.classList.add('dragging');
    document.body.classList.add(horizontal ? 'resizing-x' : 'resizing-y');
    e.preventDefault();
  });

  handle.addEventListener('pointermove', e => {
    if (!handle.hasPointerCapture(e.pointerId)) return;
    const delta = (horizontal ? e.clientX : e.clientY) - start;
    apply(startSize + sign * delta);
    notify();
  });

  const finish = e => {
    if (!handle.hasPointerCapture(e.pointerId)) return;
    handle.releasePointerCapture(e.pointerId);
    handle.classList.remove('dragging');
    document.body.classList.remove('resizing-x', 'resizing-y');
    localStorage.setItem(storageKey, parseInt(panel.style[prop], 10));
    onResize?.();
  };
  handle.addEventListener('pointerup', finish);
  handle.addEventListener('pointercancel', finish);
}
