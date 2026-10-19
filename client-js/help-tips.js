// Instant hover help for elements carrying a data-tip attribute. Native
// title tooltips appear only after a ~1s delay (and not at all on some
// setups), which made the parameter help easy to miss.

export function installTooltips() {
  const tip = document.createElement('div');
  tip.id = 'ui-tooltip';
  document.body.appendChild(tip);

  const hide = () => { tip.classList.remove('visible'); };

  document.addEventListener('mouseover', e => {
    const el = e.target.closest('[data-tip]');
    if (!el) { hide(); return; }
    tip.textContent = el.dataset.tip;
    tip.classList.add('visible');
    const rect = el.getBoundingClientRect();
    const tipRect = tip.getBoundingClientRect();
    const left = Math.min(rect.right + 8, window.innerWidth - tipRect.width - 8);
    const top = Math.min(Math.max(8, rect.top), window.innerHeight - tipRect.height - 8);
    tip.style.left = `${Math.max(8, left)}px`;
    tip.style.top = `${top}px`;
  });
  document.addEventListener('mouseleave', hide);
  document.addEventListener('scroll', hide, true);
}
