// Positioned label that follows the mouse over a hovered point. Doesn't know
// points exist — just text at a place.

export class Tooltip {
  constructor(containerEl) {
    this.el = document.createElement('div');
    this.el.className = 'point-tooltip';
    // TODO: make consistent, where the elements are appended as children to root dom element
    containerEl.appendChild(this.el);
  }

  show(x, y, text) {
    this.el.textContent = text;
    this.el.style.left = `${x + 12}px`;
    this.el.style.top = `${y + 12}px`;
    this.el.style.display = 'block';
  }

  hide() {
    this.el.style.display = 'none';
  }
}
