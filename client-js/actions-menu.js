// Shared "Actions ▾" dropdown for one pinned point — currently used only
// by DataTable, but built generic (an `ownerKey` scopes which container's
// menu is open) in case another view reuses it later. Every render reads
// current toggle state from `detail` and calls back into App, which owns
// _detailState as the single source of truth. Which menu is *open*,
// however, is tracked here as module-level state (not a DOM node's CSS
// class) — DataTable does a full tbody teardown-and-rebuild on every
// toggle (see data-table.js's _render()), which would otherwise destroy
// the "open" class the instant a checklist line inside this very menu is
// clicked.
let openMenuId = null; // at most one Actions menu open per container (keyed by `${ownerKey}:${point.id}`) at a time

export function renderActionsMenu(point, detail, canTranslate, { onToggleTranslations, onToggleTokens }, ownerKey) {
  const wrap = document.createElement('span');
  wrap.className = 'actions-menu';

  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'actions-menu-btn';
  btn.textContent = 'Actions ▾';
  wrap.appendChild(btn);

  const menu = document.createElement('div');
  menu.className = 'actions-menu-dropdown';
  wrap.appendChild(menu);

  menu.appendChild(_menuLine({
    label: 'Translations',
    checked: detail.translations,
    disabled: !canTranslate,
    disabledTitle: 'Only available with a single point selected',
    onToggle: () => onToggleTranslations(point),
  }));

  if (point.hasTokens) {
    menu.appendChild(_menuLine({
      label: 'Tokens',
      checked: detail.tokens,
      disabled: false,
      onToggle: () => onToggleTokens(point),
    }));
  }

  const divider = document.createElement('div');
  divider.className = 'actions-menu-divider';
  menu.appendChild(divider);

  const configureLine = document.createElement('div');
  configureLine.className = 'actions-menu-line actions-menu-disabled';
  configureLine.textContent = 'Configure…';
  configureLine.setAttribute('aria-disabled', 'true');
  menu.appendChild(configureLine);

  const menuKey = `${ownerKey}:${point.id}`;

  if (openMenuId === menuKey) menu.classList.add('open');

  btn.addEventListener('click', e => {
    e.stopPropagation();
    openMenuId = (openMenuId === menuKey) ? null : menuKey;
    document.querySelectorAll('.actions-menu-dropdown.open').forEach(m => m.classList.remove('open'));
    if (openMenuId === menuKey) menu.classList.add('open');
  });

  return wrap;
}

function _menuLine({ label, checked, disabled, disabledTitle, onToggle }) {
  const line = document.createElement('div');
  line.className = 'actions-menu-line' + (disabled ? ' actions-menu-disabled' : '');
  if (disabled && disabledTitle) line.title = disabledTitle;

  const check = document.createElement('span');
  check.className = 'actions-menu-check';
  check.textContent = checked ? '☑' : '☐';
  line.appendChild(check);

  const text = document.createElement('span');
  text.textContent = label;
  line.appendChild(text);

  if (!disabled) {
    line.addEventListener('click', e => { e.stopPropagation(); onToggle(); });
  }
  return line;
}

// Closes any open Actions dropdown on an outside click or Escape. Call
// once at app startup (App.init()), not once per rendered menu.
export function installActionsMenuOutsideClickHandler() {
  document.addEventListener('click', () => {
    openMenuId = null;
    document.querySelectorAll('.actions-menu-dropdown.open').forEach(m => m.classList.remove('open'));
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      openMenuId = null;
      document.querySelectorAll('.actions-menu-dropdown.open').forEach(m => m.classList.remove('open'));
    }
  });
}
