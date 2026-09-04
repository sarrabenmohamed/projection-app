// The sidebar detail panel shown when a point is selected alongside its
// translations and an optional per-point token-embedding toggle.

export class DetailPanel {
  constructor({ placeholderId, langId, textId, translationsId, tokensToggleId }) {
    this.placeholder = document.getElementById(placeholderId);
    this.langEl = document.getElementById(langId);
    this.textEl = document.getElementById(textId);
    this.transEl = document.getElementById(translationsId);
    this.tokensToggle = document.getElementById(tokensToggleId);
  }

  show(point, translations) {
    this.placeholder.style.display = 'none';

    this.langEl.style.display = 'block';
    this.langEl.textContent = point.lang;

    this.textEl.style.display = 'block';
    this.textEl.textContent = point.text;

    if (translations.length === 0) {
      this.transEl.style.display = 'none';
      this.transEl.innerHTML = '';
    } else {
      this.transEl.style.display = 'block';
      this.transEl.innerHTML = '<b>Translations</b>' +
        translations.map(p => `<div><i>${p.lang}</i>: ${p.text}</div>`).join('');
    }

    this.tokensToggle.style.display = point.hasTokens ? 'block' : 'none';
    this.setTokensShown(false);
  }

  clear() {
    this.placeholder.style.display = 'block';
    [this.langEl, this.textEl, this.transEl].forEach(el => {
      el.style.display = 'none';
      el.textContent = '';
    });
    this.tokensToggle.style.display = 'none';
  }

  setTokensShown(shown) {
    this.tokensToggle.textContent = shown ? 'Hide tokens' : 'Show tokens';
  }
}
