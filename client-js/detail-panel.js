// The sidebar detail panel shown when a point is selected — its language
// and an optional per-point token-embedding toggle. Translations aren't
// listed as text here; they're isolated (faded in/out) in the main view.

export class DetailPanel {
  constructor({ placeholderId, langId, tokensToggleId }) {
    this.placeholder = document.getElementById(placeholderId);
    this.langEl = document.getElementById(langId);
    this.tokensToggle = document.getElementById(tokensToggleId);
  }

  show(point) {
    this.placeholder.style.display = 'none';

    this.langEl.style.display = 'block';
    this.langEl.textContent = point.lang;

    this.tokensToggle.style.display = point.hasTokens ? 'block' : 'none';
    this.setTokensShown(false);
  }

  clear() {
    this.placeholder.style.display = 'block';
    this.langEl.style.display = 'none';
    this.langEl.textContent = '';
    this.tokensToggle.style.display = 'none';
  }

  setTokensShown(shown) {
    this.tokensToggle.textContent = shown ? 'Hide tokens' : 'Show tokens';
  }
}
