// The "embed FLORES-200 sentences" form. Once a FLORES-200 dataset is loaded
// it has two modes:
//  - edit (the default): change the loaded dataset's languages and sentence
//    count. Only what is new gets embedded and the points already plotted
//    keep their positions. The model and pooling stay fixed — other
//    embeddings would mean a new projection.
//  - load: embed a different selection, model and pooling from scratch.
// An uploaded dataset has no FLORES-200 source to extend, so it only has
// "load". The panel reports what the user asked for via onSubmit({mode,
// config}); doing it (and showing progress) is the App's job.

const byId = id => document.getElementById(id);

export function parseLangs(text) {
  return [...new Set(text.split(/[\s,;]+/).filter(Boolean))];
}

// What editing `loaded` ({langs, n, ...}) into `langs` × `n` sentences would change.
export function describeEdit(loaded, langs, n) {
  const have = new Set(loaded.langs);
  const want = new Set(langs);
  const added = langs.filter(l => !have.has(l));
  const removed = loaded.langs.filter(l => !want.has(l));
  const sentences = Number.isInteger(n) && n !== loaded.n ? { from: loaded.n, to: n } : null;
  return { added, removed, sentences, changed: added.length > 0 || removed.length > 0 || sentences !== null };
}

function summarize(edit) {
  const list = codes => (codes.length > 4 ? `${codes.slice(0, 4).join(', ')} +${codes.length - 4} more` : codes.join(', '));
  const parts = [
    edit.added.length && `+ ${list(edit.added)}`,
    edit.removed.length && `− ${list(edit.removed)}`,
    edit.sentences && `sentences ${edit.sentences.from} → ${edit.sentences.to}`,
  ].filter(Boolean);
  return parts.length ? parts.join(' · ') : 'No changes yet.';
}

export class FloresPanel {
  constructor(onSubmit) {
    this._onSubmit = onSubmit;
    this._loaded = null;     // {langs, n, model, pooling} of the loaded FLORES-200 dataset; null for none/an upload
    this._signature = null;  // of _loaded, so the form is only reset when the dataset really changed
    this._mode = 'load';
    this._busy = false;

    this._langs = byId('flores-langs');
    this._n = byId('flores-n');
    this._model = byId('flores-model');
    this._pooling = byId('flores-pooling');
    this._button = byId('flores-load-button');
    this._modeBar = byId('flores-mode');
    this._diff = byId('flores-diff');
    this._error = byId('flores-error');

    for (const field of [this._langs, this._n, this._model, this._pooling]) field.addEventListener('input', () => this._render());
    for (const button of this._modeBar.querySelectorAll('button')) {
      button.addEventListener('click', () => { this._mode = button.dataset.mode; this._render(); });
    }
    this._button.addEventListener('click', () => this._submit());
    this._render();
  }

  // Call after every dataset load with GET /points' answer. The form starts
  // from the dataset's own settings whenever those change (a different
  // dataset, or one that was just edited), and is left alone otherwise (a
  // re-projection mustn't throw away what the user is typing).
  setDataset(meta) {
    if (meta.source?.kind !== 'flores') {
      this._loaded = null;
      this._signature = null;
      this._mode = 'load';
    } else {
      this._loaded = { langs: meta.lang_codes, n: meta.source.n_sentences, model: meta.model, pooling: meta.pooling };
      const signature = JSON.stringify(this._loaded);
      if (signature !== this._signature) {
        this._signature = signature;
        this._langs.value = this._loaded.langs.join(', ');
        this._n.value = this._loaded.n;
        this._mode = 'edit';
      }
    }
    this._render();
  }

  setBusy(busy) {
    this._busy = busy;
    this._render();
  }

  showError(message) {
    this._error.textContent = message;
    this._error.hidden = false;
  }

  clearError() {
    this._error.hidden = true;
  }

  get _editing() {
    return this._loaded !== null && this._mode === 'edit';
  }

  _request() {
    const lang_codes = parseLangs(this._langs.value);
    const n_sentences = parseInt(this._n.value, 10);
    return { lang_codes, n_sentences };
  }

  _submit() {
    const config = this._request();
    if (this._editing) {
      this._onSubmit({ mode: 'edit', config });
    } else {
      this._onSubmit({ mode: 'load', config: { ...config, model: this._model.value.trim(), pooling: this._pooling.value } });
    }
  }

  _render() {
    const editing = this._editing;
    this._modeBar.hidden = this._loaded === null;
    for (const button of this._modeBar.querySelectorAll('button')) {
      button.setAttribute('aria-pressed', String(button.dataset.mode === this._mode));
    }

    this._model.disabled = this._pooling.disabled = editing;
    if (editing) {
      this._model.value = this._loaded.model;
      this._pooling.value = this._loaded.pooling;
    }

    const { lang_codes, n_sentences } = this._request();
    const edit = editing ? describeEdit(this._loaded, lang_codes, n_sentences) : null;
    this._diff.hidden = !editing;
    if (editing) this._diff.textContent = summarize(edit);

    this._button.textContent = editing ? 'Update dataset' : 'Load FLORES-200';
    this._button.disabled = this._busy || (editing && !edit.changed);
  }
}
