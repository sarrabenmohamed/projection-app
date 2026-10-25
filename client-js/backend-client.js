// Talks to server.py. Centralizes the "network unreachable" vs "backend
// rejected it" error handling in one place instead of duplicating a
// try/fetch/catch block at every call site.

export class BackendClient {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
  }

  fetchDatasetInfo() {
    return this._request('/dataset/info');
  }

  fetchPoints() {
    return this._request('/points');
  }

  // hyperparams is {n_neighbors, min_dist} for umap or {perplexity, learning_rate}
  // for tsne, plus n_components (2 or 3) — any key can be omitted/undefined to let the backend pick its default.
  // refit: true fits UMAP on every point again; otherwise points added since the
  // last fit are placed into its layout (the response flags them `overlaid`).
  fetchProjection(method, metric, hyperparams = {}) {
    const params = new URLSearchParams({ method, metric });
    for (const [k, v] of Object.entries(hyperparams)) {
      if (v !== undefined && v !== null && v !== '') params.set(k, v);
    }
    return this._request(`/project?${params}`);
  }

  // umapParams is {n_neighbors, min_dist, n_components} — must match whatever the main
  // projection is currently using so the token vectors land in the same
  // fitted UMAP space (see backend/projection.py).
  fetchTokenProjection(id, metric, umapParams = {}) {
    const params = new URLSearchParams({ metric });
    for (const [k, v] of Object.entries(umapParams)) {
      if (v !== undefined && v !== null && v !== '') params.set(k, v);
    }
    return this._request(`/project/tokens/${encodeURIComponent(id)}?${params}`);
  }

  fetchStats(ids) {
    return this._request('/stats', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids }),
    });
  }

  // Raw file text; `format` is json, jsonl or csv (parsed server-side).
  uploadDataset(text, format) {
    return this._request(`/dataset/upload?format=${encodeURIComponent(format)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain; charset=utf-8' },
      body: text,
    });
  }

  // config is {lang_codes, n_sentences, model, pooling}; a first build can take minutes.
  // Aborting `signal` stops waiting; cancelFlores() stops the server's work.
  loadFlores(config, signal) {
    return this._request('/dataset/flores', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
      signal,
    });
  }

  // Changes the loaded FLORES-200 dataset's languages and sentence count;
  // config is {lang_codes, n_sentences}. Only what's new is embedded, and
  // the model and pooling stay those of the loaded dataset.
  updateFlores(config, signal) {
    return this._request('/dataset/flores', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
      signal,
    });
  }

  cancelFlores() {
    return this._request('/dataset/flores/cancel', { method: 'POST' });
  }

  async _request(path, options) {
    let res;
    try {
      res = await fetch(`${this.baseUrl}${path}`, options);
    } catch (err) {
      if (err.name === 'AbortError') throw err;
      throw new Error(`Can't reach the backend (${this.baseUrl || window.location.origin}) — start it with "make run".`);
    }
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `${res.status} ${res.statusText}`);
    }
    return res.json();
  }
}
