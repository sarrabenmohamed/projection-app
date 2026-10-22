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
  // for tsne — any key can be omitted/undefined to let the backend pick its default.
  fetchProjection(method, metric, hyperparams = {}) {
    const params = new URLSearchParams({ method, metric });
    for (const [k, v] of Object.entries(hyperparams)) {
      if (v !== undefined && v !== null && v !== '') params.set(k, v);
    }
    return this._request(`/project?${params}`);
  }

  // umapParams is {n_neighbors, min_dist} — must match whatever the main
  // projection is currently using so the token vectors land in the same
  // fitted UMAP space (see server.py's _get_umap_reducer).
  fetchTokenProjection(id, metric, umapParams = {}) {
    const params = new URLSearchParams({ metric });
    for (const [k, v] of Object.entries(umapParams)) {
      if (v !== undefined && v !== null && v !== '') params.set(k, v);
    }
    return this._request(`/project/tokens/${encodeURIComponent(id)}?${params}`);
  }

  fetchNeighbors(id, k, metric) {
    return this._request(`/neighbors/${encodeURIComponent(id)}?k=${k}&metric=${metric}`);
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

  postDataset(raw) {
    return this._request('/dataset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(raw),
    });
  }

  // config is {lang_codes, n_sentences, model, pooling}; a first build can take minutes.
  loadFlores(config) {
    return this._request('/dataset/flores', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
  }

  async _request(path, options) {
    let res;
    try {
      res = await fetch(`${this.baseUrl}${path}`, options);
    } catch {
      throw new Error(`Can't reach the backend (${this.baseUrl || window.location.origin}) — start it with "make run".`);
    }
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `${res.status} ${res.statusText}`);
    }
    return res.json();
  }
}
