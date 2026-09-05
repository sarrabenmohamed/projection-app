// Talks to server.py. Centralizes the "network unreachable" vs "backend
// rejected it" error handling in one place instead of duplicating a
// try/fetch/catch block at every call site.

export class BackendClient {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
  }

  fetchPoints() {
    return this._request('/points');
  }

  fetchProjection(method, metric) {
    return this._request(`/project?method=${method}&metric=${metric}`);
  }

  fetchTokenProjection(id, metric) {
    return this._request(`/project/tokens/${encodeURIComponent(id)}?metric=${metric}`);
  }

  postDataset(raw) {
    return this._request('/dataset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(raw),
    });
  }

  async _request(path, options) {
    let res;
    try {
      res = await fetch(`${this.baseUrl}${path}`, options);
    } catch {
      throw new Error(`Can't reach the backend (${this.baseUrl}) — run "uvicorn server:app --port 8500" first.`);
    }
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `${res.status} ${res.statusText}`);
    }
    return res.json();
  }
}
