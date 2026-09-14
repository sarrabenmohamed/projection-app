// Reads/writes the app's shareable subset of state (method, metric,
// label-by, and single-point selection) to the URL's query string via
// history.replaceState — lets a "Copy link" button hand off the current
// view without a page reload or extra browser-history entries.

const KEYS = ['method', 'metric', 'labelBy', 'ids'];

export function readUrlState() {
  const params = new URLSearchParams(window.location.search);
  const state = {};
  for (const key of KEYS) {
    if (params.has(key)) state[key] = params.get(key);
  }
  return state;
}

export function writeUrlState(state) {
  const params = new URLSearchParams();
  for (const key of KEYS) {
    const value = state[key];
    if (value !== undefined && value !== null && value !== '') params.set(key, value);
  }
  const qs = params.toString();
  const url = qs ? `${window.location.pathname}?${qs}` : window.location.pathname;
  window.history.replaceState(null, '', url);
}
